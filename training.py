from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter


def move_to_device(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def _as_targets(batch: Any, output: Mapping[str, Any]) -> torch.Tensor:
    if "targets" in output and torch.is_tensor(output["targets"]):
        return output["targets"].long()
    if isinstance(batch, list):
        labels = [sample["label"] for sample in batch]
        return torch.as_tensor(labels, dtype=torch.long)
    if isinstance(batch, Mapping) and "label" in batch:
        return torch.as_tensor(batch["label"], dtype=torch.long)
    raise KeyError("Could not find targets in model output or batch.")


def _as_logits(output: Mapping[str, Any]) -> torch.Tensor:
    logits = output.get("logits")
    if not torch.is_tensor(logits):
        raise TypeError("model(batch) must return a dict with a tensor at key 'logits'.")
    return logits


def confusion_matrix_from_predictions(
    predictions: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    flat = targets.to(torch.long) * num_classes + predictions.to(torch.long)
    counts = torch.bincount(flat, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def compute_metrics_from_confusion(confusion: torch.Tensor) -> Dict[str, float]:
    confusion = confusion.to(torch.float64)
    total = confusion.sum().clamp(min=1.0)
    diagonal = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)

    recall = diagonal / support.clamp(min=1.0)
    precision = diagonal / predicted.clamp(min=1.0)
    f1_denom = precision + recall
    f1 = torch.where(f1_denom > 0, 2.0 * precision * recall / f1_denom, torch.zeros_like(f1_denom))

    weighted_f1 = (f1 * support).sum() / support.sum().clamp(min=1.0)
    metrics = {
        "wa": float(diagonal.sum().item() / total.item()),
        "ua": float(recall.mean().item()),
        "wf1": float(weighted_f1.item()),
        "macro_f1": float(f1.mean().item()),
    }
    return metrics


class ClassificationMetrics:
    def __init__(self, num_classes: Optional[int] = None) -> None:
        self.num_classes = num_classes
        self.confusion = torch.zeros(0, 0, dtype=torch.long)

    def reset(self) -> None:
        self.confusion = torch.zeros(0, 0, dtype=torch.long)

    def _ensure_size(self, num_classes: int) -> None:
        if self.confusion.numel() == 0:
            self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
            self.num_classes = num_classes
            return
        if self.confusion.shape[0] != num_classes:
            if self.confusion.shape[0] != 0 and self.confusion.shape[0] != num_classes:
                raise ValueError("Confusion matrix size does not match num_classes.")

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.ndim == 1:
            predictions = logits.to(torch.long)
            num_classes = int(max(predictions.max().item(), targets.max().item()) + 1)
        else:
            predictions = logits.argmax(dim=-1)
            num_classes = int(logits.shape[-1])
        if self.num_classes is not None:
            num_classes = self.num_classes
        self._ensure_size(num_classes)
        self.confusion += confusion_matrix_from_predictions(
            predictions.reshape(-1).cpu(), targets.reshape(-1).cpu(), num_classes
        )

    def compute(self) -> Dict[str, float]:
        if self.confusion.numel() == 0:
            return {"wa": 0.0, "ua": 0.0, "wf1": 0.0, "macro_f1": 0.0}
        return compute_metrics_from_confusion(self.confusion)

    def state_dict(self) -> Dict[str, Any]:
        return {"num_classes": self.num_classes, "confusion": self.confusion.clone()}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.num_classes = state_dict.get("num_classes")
        confusion = state_dict.get("confusion")
        if torch.is_tensor(confusion):
            self.confusion = confusion.clone()
        else:
            self.confusion = torch.as_tensor(confusion, dtype=torch.long)


@dataclass
class EMAGradientBalancer:
    momentum: float = 0.99
    epsilon: float = 1e-8
    min_weight: float = 0.1
    max_weight: float = 10.0

    def __post_init__(self) -> None:
        self.ema_norms: Dict[str, float] = {}
        self.last_weights: Dict[str, float] = {}
        self.steps: int = 0

    def _trainable_parameters(self, model: torch.nn.Module) -> list[torch.nn.Parameter]:
        return [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and not isinstance(parameter, UninitializedParameter)
        ]

    def _grad_norm(self, loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter]) -> float:
        if not parameters or not loss.requires_grad:
            return 0.0
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        total = 0.0
        for grad in grads:
            if grad is None:
                continue
            value = float(grad.detach().norm().item())
            total += value * value
        return total ** 0.5

    def weights(self, cls_loss: torch.Tensor, aux_losses: Mapping[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, float]:
        if not aux_losses:
            self.last_weights = {}
            return {}
        parameters = self._trainable_parameters(model)
        primary_norm = self._grad_norm(cls_loss, parameters)
        self.ema_norms["primary"] = primary_norm if "primary" not in self.ema_norms else (
            self.momentum * self.ema_norms["primary"] + (1.0 - self.momentum) * primary_norm
        )
        target_norm = max(self.ema_norms["primary"], self.epsilon)

        weights: Dict[str, float] = {}
        for name, loss in aux_losses.items():
            norm = self._grad_norm(loss, parameters)
            if name in self.ema_norms:
                self.ema_norms[name] = self.momentum * self.ema_norms[name] + (1.0 - self.momentum) * norm
            else:
                self.ema_norms[name] = norm
            weight = target_norm / max(self.ema_norms[name], self.epsilon)
            weight = max(self.min_weight, min(self.max_weight, weight))
            weights[name] = float(weight)
        self.last_weights = weights
        self.steps += 1
        return weights

    def state_dict(self) -> Dict[str, Any]:
        return {
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "ema_norms": dict(self.ema_norms),
            "last_weights": dict(self.last_weights),
            "steps": self.steps,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.momentum = float(state_dict.get("momentum", self.momentum))
        self.epsilon = float(state_dict.get("epsilon", self.epsilon))
        self.min_weight = float(state_dict.get("min_weight", self.min_weight))
        self.max_weight = float(state_dict.get("max_weight", self.max_weight))
        self.ema_norms = {str(key): float(value) for key, value in state_dict.get("ema_norms", {}).items()}
        self.last_weights = {str(key): float(value) for key, value in state_dict.get("last_weights", {}).items()}
        self.steps = int(state_dict.get("steps", self.steps))


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    balancer: Optional[EMAGradientBalancer] = None,
    epoch: int = 0,
    metrics: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if balancer is not None:
        payload["balancer"] = balancer.state_dict()
    if extra is not None:
        payload["extra"] = dict(extra)
    torch.save(payload, str(checkpoint_path))


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    balancer: Optional[EMAGradientBalancer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    try:
        payload = torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if balancer is not None and "balancer" in payload:
        balancer.load_state_dict(payload["balancer"])
    return payload


@dataclass
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device | str
    balancer: Optional[EMAGradientBalancer] = None
    grad_clip_norm: Optional[float] = None
    class_weights: Optional[torch.Tensor] = None
    label_smoothing: float = 0.0
    scheduler: Optional[Any] = None

    def _resolve_batch_targets(self, batch: Any, output: Mapping[str, Any]) -> torch.Tensor:
        return _as_targets(batch, output).to(self.device)

    def _loss_terms(self, output: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        terms = output.get("loss_terms") or {}
        if not isinstance(terms, Mapping):
            raise TypeError("model(batch) must return loss_terms as a mapping.")
        result: Dict[str, torch.Tensor] = {}
        for name in ("struct", "entropy", "consistency"):
            value = terms.get(name)
            if value is None:
                continue
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            result[name] = value
        return result

    def _step(self, batch: list[dict[str, Any]], training: bool) -> Dict[str, Any]:
        output = self.model(batch)
        logits = _as_logits(output)
        targets = self._resolve_batch_targets(batch, output)
        weight = self.class_weights.to(self.device) if self.class_weights is not None else None
        cls_loss = F.cross_entropy(logits, targets, weight=weight, label_smoothing=self.label_smoothing)
        aux_losses = self._loss_terms(output)
        weights = {}
        if training and self.balancer is not None and aux_losses:
            weights = self.balancer.weights(cls_loss, aux_losses, self.model)
        total_loss = cls_loss
        for name, loss in aux_losses.items():
            total_loss = total_loss + weights.get(name, 1.0) * loss
        if training:
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
        return {
            "logits": logits.detach(),
            "targets": targets.detach(),
            "loss_total": float(total_loss.detach().item()),
            "loss_cls": float(cls_loss.detach().item()),
            "loss_terms": {name: float(loss.detach().item()) for name, loss in aux_losses.items()},
            "loss_weights": dict(weights),
        }

    def train_epoch(
        self, loader: Iterable[list[dict[str, Any]]], total_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        self.model.train()
        metrics = ClassificationMetrics()
        loss_total = 0.0
        loss_cls = 0.0
        aux_totals: Dict[str, float] = {"struct": 0.0, "entropy": 0.0, "consistency": 0.0}
        batches = 0

        if total_batches is None and hasattr(loader, "__len__"):
            total_batches = len(loader)

        for batch in loader:
            batch = move_to_device(batch, self.device)
            result = self._step(batch, training=True)
            metrics.update(result["logits"], result["targets"])
            loss_total += result["loss_total"]
            loss_cls += result["loss_cls"]
            for name, value in result["loss_terms"].items():
                aux_totals[name] = aux_totals.get(name, 0.0) + value
            batches += 1

            if batches % 50 == 0:
                running_metrics = metrics.compute()
                progress = f"  Batch {batches}/{total_batches or '?'}  loss={loss_total / batches:.4f}  WA={running_metrics['wa']:.4f}"
                print(progress)

        summary = metrics.compute()
        summary.update(
            {
                "loss_total": loss_total / max(1, batches),
                "loss_cls": loss_cls / max(1, batches),
            }
        )
        for name, value in aux_totals.items():
            summary[f"loss_{name}"] = value / max(1, batches)
        return summary

    @torch.no_grad()
    def evaluate(self, loader: Iterable[list[dict[str, Any]]]) -> Dict[str, Any]:
        self.model.eval()
        metrics = ClassificationMetrics()
        loss_total = 0.0
        loss_cls = 0.0
        aux_totals: Dict[str, float] = {"struct": 0.0, "entropy": 0.0, "consistency": 0.0}
        batches = 0

        for batch in loader:
            batch = move_to_device(batch, self.device)
            output = self.model(batch)
            logits = _as_logits(output)
            targets = self._resolve_batch_targets(batch, output)
            weight = self.class_weights.to(self.device) if self.class_weights is not None else None
            cls_loss = F.cross_entropy(logits, targets, weight=weight, label_smoothing=self.label_smoothing)
            aux_losses = self._loss_terms(output)
            total_loss = cls_loss
            for name, loss in aux_losses.items():
                total_loss = total_loss + loss
            metrics.update(logits, targets)
            loss_total += float(total_loss.item())
            loss_cls += float(cls_loss.item())
            for name, loss in aux_losses.items():
                aux_totals[name] = aux_totals.get(name, 0.0) + float(loss.item())
            batches += 1

        summary = metrics.compute()
        summary.update(
            {
                "loss_total": loss_total / max(1, batches),
                "loss_cls": loss_cls / max(1, batches),
            }
        )
        for name, value in aux_totals.items():
            summary[f"loss_{name}"] = value / max(1, batches)
        return summary

    def _score_summary(self, summary: Mapping[str, Any], metric_name: str) -> float:
        if metric_name in summary:
            return float(summary[metric_name])
        if metric_name == "paper_score":
            return 0.5 * (float(summary.get("wa", 0.0)) + float(summary.get("ua", 0.0)))
        if metric_name == "unified_paper_score":
            if "wa" in summary or "ua" in summary:
                return 0.5 * (float(summary.get("wa", 0.0)) + float(summary.get("ua", 0.0)))
            return 0.5 * (float(summary.get("wf1", 0.0)) + float(summary.get("macro_f1", 0.0)))
        if metric_name == "meld_paper_score":
            return 0.5 * (float(summary.get("wf1", 0.0)) + float(summary.get("macro_f1", 0.0)))
        if metric_name == "ravdess_paper_score":
            return float(summary.get("wa", 0.0))
        if metric_name == "wa+ua":
            return float(summary.get("wa", 0.0)) + float(summary.get("ua", 0.0))
        if metric_name == "wf1+macro_f1":
            return float(summary.get("wf1", 0.0)) + float(summary.get("macro_f1", 0.0))
        raise KeyError(f"Unknown metric for model selection: {metric_name}")

    def fit(
        self,
        train_loader: Iterable[list[dict[str, Any]]],
        val_loader: Optional[Iterable[list[dict[str, Any]]]] = None,
        epochs: int = 1,
        checkpoint_dir: str | Path | None = None,
        save_best_on: str = "wf1",
        early_stopping_patience: int | None = None,
    ) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        best_metric = float("-inf")
        best_path: Optional[Path] = None
        best_epoch: Optional[int] = None
        stale_epochs = 0
        checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if checkpoint_path is not None:
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_summary = self.train_epoch(train_loader)
            record = {"epoch": epoch, "train": train_summary}
            if val_loader is not None:
                val_summary = self.evaluate(val_loader)
                record["val"] = val_summary
                score = self._score_summary(val_summary, save_best_on)
                if score > best_metric and checkpoint_path is not None:
                    best_metric = score
                    best_epoch = epoch
                    stale_epochs = 0
                    best_path = checkpoint_path / "best.pt"
                    save_checkpoint(
                        best_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        balancer=self.balancer,
                        epoch=epoch,
                        metrics=val_summary,
                    )
                    record["best_checkpoint"] = str(best_path)
                elif score > best_metric:
                    best_metric = score
                    best_epoch = epoch
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(score)
                    else:
                        self.scheduler.step()

                print(
                    f"Epoch {epoch}/{epochs}  "
                    f"train_loss={train_summary['loss_total']:.4f}  train_WA={train_summary['wa']:.4f}  train_UA={train_summary['ua']:.4f}  "
                    f"val_loss={val_summary['loss_total']:.4f}  val_WA={val_summary['wa']:.4f}  val_UA={val_summary['ua']:.4f}  "
                    f"best_metric={best_metric:.4f}"
                )
            elif checkpoint_path is not None:
                latest = checkpoint_path / f"epoch_{epoch:03d}.pt"
                save_checkpoint(
                    latest,
                    model=self.model,
                    optimizer=self.optimizer,
                    balancer=self.balancer,
                    epoch=epoch,
                    metrics=train_summary,
                )
                record["checkpoint"] = str(latest)
                if self.scheduler is not None:
                    self.scheduler.step()
            elif self.scheduler is not None:
                self.scheduler.step()
            if val_loader is None:
                print(
                    f"Epoch {epoch}/{epochs}  "
                    f"train_loss={train_summary['loss_total']:.4f}  train_WA={train_summary['wa']:.4f}  train_UA={train_summary['ua']:.4f}"
                )
            history.append(record)

            if early_stopping_patience is not None and stale_epochs >= early_stopping_patience:
                record["early_stopped"] = True
                break

        return {
            "history": history,
            "best_checkpoint": str(best_path) if best_path is not None else None,
            "best_metric": None if best_metric == float("-inf") else best_metric,
            "best_epoch": best_epoch,
        }

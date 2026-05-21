from __future__ import annotations

import argparse
import inspect
import importlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
from torch.utils.data import DataLoader

from .paths import resolve_project_path
from .training import EMAGradientBalancer, Trainer, load_checkpoint


def _load_json(path: str | Path) -> Dict[str, Any]:
    config_path = resolve_project_path(path)
    if config_path is None:
        raise ValueError("Config path is required.")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _import_attr(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _filter_kwargs(callable_obj: Any, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(callable_obj)
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in accepted}


def _identity_collate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return samples


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_datasets(config: Mapping[str, Any]) -> Dict[str, Any]:
    dataset_cfg = dict(config.get("dataset", {}))
    dataset_type = dataset_cfg.get("type", dataset_cfg.get("name"))
    if not dataset_type:
        raise KeyError("Config must define dataset.type or dataset.name.")
    builder_cfg = {
        key: value
        for key, value in dataset_cfg.items()
        if key not in {"type", "name", "builder"}
    }

    data_module = importlib.import_module("gs3_icl.data")
    builder_name = dataset_cfg.get("builder")
    if builder_name and hasattr(data_module, builder_name):
        builder = getattr(data_module, builder_name)
        datasets = builder(dataset_cfg)
        if isinstance(datasets, Mapping):
            return dict(datasets)

    if hasattr(data_module, "build_datasets"):
        datasets = data_module.build_datasets(dataset_cfg)
        if isinstance(datasets, Mapping):
            return dict(datasets)

    if dataset_type == "savee":
        if "split_seed" in builder_cfg and "seed" not in builder_cfg:
            builder_cfg["seed"] = builder_cfg.pop("split_seed")
        dataset_cls = getattr(data_module, "SAVEEDataset")
        class_kwargs = _filter_kwargs(dataset_cls.__init__, builder_cfg)
        return {
            "train": dataset_cls(split="train", **class_kwargs),
            "val": dataset_cls(split="val", **class_kwargs),
            "test": dataset_cls(split="test", **class_kwargs),
        }

    if dataset_type == "meld":
        if "root" in builder_cfg and "data_root" not in builder_cfg:
            builder_cfg["data_root"] = builder_cfg.pop("root")
        if "audio_feature_dir" in builder_cfg and "audio_feature_root" not in builder_cfg:
            builder_cfg["audio_feature_root"] = builder_cfg.pop("audio_feature_dir")
        if "visual_feature_dir" in builder_cfg and "visual_feature_root" not in builder_cfg:
            builder_cfg["visual_feature_root"] = builder_cfg.pop("visual_feature_dir")
        dataset_cls = getattr(data_module, "MELDUtteranceDataset")
        class_kwargs = _filter_kwargs(dataset_cls.__init__, builder_cfg)
        return {
            "train": dataset_cls(split="train", **class_kwargs),
            "val": dataset_cls(split="dev", **class_kwargs),
            "test": dataset_cls(split="test", **class_kwargs),
        }

    if dataset_type == "ravdess":
        dataset_cls = getattr(data_module, "RAVDESSDataset")
        class_kwargs = _filter_kwargs(dataset_cls.__init__, builder_cfg)
        return {
            "train": dataset_cls(split="train", **class_kwargs),
            "val": dataset_cls(split="val", **class_kwargs),
            "test": dataset_cls(split="test", **class_kwargs),
        }

    raise ValueError(f"Unsupported dataset type: {dataset_type!r}")


def _build_model(
    config: Mapping[str, Any],
    *,
    num_classes: int | None = None,
    vocab_size: int | None = None,
) -> torch.nn.Module:
    model_module = importlib.import_module("gs3_icl.model")
    model_cfg = dict(config.get("model", {}))
    if "pretrained_encoders" in config and "pretrained_encoders" not in model_cfg:
        model_cfg["pretrained_encoders"] = config["pretrained_encoders"]
    builder_cfg = {key: value for key, value in model_cfg.items() if key not in {"name", "type", "builder"}}
    if hasattr(model_module, "build_model"):
        model = model_module.build_model(builder_cfg, num_classes=num_classes, vocab_size=vocab_size)
    else:
        model_cls = getattr(model_module, "GS3ICL", None) or getattr(model_module, "Model")
        if model_cls is None:
            raise AttributeError("gs3_icl.model must expose build_model() or GS3ICL/Model.")
        if num_classes is not None and "num_classes" not in builder_cfg:
            builder_cfg["num_classes"] = num_classes
        if vocab_size is not None and "vocab_size" not in builder_cfg:
            builder_cfg["vocab_size"] = vocab_size
        model = model_cls(**_filter_kwargs(model_cls.__init__, builder_cfg))
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Model builder must return a torch.nn.Module.")
    return model


def _infer_num_classes(datasets: Mapping[str, Any], config: Mapping[str, Any]) -> int | None:
    model_cfg = dict(config.get("model", {}))
    if "num_classes" in model_cfg:
        return int(model_cfg["num_classes"])
    dataset_cfg = dict(config.get("dataset", {}))
    if "num_classes" in dataset_cfg:
        return int(dataset_cfg["num_classes"])
    sample_dataset = datasets.get("train")
    labels = getattr(sample_dataset, "label_vocab", None) or getattr(sample_dataset, "labels", None)
    if isinstance(labels, Mapping):
        return len(labels)
    if isinstance(labels, (list, tuple)):
        return len(labels)
    return None


def _infer_vocab_size(datasets: Mapping[str, Any], config: Mapping[str, Any]) -> int | None:
    model_cfg = dict(config.get("model", {}))
    if "vocab_size" in model_cfg:
        return int(model_cfg["vocab_size"])
    sample_dataset = datasets.get("train")
    vocab = getattr(sample_dataset, "vocab", None)
    if vocab is not None:
        try:
            return int(len(vocab))
        except TypeError:
            return None
    return None


def _collect_labels(dataset: Any) -> list[int]:
    labels = []
    samples = getattr(dataset, "samples", None)
    if isinstance(samples, list):
        for sample in samples:
            if isinstance(sample, dict) and "label" in sample:
                labels.append(int(sample["label"]))
        if labels:
            return labels
    records = getattr(dataset, "records", None)
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict) and "label" in record:
                labels.append(int(record["label"]))
        if labels:
            return labels
    for index in range(len(dataset)):
        item = dataset[index]
        labels.append(int(item["label"]))
    return labels


def _build_class_weights(
    training_cfg: Mapping[str, Any],
    train_dataset: Any,
    num_classes: int | None,
) -> torch.Tensor | None:
    weighting = training_cfg.get("class_weighting")
    if not weighting:
        return None
    if weighting != "balanced":
        raise ValueError(f"Unsupported class_weighting: {weighting!r}")
    if num_classes is None:
        raise ValueError("num_classes is required to compute class weights.")
    labels = _collect_labels(train_dataset)
    counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long), minlength=num_classes).float()
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (counts * float(num_classes))
    return weights


def _build_scheduler(
    scheduler_cfg: Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> Any:
    if not scheduler_cfg:
        return None
    scheduler_type = str(scheduler_cfg.get("type", "")).lower()
    if scheduler_type == "cosine":
        t_max = int(scheduler_cfg.get("t_max", epochs))
        eta_min = float(scheduler_cfg.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    if scheduler_type == "plateau":
        factor = float(scheduler_cfg.get("factor", 0.5))
        patience = int(scheduler_cfg.get("patience", 5))
        mode = str(scheduler_cfg.get("mode", "max"))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
        )
    raise ValueError(f"Unsupported scheduler type: {scheduler_type!r}")


def _build_loader(dataset: Any, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    collate_fn = getattr(dataset, "collate_fn", None)
    if collate_fn is None:
        try:
            data_module = importlib.import_module("gs3_icl.data")
            collate_fn = getattr(data_module, "collate_fn", None) or getattr(data_module, "collate_samples", None)
        except Exception:
            collate_fn = None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn or _identity_collate,
    )


def run_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    config = dict(config)
    seed = int(config.get("training", {}).get("seed", 1337))
    _set_seed(seed)
    device = torch.device(config.get("training", {}).get("device", "cpu"))
    datasets = _build_datasets(config)
    num_classes = _infer_num_classes(datasets, config)
    vocab_size = _infer_vocab_size(datasets, config)
    model = _build_model(config, num_classes=num_classes, vocab_size=vocab_size).to(device)

    training_cfg = dict(config.get("training", {}))
    batch_size = int(training_cfg.get("batch_size", 1))
    num_workers = int(training_cfg.get("num_workers", 0))
    lr = float(training_cfg.get("learning_rate", 3e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _build_scheduler(training_cfg.get("scheduler"), optimizer, epochs=int(training_cfg.get("epochs", 1)))
    class_weights = _build_class_weights(training_cfg, datasets["train"], num_classes)
    balancer = EMAGradientBalancer(momentum=float(training_cfg.get("ema_momentum", 0.99)))
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        balancer=balancer,
        grad_clip_norm=training_cfg.get("grad_clip_norm"),
        class_weights=class_weights,
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
        scheduler=scheduler,
    )

    loaders = {
        split: _build_loader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)
        for split, dataset in datasets.items()
    }

    resume_from = training_cfg.get("resume_from")
    if resume_from:
        resume_path = resolve_project_path(resume_from)
        load_checkpoint(resume_path, model=model, optimizer=optimizer, balancer=balancer, map_location=device)

    if training_cfg.get("eval_only"):
        results = {split: trainer.evaluate(loader) for split, loader in loaders.items()}
        return {"config": config, "results": results}

    epochs = int(training_cfg.get("epochs", 1))
    checkpoint_dir = resolve_project_path(training_cfg.get("checkpoint_dir"))
    fit_result = trainer.fit(
        train_loader=loaders["train"],
        val_loader=loaders.get("val"),
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        save_best_on=str(training_cfg.get("save_best_on", "wf1")),
        early_stopping_patience=training_cfg.get("early_stopping_patience"),
    )
    results: Dict[str, Any] = {"train_history": fit_result}
    if "test" in loaders:
        best_checkpoint = fit_result.get("best_checkpoint")
        if best_checkpoint:
            load_checkpoint(best_checkpoint, model=model, optimizer=None, balancer=None, map_location=device)
        results["test"] = trainer.evaluate(loaders["test"])
    return {"config": config, "results": results}


def run(config_path: str | Path) -> Dict[str, Any]:
    return run_config(_load_json(config_path))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train or evaluate GS3-ICL models.")
    parser.add_argument("--config", required=True, help="Path to a JSON configuration file.")
    args = parser.parse_args(argv)
    result = run(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

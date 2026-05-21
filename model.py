from __future__ import annotations

import math
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class GS3ICLConfig:
    num_classes: int
    vocab_size: int
    audio_input_dim: int = 80
    visual_input_dim: int = 512
    audio_backbone: str = "conformer"
    visual_backbone: str = "conformer"
    text_backbone: str = "conformer"
    hidden_dim: int = 768
    num_heads: int = 8
    encoder_layers: int = 1
    fusion_layers: int = 2
    ff_multiplier: int = 4
    dropout: float = 0.1
    segment_length: int = 8
    audio_budget: int = 16
    visual_budget: int = 16
    text_budget: int = 8
    selection_temperature: float = 0.1
    graph_propagation_steps: int = 2
    graph_window_ratio: float = 0.125
    graph_window_max: int = 16
    text_graph_topk: int = 8
    gaussian_std: float = 0.1
    projector_dim: int = 256
    max_fusion_tokens: Optional[int] = None
    conformer_kernel_size: int = 15
    eps: float = 1e-5
    # Pretrained encoder configs
    pretrained_text_model: str = ""
    pretrained_audio_model: str = ""
    pretrained_visual_model: str = ""
    text_freeze_encoder: bool = True
    audio_freeze_encoder: bool = True
    visual_freeze_encoder: bool = True
    text_freeze_layers: Optional[int] = None
    audio_freeze_layers: Optional[int] = None
    visual_freeze_layers: Optional[int] = None
    gradient_checkpointing: bool = False
    pretrained_cache_dir: str = ""
    max_text_length: int = 128
    max_audio_seconds: float = 10.0
    num_visual_frames: int = 16
    pretrained_encoder_dropout: float = 0.1


@dataclass
class SelectionResult:
    modality: str
    encoded_tokens: Tensor
    propagated_tokens: Tensor
    selected_tokens: Tensor
    selected_indices: Tensor
    logits: Tensor
    probs: Tensor
    struct_loss: Tensor
    entropy_loss: Tensor


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        ff_dim = hidden_dim * ff_multiplier
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, attn_weights


class TransformerStack(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        ff_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerBlock(hidden_dim, num_heads, ff_multiplier, dropout)
            for _ in range(num_layers)
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        last_attention = None
        for index, layer in enumerate(self.layers):
            want_attention = return_attention and index == len(self.layers) - 1
            x, last_attention = layer(
                x, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                return_attention=want_attention,
            )
        return x, last_attention


class ConformerFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        ff_dim = hidden_dim * ff_multiplier
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * self.net(self.norm(x))


class ConformerConvModule(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Conformer kernel_size must be odd for same-length padding.")
        self.norm = nn.LayerNorm(hidden_dim)
        self.pointwise_in = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.batch_norm = nn.InstanceNorm1d(hidden_dim, affine=True)
        self.pointwise_out = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) < 2:
            return x
        y = self.norm(x).transpose(1, 2)
        y = F.glu(self.pointwise_in(y), dim=1)
        y = self.depthwise(y)
        y = self.batch_norm(y)
        y = F.silu(y)
        y = self.pointwise_out(y)
        y = self.dropout(y)
        return y.transpose(1, 2).contiguous()


class ConformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_multiplier: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ffn1 = ConformerFeedForward(hidden_dim, ff_multiplier, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(hidden_dim, kernel_size, dropout)
        self.ffn2 = ConformerFeedForward(hidden_dim, ff_multiplier, dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = x + self.ffn1(x)
        h = self.attn_norm(x)
        attn_out, attn_weights = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + self.attn_dropout(attn_out)
        x = x + self.conv_module(x)
        x = x + self.ffn2(x)
        return self.final_norm(x), attn_weights


class ConformerStack(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        ff_multiplier: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            ConformerBlock(
                hidden_dim,
                num_heads,
                ff_multiplier,
                kernel_size,
                dropout,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        last_attention = None
        for index, layer in enumerate(self.layers):
            want_attention = return_attention and index == len(self.layers) - 1
            x, last_attention = layer(
                x, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                return_attention=want_attention,
            )
        return x, last_attention


class PretrainedEncoderWrapper(nn.Module):
    """Wraps a HuggingFace pretrained model for feature extraction."""

    def __init__(
        self,
        model_name: str,
        model_class,
        freeze: bool = True,
        freeze_layers: Optional[int] = None,
        gradient_checkpointing: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        self.encoder = model_class.from_pretrained(model_name, **kwargs)
        self.output_dim = self._get_output_dim()

        if freeze:
            self._freeze_all()
        elif freeze_layers is not None:
            self._freeze_partial(freeze_layers)

        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def _get_output_dim(self) -> int:
        config = self.encoder.config
        for attr in ("hidden_size", "d_model", "n_embd", "dim"):
            if hasattr(config, attr):
                return getattr(config, attr)
        return 768

    def _freeze_all(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _freeze_partial(self, num_layers: int):
        if hasattr(self.encoder, "embeddings"):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
        layers_attr = None
        for attr in ("layer", "layers", "encoder", "h"):
            if hasattr(self.encoder, attr):
                candidate = getattr(self.encoder, attr)
                if isinstance(candidate, (nn.ModuleList, list)):
                    layers_attr = candidate
                    break
        if layers_attr is not None:
            for i, layer in enumerate(layers_attr):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(self, *args, **kwargs):
        with torch.no_grad() if not any(p.requires_grad for p in self.encoder.parameters()) else _nullcontext():
            output = self.encoder(*args, **kwargs)
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if hasattr(output, "hidden_states"):
            return output.hidden_states[-1]
        return output[0]


class GS3ICLModel(nn.Module):
    """Graph-structured sparse selection with invariance-consistent learning."""

    modality_to_index = {"audio": 0, "visual": 1, "text": 2}

    def __init__(self, config: GS3ICLConfig) -> None:
        super().__init__()
        self.config = config
        max_fusion_tokens = config.max_fusion_tokens
        if max_fusion_tokens is None:
            max_fusion_tokens = config.audio_budget + config.visual_budget + config.text_budget

        self.text_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.text_feature_projection = nn.LazyLinear(config.hidden_dim)
        self.audio_projection = nn.Linear(config.audio_input_dim, config.hidden_dim)
        self.visual_projection = nn.Linear(config.visual_input_dim, config.hidden_dim)
        self.audio_conformer = ConformerStack(
            config.hidden_dim,
            config.num_heads,
            config.encoder_layers,
            config.ff_multiplier,
            config.conformer_kernel_size,
            config.dropout,
        )
        self.visual_conformer = ConformerStack(
            config.hidden_dim,
            config.num_heads,
            config.encoder_layers,
            config.ff_multiplier,
            config.conformer_kernel_size,
            config.dropout,
        )
        self.text_conformer = ConformerStack(
            config.hidden_dim,
            config.num_heads,
            config.encoder_layers,
            config.ff_multiplier,
            config.conformer_kernel_size,
            config.dropout,
        )

        # Pretrained encoders (optional)
        self.text_pretrained_encoder = None
        self.audio_pretrained_encoder = None
        self.visual_pretrained_encoder = None

        if config.pretrained_text_model and config.text_backbone == "bert":
            from transformers import BertModel

            self.text_pretrained_encoder = PretrainedEncoderWrapper(
                config.pretrained_text_model,
                BertModel,
                freeze=config.text_freeze_encoder,
                freeze_layers=config.text_freeze_layers,
                gradient_checkpointing=config.gradient_checkpointing,
                cache_dir=config.pretrained_cache_dir or None,
            )
            self.text_pretrained_proj = nn.Linear(
                self.text_pretrained_encoder.output_dim, config.hidden_dim
            )
            self.text_pretrained_norm = nn.LayerNorm(config.hidden_dim)

        if config.pretrained_audio_model and config.audio_backbone == "wav2vec2":
            from transformers import Wav2Vec2Model

            self.audio_pretrained_encoder = PretrainedEncoderWrapper(
                config.pretrained_audio_model,
                Wav2Vec2Model,
                freeze=config.audio_freeze_encoder,
                freeze_layers=config.audio_freeze_layers,
                gradient_checkpointing=config.gradient_checkpointing,
                cache_dir=config.pretrained_cache_dir or None,
            )
            self.audio_pretrained_proj = nn.Linear(
                self.audio_pretrained_encoder.output_dim, config.hidden_dim
            )
            self.audio_pretrained_norm = nn.LayerNorm(config.hidden_dim)

        if config.pretrained_visual_model and config.visual_backbone == "vit":
            from transformers import ViTModel

            self.visual_pretrained_encoder = PretrainedEncoderWrapper(
                config.pretrained_visual_model,
                ViTModel,
                freeze=config.visual_freeze_encoder,
                freeze_layers=config.visual_freeze_layers,
                gradient_checkpointing=config.gradient_checkpointing,
                cache_dir=config.pretrained_cache_dir or None,
            )
            self.visual_pretrained_proj = nn.Linear(
                self.visual_pretrained_encoder.output_dim, config.hidden_dim
            )
            self.visual_pretrained_norm = nn.LayerNorm(config.hidden_dim)

        self.audio_encoder = TransformerStack(
            config.hidden_dim,
            config.num_heads,
            config.encoder_layers,
            config.ff_multiplier,
            config.dropout,
        )
        self.visual_encoder = TransformerStack(
            config.hidden_dim,
            config.num_heads,
            config.encoder_layers,
            config.ff_multiplier,
            config.dropout,
        )
        self.text_encoder = TransformerStack(
            config.hidden_dim,
            config.num_heads,
            config.encoder_layers,
            config.ff_multiplier,
            config.dropout,
        )
        self.fusion_encoder = TransformerStack(
            config.hidden_dim,
            config.num_heads,
            config.fusion_layers,
            config.ff_multiplier,
            config.dropout,
        )

        self.audio_scorer = nn.Linear(config.hidden_dim, 1)
        self.visual_scorer = nn.Linear(config.hidden_dim, 1)
        self.text_scorer = nn.Linear(config.hidden_dim, 1)

        self.modality_embeddings = nn.Embedding(3, config.hidden_dim)
        self.position_embeddings = nn.Embedding(max_fusion_tokens, config.hidden_dim)
        self.mask_tokens = nn.Parameter(torch.randn(3, config.hidden_dim) * 0.02)

        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.projector_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(config.projector_dim, config.projector_dim),
            nn.GELU(),
            nn.Linear(config.projector_dim, config.projector_dim),
        )

    def forward(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("GS3ICLModel.forward(batch) expects a non-empty list of samples.")

        device = self.classifier.weight.device

        # --- Batched encoding (major speedup vs per-sample) ---
        audio_tokens_list = [s.get("audio_tokens") for s in batch]
        visual_tokens_list = [s.get("visual_tokens") for s in batch]
        text_tokens_list = [s.get("text_tokens") for s in batch]

        audio_encoded_list = self._batch_encode_features(
            audio_tokens_list,
            self.audio_projection,
            self.audio_encoder,
            device,
            backbone=self.config.audio_backbone,
            pretrained_proj=getattr(self, "audio_pretrained_proj", None),
            pretrained_norm=getattr(self, "audio_pretrained_norm", None),
            pretrained_encoder_wrapper=self.audio_pretrained_encoder,
            conformer=self.audio_conformer,
        )
        visual_encoded_list = self._batch_encode_features(
            visual_tokens_list,
            self.visual_projection,
            self.visual_encoder,
            device,
            backbone=self.config.visual_backbone,
            pretrained_proj=getattr(self, "visual_pretrained_proj", None),
            pretrained_norm=getattr(self, "visual_pretrained_norm", None),
            pretrained_encoder_wrapper=self.visual_pretrained_encoder,
            conformer=self.visual_conformer,
        )
        text_encoded_list, text_attention_list = self._batch_encode_text(
            text_tokens_list, device,
            backbone=self.config.text_backbone,
            conformer=self.text_conformer,
        )

        # --- Per-sample: graph, selection, fusion, classification ---
        sample_outputs = []
        for i, sample in enumerate(batch):
            audio_encoded = audio_encoded_list[i]
            visual_encoded = visual_encoded_list[i]
            text_encoded = text_encoded_list[i]
            text_attention = text_attention_list[i]

            audio_selection = self._select_modality(
                "audio", audio_encoded,
                self._build_local_graph(audio_encoded),
                self.audio_scorer, self.config.audio_budget,
            )
            visual_selection = self._select_modality(
                "visual", visual_encoded,
                self._build_local_graph(visual_encoded),
                self.visual_scorer, self.config.visual_budget,
            )
            text_graph = self._build_text_graph(text_attention, text_encoded.size(0), device)
            text_selection = self._select_modality(
                "text", text_encoded, text_graph,
                self.text_scorer, self.config.text_budget,
            )

            selections = {"audio": audio_selection, "visual": visual_selection, "text": text_selection}
            fused_tokens, modality_ids, selection_metadata = self._build_fusion_tokens(selections, device)
            pooled = self._run_fusion(fused_tokens, modality_ids)
            logits_i = self.classifier(pooled)

            if self.training:
                gaussian_view = pooled + torch.randn_like(pooled) * self.config.gaussian_std
                masked_representation, masked_modality = self._build_mask_view(fused_tokens, modality_ids)
                consistency = self._consistency_loss(gaussian_view, masked_representation)
            else:
                masked_modality = None
                consistency = pooled.new_zeros(())

            struct_loss = (audio_selection.struct_loss + visual_selection.struct_loss + text_selection.struct_loss) / 3.0
            entropy_loss = (audio_selection.entropy_loss + visual_selection.entropy_loss + text_selection.entropy_loss) / 3.0

            target = self._prepare_label(sample.get("label"), device)
            metadata_i = {
                "meta": sample.get("meta"),
                "lengths": {
                    "audio": int(audio_selection.encoded_tokens.size(0)),
                    "visual": int(visual_selection.encoded_tokens.size(0)),
                    "text": int(text_selection.encoded_tokens.size(0)),
                },
                "selected_indices": selection_metadata,
                "masked_modality": masked_modality,
            }
            sample_outputs.append({
                "logits": logits_i,
                "target": target,
                "loss_terms": {"struct": struct_loss, "entropy": entropy_loss, "consistency": consistency},
                "metadata": metadata_i,
            })

        logits = torch.stack([item["logits"] for item in sample_outputs], dim=0)
        targets = torch.stack([item["target"] for item in sample_outputs], dim=0)
        struct = torch.stack([item["loss_terms"]["struct"] for item in sample_outputs]).mean()
        entropy = torch.stack([item["loss_terms"]["entropy"] for item in sample_outputs]).mean()
        consistency = torch.stack([item["loss_terms"]["consistency"] for item in sample_outputs]).mean()
        metadata = [item["metadata"] for item in sample_outputs]

        return {
            "logits": logits,
            "targets": targets,
            "loss_terms": {
                "struct": struct,
                "entropy": entropy,
                "consistency": consistency,
            },
            "metadata": metadata,
        }

    def _prepare_label(self, label: Any, device: torch.device) -> Tensor:
        if isinstance(label, Tensor):
            return label.to(device=device, dtype=torch.long).reshape(())
        return torch.tensor(int(label), device=device, dtype=torch.long)

    def _segment_local_normalize(self, x: Tensor) -> Tensor:
        if x.numel() == 0:
            return x
        segment_length = max(1, self.config.segment_length)
        chunks = []
        for start in range(0, x.size(0), segment_length):
            chunk = x[start : start + segment_length]
            mean = chunk.mean(dim=0, keepdim=True)
            var = chunk.var(dim=0, unbiased=False, keepdim=True)
            chunks.append((chunk - mean) / torch.sqrt(var + self.config.eps))
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------
    # Batched encoding – processes all samples in a batch through the
    # projection + encoder in a single GPU call instead of one-by-one.
    # ------------------------------------------------------------------

    def _batch_encode_features(
        self,
        tokens_list: List[Any],
        projection: nn.Module,
        encoder,
        device: torch.device,
        backbone: str = "linear",
        pretrained_proj: Optional[nn.Module] = None,
        pretrained_norm: Optional[nn.Module] = None,
        pretrained_encoder_wrapper: Optional[nn.Module] = None,
        conformer: Optional[nn.Module] = None,
    ) -> List[Tensor]:
        """Batch-encode a list of feature tensors through projection + encoder."""
        target_encoder = conformer if (backbone == "conformer" and conformer is not None) else encoder

        # --- Pretrained frozen encoder: run per-sample, then batch project+encode ---
        if pretrained_encoder_wrapper is not None:
            valid_items = []
            for idx, tok in enumerate(tokens_list):
                t = self._as_tensor(tok, device)
                if t is None:
                    continue
                if t.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                    if t.ndim == 1:
                        t = t.unsqueeze(-1)
                    if t.ndim > 2:
                        t = t.reshape(t.shape[0], -1)
                    hs = self._segment_local_normalize(t.float())
                else:
                    with torch.no_grad():
                        out = pretrained_encoder_wrapper.encoder(
                            t.float().unsqueeze(0) if t.ndim < 3 else t.float()
                        )
                    hs = out.last_hidden_state.squeeze(0)
                valid_items.append((idx, hs))
            if not valid_items:
                return [self._empty_sequence(device)] * len(tokens_list)
            # Batch project + normalise
            stacked = torch.cat([self._segment_local_normalize(p.float()) for _, p in valid_items], dim=0)
            proj = pretrained_proj(stacked)
            proj = pretrained_norm(proj)
            offsets, cur = [], 0
            for _, p in valid_items:
                offsets.append((cur, cur + p.size(0)))
                cur += p.size(0)
            proj_dict = {idx: proj[s:e] for (idx, _), (s, e) in zip(valid_items, offsets)}
            projected = [proj_dict.get(i, None) for i in range(len(tokens_list))]
        else:
            # --- Standard: locally normalise per-sample, then batch project+encode ---
            normed = []
            for tok in tokens_list:
                t = self._as_tensor(tok, device)
                if t is None:
                    normed.append(None)
                    continue
                if t.ndim == 1:
                    t = t.unsqueeze(-1)
                if t.ndim > 2:
                    t = t.reshape(t.shape[0], -1)
                normed.append(self._segment_local_normalize(t.float()))

            valid = [(i, n) for i, n in enumerate(normed) if n is not None]
            if not valid:
                return [self._empty_sequence(device)] * len(tokens_list)

            stacked = torch.cat([n for _, n in valid], dim=0)
            proj = projection(stacked)
            offsets, cur = [], 0
            for _, n in valid:
                offsets.append((cur, cur + n.size(0)))
                cur += n.size(0)
            projected = [(idx, proj[s:e]) for idx, (s, e) in zip([i for i, _ in valid], offsets)]

            proj_dict = {idx: p for idx, p in projected}
            projected = [proj_dict.get(i, None) for i in range(len(tokens_list))]

        # Pad → batched encoder → unpad
        return self._run_batched_encoder(projected, target_encoder, device)

    def _run_batched_encoder(
        self,
        projected_list: List[Optional[Tensor]],
        encoder,
        device: torch.device,
        return_attention: bool = False,
    ) -> List[Tensor] | Tuple[List[Tensor], List[Optional[Tensor]]]:
        """Pad projected features, run through encoder, unpad."""
        valid = [(i, p) for i, p in enumerate(projected_list) if p is not None]
        if not valid:
            empty = [self._empty_sequence(device)] * len(projected_list)
            if return_attention:
                return empty, [None] * len(projected_list)
            return empty

        lengths = [p.size(0) for _, p in valid]
        max_len = max(lengths)
        hidden_dim = valid[0][1].size(-1)

        padded = torch.zeros(len(valid), max_len, hidden_dim, device=device)
        pad_mask = torch.ones(len(valid), max_len, dtype=torch.bool, device=device)
        for j, (_, p) in enumerate(valid):
            padded[j, : p.size(0)] = p
            pad_mask[j, : p.size(0)] = False  # False = not padded (valid)

        encoded, attention = encoder(
            padded, attn_mask=None, key_padding_mask=pad_mask,
            return_attention=return_attention,
        )

        results: Dict[int, Tensor] = {}
        attn_results: Dict[int, Optional[Tensor]] = {}
        if return_attention and attention is not None:
            # attention can be [B, H, T, T] or [B, T, T]
            _attn_shape = attention.shape
        for j, (idx, _) in enumerate(valid):
            results[idx] = encoded[j, : lengths[j]]
            if return_attention and attention is not None:
                L = lengths[j]
                if attention.ndim == 4:
                    # [B, H, T, T] → slice per sample, average over heads
                    attn_j = attention[j, :, :L, :L].mean(dim=0)  # [L, L]
                elif attention.ndim == 3:
                    # [B, T, T]
                    attn_j = attention[j, :L, :L]  # [L, L]
                else:
                    attn_j = None
                attn_results[idx] = attn_j
            else:
                attn_results[idx] = None

        out_list = [results.get(i, self._empty_sequence(device)) for i in range(len(projected_list))]
        if return_attention:
            attn_list = [attn_results.get(i, None) for i in range(len(projected_list))]
            return out_list, attn_list
        return out_list

    def _batch_encode_text(
        self,
        tokens_list: List[Any],
        device: torch.device,
        backbone: str = "transformer",
        conformer: Optional[nn.Module] = None,
    ) -> Tuple[List[Tensor], List[Optional[Tensor]]]:
        """Batch-encode text tokens, returning (encoded_list, attention_list)."""
        target_encoder = conformer if (backbone == "conformer" and conformer is not None) else self.text_encoder

        if backbone == "bert" and self.text_pretrained_encoder is not None:
            return self._batch_encode_text_bert(tokens_list, device, target_encoder)

        projected_list = []
        for tok in tokens_list:
            t = self._as_tensor(tok, device)
            if t is None:
                projected_list.append(None)
                continue
            if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                token_ids = t.long().reshape(-1)
                embedded = self._segment_local_normalize(self.text_embedding(token_ids).float())
            else:
                if t.ndim == 1:
                    t = t.unsqueeze(-1)
                if t.ndim > 2:
                    t = t.reshape(t.shape[0], -1)
                normalized = self._segment_local_normalize(t.float())
                embedded = self.text_feature_projection(normalized)
            projected_list.append(embedded)

        return self._run_batched_encoder(projected_list, target_encoder, device, return_attention=True)

    def _batch_encode_text_bert(
        self,
        tokens_list: List[Any],
        device: torch.device,
        target_encoder,
    ) -> Tuple[List[Tensor], List[Optional[Tensor]]]:
        """Batch-encode text with pretrained BERT + projection + encoder."""
        projected_list = []
        for tok in tokens_list:
            t = self._as_tensor(tok, device)
            if t is None:
                projected_list.append(None)
                continue
            if t.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                if t.ndim == 1:
                    t = t.unsqueeze(-1)
                if t.ndim > 2:
                    t = t.reshape(t.shape[0], -1)
                hs = t.float()
            else:
                token_ids = t.long().reshape(1, -1)
                max_len = self.config.max_text_length
                if token_ids.size(1) > max_len:
                    token_ids = token_ids[:, :max_len]
                attention_mask = torch.ones_like(token_ids)
                _frozen = not any(p.requires_grad for p in self.text_pretrained_encoder.encoder.parameters())
                with torch.no_grad() if _frozen else _nullcontext():
                    bert_output = self.text_pretrained_encoder.encoder(
                        input_ids=token_ids, attention_mask=attention_mask
                    )
                hs = bert_output.last_hidden_state.squeeze(0)
            hs = self._segment_local_normalize(hs.float())
            proj = self.text_pretrained_proj(hs)
            proj = self.text_pretrained_norm(proj)
            projected_list.append(proj)

        return self._run_batched_encoder(projected_list, target_encoder, device, return_attention=True)

    def _build_local_graph(self, tokens: Tensor) -> Tensor:
        length = tokens.size(0)
        if length == 0:
            return tokens.new_zeros((0, 0))
        if length == 1:
            return tokens.new_zeros((1, 1))

        normalized = F.normalize(tokens, dim=-1, eps=self.config.eps)
        similarity = normalized @ normalized.transpose(0, 1)
        similarity = similarity.clamp_min(0.0)

        radius = self._graph_window_radius(length)
        positions = torch.arange(length, device=tokens.device)
        local_mask = (positions[:, None] - positions[None, :]).abs() <= radius
        adjacency = similarity * local_mask.float()
        adjacency.fill_diagonal_(0.0)
        return adjacency

    def _build_text_graph(
        self,
        attention: Optional[Tensor],
        length: int,
        device: torch.device,
    ) -> Tensor:
        if length == 0:
            return torch.zeros((0, 0), device=device)
        if length == 1:
            return torch.zeros((1, 1), device=device)
        if attention is None:
            return torch.zeros((length, length), device=device)

        adjacency = 0.5 * (attention + attention.transpose(0, 1))
        topk = min(self.config.text_graph_topk, length)
        if topk < length:
            values, indices = adjacency.topk(topk, dim=-1)
            mask = torch.zeros_like(adjacency)
            mask.scatter_(1, indices, torch.ones_like(values))
            adjacency = adjacency * mask
        adjacency = 0.5 * (adjacency + adjacency.transpose(0, 1))
        adjacency = adjacency.clamp_min(0.0)
        adjacency.fill_diagonal_(0.0)
        return adjacency

    def _select_modality(
        self,
        modality: str,
        tokens: Tensor,
        adjacency: Tensor,
        scorer: nn.Linear,
        budget: int,
    ) -> SelectionResult:
        if tokens.size(0) == 0:
            zero = tokens.new_zeros(())
            return SelectionResult(
                modality=modality,
                encoded_tokens=tokens,
                propagated_tokens=tokens,
                selected_tokens=tokens,
                selected_indices=torch.zeros(0, device=tokens.device, dtype=torch.long),
                logits=tokens.new_zeros((0,)),
                probs=tokens.new_zeros((0,)),
                struct_loss=zero,
                entropy_loss=zero,
            )

        _, normalized_adjacency, laplacian = self._graph_matrices(adjacency)
        propagated = tokens
        for _ in range(self.config.graph_propagation_steps):
            propagated = normalized_adjacency @ propagated

        logits = scorer(propagated).squeeze(-1)
        probs = F.softmax(logits / self.config.selection_temperature, dim=0)
        k = min(max(budget, 0), propagated.size(0))

        if k > 0:
            selected_indices = torch.topk(probs, k=k, dim=0).indices.sort().values
            hard = torch.zeros_like(probs)
            hard[selected_indices] = 1.0
            soft = probs * float(k)
            ste_mask = hard + soft - soft.detach()
            selected_tokens = propagated[selected_indices] * ste_mask[selected_indices].unsqueeze(-1)
        else:
            selected_indices = torch.zeros(0, device=tokens.device, dtype=torch.long)
            selected_tokens = propagated.new_zeros((0, propagated.size(-1)))

        norm = max(1, propagated.size(0))
        struct_loss = (logits @ (laplacian @ logits)) / norm
        entropy_loss = -(probs * (probs + self.config.eps).log()).sum()

        return SelectionResult(
            modality=modality,
            encoded_tokens=tokens,
            propagated_tokens=propagated,
            selected_tokens=selected_tokens,
            selected_indices=selected_indices,
            logits=logits,
            probs=probs,
            struct_loss=struct_loss,
            entropy_loss=entropy_loss,
        )

    def _graph_matrices(self, adjacency: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        size = adjacency.size(0)
        identity = torch.eye(size, device=adjacency.device, dtype=adjacency.dtype)
        adjacency_hat = adjacency + identity
        degree = adjacency_hat.sum(dim=-1).clamp_min(self.config.eps)
        inv_sqrt_degree = degree.rsqrt()
        normalized_adjacency = inv_sqrt_degree[:, None] * adjacency_hat * inv_sqrt_degree[None, :]
        laplacian = torch.diag(degree) - adjacency_hat
        return adjacency_hat, normalized_adjacency, laplacian

    def _build_fusion_tokens(
        self,
        selections: Dict[str, SelectionResult],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Dict[str, List[int]]]:
        tokens = []
        modality_ids = []
        selected_indices: Dict[str, List[int]] = {}
        for modality in ("audio", "visual", "text"):
            selection = selections[modality]
            if selection.selected_tokens.size(0) == 0:
                selected_indices[modality] = []
                continue
            tokens.append(selection.selected_tokens)
            modality_ids.append(
                torch.full(
                    (selection.selected_tokens.size(0),),
                    self.modality_to_index[modality],
                    device=device,
                    dtype=torch.long,
                )
            )
            selected_indices[modality] = selection.selected_indices.tolist()

        if not tokens:
            fallback = self.mask_tokens.mean(dim=0, keepdim=True)
            return fallback, torch.tensor([0], device=device), selected_indices

        return torch.cat(tokens, dim=0), torch.cat(modality_ids, dim=0), selected_indices

    def _run_fusion(
        self,
        tokens: Tensor,
        modality_ids: Tensor,
        blocked_positions: Optional[Tensor] = None,
    ) -> Tensor:
        positions = torch.arange(tokens.size(0), device=tokens.device)
        positions = positions.clamp(max=self.position_embeddings.num_embeddings - 1)
        fused = (
            tokens
            + self.modality_embeddings(modality_ids)
            + self.position_embeddings(positions)
        )

        attn_mask = None
        if blocked_positions is not None and blocked_positions.any():
            length = fused.size(0)
            attn_mask = torch.zeros((length, length), device=fused.device, dtype=torch.bool)
            attn_mask[blocked_positions, :] = True
            attn_mask[:, blocked_positions] = True
            attn_mask.fill_diagonal_(False)

        fused, _ = self.fusion_encoder(fused.unsqueeze(0), attn_mask=attn_mask)
        fused = fused.squeeze(0)

        if blocked_positions is not None:
            active = ~blocked_positions
            if active.any():
                return fused[active].mean(dim=0)
        return fused.mean(dim=0)

    def _build_mask_view(self, tokens: Tensor, modality_ids: Tensor) -> Tuple[Tensor, Optional[str]]:
        present_modalities = torch.unique(modality_ids).tolist()
        if not present_modalities:
            return self._run_fusion(tokens, modality_ids), None
        if len(present_modalities) < 2:
            return self._run_fusion(tokens, modality_ids), None

        modality_choice = int(torch.randint(len(present_modalities), (1,), device=tokens.device).item())
        masked_modality_id = present_modalities[modality_choice]
        masked_positions = modality_ids == masked_modality_id

        view_tokens = tokens.clone()
        view_tokens[masked_positions] = self.mask_tokens[masked_modality_id]
        masked_representation = self._run_fusion(
            view_tokens,
            modality_ids,
            blocked_positions=masked_positions,
        )
        reverse_lookup = {value: key for key, value in self.modality_to_index.items()}
        return masked_representation, reverse_lookup[masked_modality_id]

    def _consistency_loss(self, gaussian_view: Tensor, masked_view: Tensor) -> Tensor:
        view1 = self.projector(gaussian_view)
        view2 = self.projector(masked_view)
        pred1 = self.predictor(view1)
        pred2 = self.predictor(view2)

        pred1 = F.normalize(pred1, dim=-1, eps=self.config.eps)
        pred2 = F.normalize(pred2, dim=-1, eps=self.config.eps)
        target1 = F.normalize(view1.detach(), dim=-1, eps=self.config.eps)
        target2 = F.normalize(view2.detach(), dim=-1, eps=self.config.eps)

        return 0.5 * (pred1 - target2).pow(2).sum() + 0.5 * (pred2 - target1).pow(2).sum()

    def _graph_window_radius(self, length: int) -> int:
        if length <= 1:
            return 0
        adaptive = int(math.ceil(length * self.config.graph_window_ratio))
        adaptive = max(1, adaptive)
        return min(adaptive, self.config.graph_window_max)

    def _as_tensor(self, value: Any, device: torch.device) -> Optional[Tensor]:
        if value is None:
            return None
        if isinstance(value, Tensor):
            if value.numel() == 0:
                return None
            return value.to(device)
        tensor = torch.as_tensor(value, device=device)
        if tensor.numel() == 0:
            return None
        return tensor

    def _empty_sequence(self, device: torch.device) -> Tensor:
        return torch.zeros((0, self.config.hidden_dim), device=device)


def _read_budget(config: Dict[str, Any], name: str, default: int) -> int:
    top_k = config.get("selection_top_k")
    if isinstance(top_k, dict) and name in top_k:
        return int(top_k[name])
    key = f"{name}_budget"
    if key in config:
        return int(config[key])
    return int(default)


def build_model(
    config: Dict[str, Any],
    *,
    num_classes: Optional[int] = None,
    vocab_size: Optional[int] = None,
) -> GS3ICLModel:
    hidden_dim = int(config.get("hidden_dim", config.get("hidden_size", 768)))
    num_classes = int(num_classes if num_classes is not None else config.get("num_classes", 7))
    vocab_size = int(vocab_size if vocab_size is not None else config.get("vocab_size", 32000))

    # Handle pretrained_encoders section
    pretrained = config.get("pretrained_encoders", {})
    if pretrained:
        text_cfg = pretrained.get("text", {})
        audio_cfg = pretrained.get("audio", {})
        visual_cfg = pretrained.get("visual", {})
        config.setdefault("pretrained_text_model", text_cfg.get("model_name", ""))
        config.setdefault("pretrained_audio_model", audio_cfg.get("model_name", ""))
        config.setdefault("pretrained_visual_model", visual_cfg.get("model_name", ""))
        config.setdefault("text_freeze_encoder", text_cfg.get("freeze", True))
        config.setdefault("audio_freeze_encoder", audio_cfg.get("freeze", True))
        config.setdefault("visual_freeze_encoder", visual_cfg.get("freeze", True))
        config.setdefault("text_freeze_layers", text_cfg.get("freeze_layers"))
        config.setdefault("audio_freeze_layers", audio_cfg.get("freeze_layers"))
        config.setdefault("visual_freeze_layers", visual_cfg.get("freeze_layers"))

    gs3_config = GS3ICLConfig(
        num_classes=num_classes,
        vocab_size=vocab_size,
        audio_input_dim=int(config.get("audio_input_dim", 80)),
        visual_input_dim=int(config.get("visual_input_dim", 512)),
        audio_backbone=str(config.get("audio_backbone", "conformer")).lower(),
        visual_backbone=str(config.get("visual_backbone", "conformer")).lower(),
        text_backbone=str(config.get("text_backbone", "conformer")).lower(),
        hidden_dim=hidden_dim,
        num_heads=int(config.get("num_heads", 8)),
        encoder_layers=int(config.get("encoder_layers", 1)),
        fusion_layers=int(config.get("fusion_layers", 2)),
        ff_multiplier=int(config.get("ff_multiplier", 4)),
        dropout=float(config.get("dropout", 0.1)),
        segment_length=int(config.get("segment_length", 8)),
        audio_budget=_read_budget(config, "audio", 16),
        visual_budget=_read_budget(config, "visual", 16),
        text_budget=_read_budget(config, "text", 8),
        selection_temperature=float(config.get("selection_temperature", 0.1)),
        graph_propagation_steps=int(config.get("graph_propagation_steps", 2)),
        graph_window_ratio=float(config.get("graph_window_ratio", 0.125)),
        graph_window_max=int(config.get("graph_window_max", 16)),
        text_graph_topk=int(config.get("text_graph_topk", 8)),
        gaussian_std=float(config.get("gaussian_std", 0.1)),
        projector_dim=int(config.get("projector_dim", 256)),
        conformer_kernel_size=int(config.get("conformer_kernel_size", 15)),
        pretrained_text_model=str(config.get("pretrained_text_model", "")),
        pretrained_audio_model=str(config.get("pretrained_audio_model", "")),
        pretrained_visual_model=str(config.get("pretrained_visual_model", "")),
        text_freeze_encoder=bool(config.get("text_freeze_encoder", True)),
        audio_freeze_encoder=bool(config.get("audio_freeze_encoder", True)),
        visual_freeze_encoder=bool(config.get("visual_freeze_encoder", True)),
        text_freeze_layers=config.get("text_freeze_layers"),
        audio_freeze_layers=config.get("audio_freeze_layers"),
        visual_freeze_layers=config.get("visual_freeze_layers"),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
        pretrained_cache_dir=str(config.get("pretrained_cache_dir", "")),
        max_text_length=int(config.get("max_text_length", 128)),
        max_audio_seconds=float(config.get("max_audio_seconds", 10.0)),
        num_visual_frames=int(config.get("num_visual_frames", 16)),
        pretrained_encoder_dropout=float(config.get("pretrained_encoder_dropout", 0.1)),
        max_fusion_tokens=int(
            config.get(
                "max_fusion_tokens",
                _read_budget(config, "audio", 16)
                + _read_budget(config, "visual", 16)
                + _read_budget(config, "text", 8),
            )
        ),
    )
    return GS3ICLModel(gs3_config)


GS3ICL = GS3ICLModel

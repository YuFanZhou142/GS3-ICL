from __future__ import annotations

import csv
import math
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .audio import load_wav, log_mel_spectrogram
from .paths import resolve_project_path


SAVEE_LABELS = {
    "a": 0,   # anger
    "d": 1,   # disgust
    "f": 2,   # fear
    "h": 3,   # happiness
    "n": 4,   # neutral
    "sa": 5,  # sadness
    "su": 6,  # surprise
}
SAVEE_LABEL_NAMES = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "neutral",
    5: "sadness",
    6: "surprise",
}

MELD_LABELS = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}
MELD_LABEL_NAMES = {index: name for name, index in MELD_LABELS.items()}

RAVDESS_LABELS = {
    "01": 0,  # neutral
    "02": 1,  # calm
    "03": 2,  # happy
    "04": 3,  # sad
    "05": 4,  # angry
    "06": 5,  # fearful
    "07": 6,  # disgust
    "08": 7,  # surprised
}
RAVDESS_LABEL_NAMES = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+|[^\w\s]")


def _to_path(path: str | Path | None) -> Path | None:
    return resolve_project_path(path)


def tokenize_text(text: str, lowercase: bool = True) -> list[str]:
    if lowercase:
        text = text.lower()
    return TOKEN_PATTERN.findall(text)


class SimpleVocab:
    def __init__(self, token_to_id: dict[str, int] | None = None) -> None:
        if token_to_id is None:
            token_to_id = {"<pad>": 0, "<unk>": 1}
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def __len__(self) -> int:
        return len(self.token_to_id)

    @classmethod
    def build_from_texts(
        cls,
        texts: Iterable[str],
        *,
        min_freq: int = 1,
        max_size: int | None = None,
        lowercase: bool = True,
    ) -> "SimpleVocab":
        counts: dict[str, int] = {}
        for text in texts:
            for token in tokenize_text(text, lowercase=lowercase):
                counts[token] = counts.get(token, 0) + 1

        items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        token_to_id = {"<pad>": 0, "<unk>": 1}
        for token, count in items:
            if count < min_freq:
                continue
            if max_size is not None and len(token_to_id) >= max_size:
                break
            token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    def encode(self, text: str, *, lowercase: bool = True) -> torch.Tensor:
        tokens = tokenize_text(text, lowercase=lowercase)
        if not tokens:
            return torch.empty(0, dtype=torch.long)
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Sequence[int]) -> str:
        tokens = [self.id_to_token.get(int(index), self.unk_token) for index in ids]
        return " ".join(tokens)


def build_vocab_from_texts(
    texts: Iterable[str],
    *,
    min_freq: int = 1,
    max_size: int | None = None,
    lowercase: bool = True,
) -> SimpleVocab:
    return SimpleVocab.build_from_texts(texts, min_freq=min_freq, max_size=max_size, lowercase=lowercase)


def _split_ratios_to_counts(total: int, ratios: Sequence[float]) -> list[int]:
    if total == 0:
        return [0 for _ in ratios]
    raw = [ratio * total for ratio in ratios]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def _stable_shuffle(items: list[Any], seed: int) -> list[Any]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    return items


def split_items_deterministically(
    items: Sequence[Any],
    *,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 1337,
    stratify_key: Callable[[Any], Any] | None = None,
) -> dict[str, list[Any]]:
    split_names = ("train", "val", "test")
    if len(ratios) != len(split_names):
        raise ValueError("ratios must have exactly three entries for train/val/test")

    buckets: dict[str, list[Any]] = {name: [] for name in split_names}
    if stratify_key is None:
        grouped = {None: list(items)}
    else:
        grouped: dict[Any, list[Any]] = {}
        for item in items:
            grouped.setdefault(stratify_key(item), []).append(item)

    for group_index, group_items in enumerate(grouped.values()):
        ordered = sorted(group_items, key=lambda item: str(item))
        shuffled = _stable_shuffle(ordered, seed + group_index)
        counts = _split_ratios_to_counts(len(shuffled), ratios)
        start = 0
        for split_name, count in zip(split_names, counts):
            buckets[split_name].extend(shuffled[start : start + count])
            start += count

    for split_name in split_names:
        buckets[split_name] = sorted(buckets[split_name], key=lambda item: str(item))
    return buckets


def _coerce_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            return torch.empty(0)
        if all(isinstance(item, torch.Tensor) for item in value):
            try:
                return torch.stack(list(value), dim=0)
            except Exception:
                return torch.cat([item.reshape(1, *item.shape) for item in value], dim=0)
        return torch.tensor(value)
    if isinstance(value, (int, float, bool)):
        return torch.tensor(value)
    return torch.empty(0)


def load_tensor_feature(path: str | Path | None) -> torch.Tensor:
    feature_path = _to_path(path)
    if feature_path is None or not feature_path.exists():
        return torch.empty(0)
    try:
        payload = torch.load(feature_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(feature_path, map_location="cpu")

    if isinstance(payload, torch.Tensor):
        return payload.detach().cpu()
    if isinstance(payload, dict):
        for key in ("tensor", "feature", "features", "audio", "audio_tokens", "visual", "visual_tokens", "data"):
            if key in payload:
                tensor = _coerce_tensor(payload[key])
                if tensor.numel() > 0:
                    return tensor.detach().cpu()
        for value in payload.values():
            tensor = _coerce_tensor(value)
            if tensor.numel() > 0:
                return tensor.detach().cpu()
        return torch.empty(0)
    return _coerce_tensor(payload).detach().cpu()


def _empty_text_tokens() -> torch.Tensor:
    return torch.empty(0, dtype=torch.long)


def _empty_float_tokens() -> torch.Tensor:
    return torch.empty(0, dtype=torch.float32)


def collate_samples(batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(batch)


def _extract_savee_code(path: Path) -> tuple[str, str, int]:
    stem = path.stem
    match = re.match(r"^(?P<speaker>[A-Za-z]{2})_(?P<code>[A-Za-z]+)(?P<utterance_id>\d+)$", stem)
    if match is None:
        raise ValueError(f"Unrecognized SAVEE filename: {path.name}")
    speaker = match.group("speaker").upper()
    code = match.group("code").lower()
    utterance_id = int(match.group("utterance_id"))
    if code not in SAVEE_LABELS:
        raise ValueError(f"Unknown SAVEE emotion code '{code}' in file {path.name}")
    return speaker, code, utterance_id


def scan_savee_files(root: str | Path) -> list[dict[str, Any]]:
    root_path = _to_path(root)
    if root_path is None:
        raise ValueError("SAVEE root path is required")
    wav_paths = sorted(root_path.rglob("*.wav"))
    samples: list[dict[str, Any]] = []
    for wav_path in wav_paths:
        speaker, code, utterance_id = _extract_savee_code(wav_path)
        samples.append(
            {
                "dataset": "SAVEE",
                "path": wav_path,
                "speaker": speaker,
                "emotion_code": code,
                "label": SAVEE_LABELS[code],
                "utterance_id": utterance_id,
            }
        )
    return samples


def list_savee_speakers(root: str | Path) -> list[str]:
    speakers = {sample["speaker"] for sample in scan_savee_files(root)}
    return sorted(speakers)


def generate_savee_splits(
    root: str | Path,
    *,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 1337,
) -> dict[str, list[dict[str, Any]]]:
    samples = scan_savee_files(root)
    return split_items_deterministically(
        samples,
        ratios=ratios,
        seed=seed,
        stratify_key=lambda sample: sample["label"],
    )


def _filter_samples_by_speakers(
    samples: Sequence[dict[str, Any]],
    speakers: Sequence[str] | None,
) -> list[dict[str, Any]]:
    if not speakers:
        return list(samples)
    speaker_set = {speaker.upper() for speaker in speakers}
    return [sample for sample in samples if sample["speaker"] in speaker_set]


def build_savee_speaker_split(
    root: str | Path,
    *,
    train_speakers: Sequence[str],
    val_speakers: Sequence[str],
    test_speakers: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    samples = scan_savee_files(root)
    return {
        "train": _filter_samples_by_speakers(samples, train_speakers),
        "val": _filter_samples_by_speakers(samples, val_speakers),
        "test": _filter_samples_by_speakers(samples, test_speakers),
    }


def build_savee_fold_splits(
    root: str | Path,
    *,
    test_speaker: str,
    val_speaker: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    speakers = list_savee_speakers(root)
    normalized_test = test_speaker.upper()
    if normalized_test not in speakers:
        raise ValueError(f"Unknown SAVEE test speaker: {test_speaker}")
    if val_speaker is None:
        candidate_speakers = [speaker for speaker in speakers if speaker != normalized_test]
        if not candidate_speakers:
            raise ValueError("Could not infer a validation speaker for SAVEE fold.")
        val_speaker = candidate_speakers[0]
    normalized_val = val_speaker.upper()
    if normalized_val == normalized_test:
        raise ValueError("Validation speaker and test speaker must differ.")
    train_speakers = [speaker for speaker in speakers if speaker not in {normalized_test, normalized_val}]
    return build_savee_speaker_split(
        root,
        train_speakers=train_speakers,
        val_speakers=[normalized_val],
        test_speakers=[normalized_test],
    )


def default_savee_loso_folds(root: str | Path) -> list[dict[str, list[str]]]:
    speakers = list_savee_speakers(root)
    if len(speakers) < 3:
        raise ValueError("SAVEE LOSO folds require at least three speakers.")
    folds = []
    for index, test_speaker in enumerate(speakers):
        val_speaker = speakers[(index + 1) % len(speakers)]
        if val_speaker == test_speaker:
            val_speaker = speakers[(index + 2) % len(speakers)]
        train_speakers = [speaker for speaker in speakers if speaker not in {test_speaker, val_speaker}]
        folds.append(
            {
                "train_speakers": train_speakers,
                "val_speakers": [val_speaker],
                "test_speakers": [test_speaker],
            }
        )
    return folds


def _extract_ravdess_codes(path: Path) -> dict[str, Any]:
    parts = path.stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unrecognized RAVDESS filename: {path.name}")
    emotion_code = parts[2]
    if emotion_code not in RAVDESS_LABELS:
        raise ValueError(f"Unknown RAVDESS emotion code '{emotion_code}' in file {path.name}")
    return {
        "modality_code": parts[0],
        "channel_code": parts[1],
        "emotion_code": emotion_code,
        "intensity_code": parts[3],
        "statement": int(parts[4]),
        "repetition": int(parts[5]),
        "actor_id": int(parts[6]),
    }


def scan_ravdess_files(root: str | Path) -> list[dict[str, Any]]:
    root_path = _to_path(root)
    if root_path is None:
        raise ValueError("RAVDESS root path is required")
    wav_paths = sorted(root_path.rglob("*.wav"))
    samples: list[dict[str, Any]] = []
    for wav_path in wav_paths:
        info = _extract_ravdess_codes(wav_path)
        samples.append(
            {
                "dataset": "RAVDESS",
                "path": wav_path,
                "actor_id": info["actor_id"],
                "emotion_code": info["emotion_code"],
                "label": RAVDESS_LABELS[info["emotion_code"]],
                "statement": info["statement"],
                "repetition": info["repetition"],
                "modality_code": info["modality_code"],
                "channel_code": info["channel_code"],
                "intensity_code": info["intensity_code"],
            }
        )
    return samples


def list_ravdess_actors(root: str | Path) -> list[int]:
    actors = {sample["actor_id"] for sample in scan_ravdess_files(root)}
    return sorted(actors)


def generate_ravdess_splits(
    root: str | Path,
    *,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 1337,
) -> dict[str, list[dict[str, Any]]]:
    samples = scan_ravdess_files(root)
    return split_items_deterministically(
        samples,
        ratios=ratios,
        seed=seed,
        stratify_key=lambda sample: sample["label"],
    )


def _filter_samples_by_actors(
    samples: Sequence[dict[str, Any]],
    actors: Sequence[int | str] | None,
) -> list[dict[str, Any]]:
    if not actors:
        return list(samples)
    actor_set = {int(actor) for actor in actors}
    return [sample for sample in samples if sample["actor_id"] in actor_set]


def build_ravdess_actor_split(
    root: str | Path,
    *,
    train_actors: Sequence[int | str],
    val_actors: Sequence[int | str],
    test_actors: Sequence[int | str],
) -> dict[str, list[dict[str, Any]]]:
    samples = scan_ravdess_files(root)
    return {
        "train": _filter_samples_by_actors(samples, train_actors),
        "val": _filter_samples_by_actors(samples, val_actors),
        "test": _filter_samples_by_actors(samples, test_actors),
    }


def build_ravdess_fold_splits(
    root: str | Path,
    *,
    test_actor: int | str,
    val_actor: int | str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    actors = list_ravdess_actors(root)
    normalized_test = int(test_actor)
    if normalized_test not in actors:
        raise ValueError(f"Unknown RAVDESS test actor: {test_actor}")
    if val_actor is None:
        candidate_actors = [actor for actor in actors if actor != normalized_test]
        if not candidate_actors:
            raise ValueError("Could not infer a validation actor for RAVDESS fold.")
        val_actor = candidate_actors[0]
    normalized_val = int(val_actor)
    if normalized_val == normalized_test:
        raise ValueError("Validation actor and test actor must differ.")
    train_actors = [actor for actor in actors if actor not in {normalized_test, normalized_val}]
    return build_ravdess_actor_split(
        root,
        train_actors=train_actors,
        val_actors=[normalized_val],
        test_actors=[normalized_test],
    )


def default_ravdess_loso_folds(root: str | Path) -> list[dict[str, list[int]]]:
    actors = list_ravdess_actors(root)
    if len(actors) < 3:
        raise ValueError("RAVDESS LOSO folds require at least three actors.")
    folds = []
    for index, test_actor in enumerate(actors):
        val_actor = actors[(index + 1) % len(actors)]
        if val_actor == test_actor:
            val_actor = actors[(index + 2) % len(actors)]
        train_actors = [actor for actor in actors if actor not in {test_actor, val_actor}]
        folds.append(
            {
                "train_actors": train_actors,
                "val_actors": [val_actor],
                "test_actors": [test_actor],
            }
        )
    return folds


def _waveform_time_shift(waveform: torch.Tensor, max_fraction: float) -> torch.Tensor:
    if waveform.numel() == 0 or max_fraction <= 0:
        return waveform
    max_shift = int(waveform.numel() * max_fraction)
    if max_shift <= 0:
        return waveform
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    return torch.roll(waveform, shifts=shift, dims=0)


def _waveform_gain_jitter(waveform: torch.Tensor, max_db: float) -> torch.Tensor:
    if waveform.numel() == 0 or max_db <= 0:
        return waveform
    gain_db = float(torch.empty(1).uniform_(-max_db, max_db).item())
    gain = 10.0 ** (gain_db / 20.0)
    return waveform * gain


def _waveform_gaussian_noise(waveform: torch.Tensor, std: float) -> torch.Tensor:
    if waveform.numel() == 0 or std <= 0:
        return waveform
    return waveform + torch.randn_like(waveform) * std


def _spec_augment(
    features: torch.Tensor,
    *,
    time_mask_width: int = 0,
    freq_mask_width: int = 0,
    num_time_masks: int = 1,
    num_freq_masks: int = 1,
) -> torch.Tensor:
    if features.numel() == 0:
        return features
    augmented = features.clone()
    time_steps, freq_bins = augmented.shape[0], augmented.shape[1]

    if time_mask_width > 0 and time_steps > 1:
        max_width = min(time_mask_width, time_steps)
        for _ in range(max(1, num_time_masks)):
            width = int(torch.randint(0, max_width + 1, (1,)).item())
            if width == 0 or width >= time_steps:
                continue
            start = int(torch.randint(0, time_steps - width + 1, (1,)).item())
            augmented[start : start + width] = 0.0

    if freq_mask_width > 0 and freq_bins > 1:
        max_width = min(freq_mask_width, freq_bins)
        for _ in range(max(1, num_freq_masks)):
            width = int(torch.randint(0, max_width + 1, (1,)).item())
            if width == 0 or width >= freq_bins:
                continue
            start = int(torch.randint(0, freq_bins - width + 1, (1,)).item())
            augmented[:, start : start + width] = 0.0

    return augmented


class SAVEEDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "dataset/SAVEE/ALL",
        *,
        split: str = "train",
        speakers: Sequence[str] | None = None,
        ratios: Sequence[float] = (0.8, 0.1, 0.1),
        seed: int = 1337,
        sample_rate: int = 16000,
        target_sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        augmentations: Mapping[str, Any] | None = None,
        audio_feature_root: str | Path | None = None,
    ) -> None:
        self.root = _to_path(root)
        self.split = split
        self.speakers = [speaker.upper() for speaker in speakers] if speakers else None
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.labels = SAVEE_LABELS
        self.label_vocab = SAVEE_LABELS
        self.augmentations = dict(augmentations or {})
        self.enable_augment = bool(self.augmentations) and split.lower() == "train"
        self.audio_feature_root = _to_path(audio_feature_root)

        if self.speakers:
            self.samples = _filter_samples_by_speakers(scan_savee_files(self.root), self.speakers)
        else:
            splits = generate_savee_splits(self.root, ratios=ratios, seed=seed)
            if split in ("all", "*", ""):
                self.samples = [sample for name in ("train", "val", "test") for sample in splits[name]]
            else:
                if split not in splits:
                    raise ValueError(f"Unknown SAVEE split: {split}")
                self.samples = list(splits[split])
        if split not in ("all", "*", "") and self.speakers:
            self.samples = sorted(self.samples, key=lambda sample: str(sample["path"]))
        else:
            self.samples = sorted(self.samples, key=lambda sample: str(sample["path"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        path = Path(sample["path"])
        sample_id = path.stem

        # Try loading pretrained audio features first
        pretrained_audio = _load_optional_feature(
            self.audio_feature_root, split=self.split, sample_id=sample_id
        ) if self.audio_feature_root is not None else torch.empty(0)

        if pretrained_audio.numel() > 0:
            audio_tokens = pretrained_audio
        else:
            waveform, sample_rate = load_wav(sample["path"])
            if self.enable_augment:
                waveform = _waveform_time_shift(waveform, float(self.augmentations.get("time_shift_fraction", 0.0)))
                waveform = _waveform_gain_jitter(waveform, float(self.augmentations.get("gain_jitter_db", 0.0)))
                waveform = _waveform_gaussian_noise(waveform, float(self.augmentations.get("noise_std", 0.0)))
            audio_tokens = log_mel_spectrogram(
                waveform,
                sample_rate,
                target_sample_rate=self.target_sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
            )
            if self.enable_augment:
                audio_tokens = _spec_augment(
                    audio_tokens,
                    time_mask_width=int(self.augmentations.get("time_mask_width", 0)),
                    freq_mask_width=int(self.augmentations.get("freq_mask_width", 0)),
                    num_time_masks=int(self.augmentations.get("num_time_masks", 1)),
                    num_freq_masks=int(self.augmentations.get("num_freq_masks", 1)),
                )

        meta = {
            "dataset": "SAVEE",
            "split": self.split,
            "path": str(path),
            "speaker": sample["speaker"],
            "utterance_id": sample["utterance_id"],
            "emotion_code": sample["emotion_code"],
            "filename": path.name,
        }
        return {
            "audio_tokens": audio_tokens,
            "text_tokens": _empty_text_tokens(),
            "visual_tokens": _empty_float_tokens(),
            "label": int(sample["label"]),
            "meta": meta,
        }


class RAVDESSDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "dataset/RAVDESS",
        *,
        split: str = "train",
        actors: Sequence[int | str] | None = None,
        ratios: Sequence[float] = (0.8, 0.1, 0.1),
        seed: int = 1337,
        sample_rate: int = 16000,
        target_sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        augmentations: Mapping[str, Any] | None = None,
        audio_feature_root: str | Path | None = None,
    ) -> None:
        self.root = _to_path(root)
        self.split = split
        self.actors = [int(actor) for actor in actors] if actors else None
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.labels = RAVDESS_LABELS
        self.label_vocab = RAVDESS_LABELS
        self.augmentations = dict(augmentations or {})
        self.enable_augment = bool(self.augmentations) and split.lower() == "train"
        self.audio_feature_root = _to_path(audio_feature_root)

        if self.actors:
            self.samples = _filter_samples_by_actors(scan_ravdess_files(self.root), self.actors)
        else:
            splits = generate_ravdess_splits(self.root, ratios=ratios, seed=seed)
            if split in ("all", "*", ""):
                self.samples = [sample for name in ("train", "val", "test") for sample in splits[name]]
            else:
                if split not in splits:
                    raise ValueError(f"Unknown RAVDESS split: {split}")
                self.samples = list(splits[split])
        self.samples = sorted(self.samples, key=lambda sample: str(sample["path"]))
        if not self.samples:
            raise FileNotFoundError(f"No RAVDESS wav files found under {self.root}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        path = Path(sample["path"])
        sample_id = path.stem

        # Try loading pretrained audio features first
        pretrained_audio = _load_optional_feature(
            self.audio_feature_root, split=self.split, sample_id=sample_id
        ) if self.audio_feature_root is not None else torch.empty(0)

        if pretrained_audio.numel() > 0:
            audio_tokens = pretrained_audio
        else:
            waveform, sample_rate = load_wav(sample["path"])
            if self.enable_augment:
                waveform = _waveform_time_shift(waveform, float(self.augmentations.get("time_shift_fraction", 0.0)))
                waveform = _waveform_gain_jitter(waveform, float(self.augmentations.get("gain_jitter_db", 0.0)))
                waveform = _waveform_gaussian_noise(waveform, float(self.augmentations.get("noise_std", 0.0)))
            audio_tokens = log_mel_spectrogram(
                waveform,
                sample_rate,
                target_sample_rate=self.target_sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
            )
            if self.enable_augment:
                audio_tokens = _spec_augment(
                    audio_tokens,
                    time_mask_width=int(self.augmentations.get("time_mask_width", 0)),
                    freq_mask_width=int(self.augmentations.get("freq_mask_width", 0)),
                    num_time_masks=int(self.augmentations.get("num_time_masks", 1)),
                    num_freq_masks=int(self.augmentations.get("num_freq_masks", 1)),
                )

        meta = {
            "dataset": "RAVDESS",
            "split": self.split,
            "path": str(path),
            "actor_id": sample["actor_id"],
            "emotion_code": sample["emotion_code"],
            "emotion_name": RAVDESS_LABEL_NAMES[int(sample["label"])],
            "statement": sample["statement"],
            "repetition": sample["repetition"],
            "modality_code": sample["modality_code"],
            "channel_code": sample["channel_code"],
            "intensity_code": sample["intensity_code"],
            "filename": path.name,
        }
        return {
            "audio_tokens": audio_tokens,
            "text_tokens": _empty_text_tokens(),
            "visual_tokens": _empty_float_tokens(),
            "label": int(sample["label"]),
            "meta": meta,
        }


def _resolve_meld_csv_path(data_root: str | Path, split: str) -> Path:
    data_root_path = _to_path(data_root)
    if data_root_path is None:
        raise ValueError("MELD data root is required")
    normalized = split.lower()
    mapping = {
        "train": "train_sent_emo.csv",
        "val": "dev_sent_emo.csv",
        "dev": "dev_sent_emo.csv",
        "test": "test_sent_emo.csv",
    }
    if normalized not in mapping:
        raise ValueError(f"Unknown MELD split: {split}")
    return data_root_path / mapping[normalized]


def read_meld_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    path = _to_path(csv_path)
    if path is None or not path.exists():
        raise FileNotFoundError(f"MELD CSV not found: {csv_path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            emotion = str(row["Emotion"]).strip().lower()
            if emotion not in MELD_LABELS:
                continue
            records.append(
                {
                    "dataset": "MELD",
                    "csv_path": path,
                    "split": path.stem.replace("_sent_emo", "").replace("_sent_emo_dya", ""),
                    "row": row,
                    "label": MELD_LABELS[emotion],
                    "emotion": emotion,
                    "sentiment": str(row.get("Sentiment", "")).strip().lower(),
                    "speaker": str(row.get("Speaker", "")).strip(),
                    "dialogue_id": int(row.get("Dialogue_ID", 0)),
                    "utterance_id": int(row.get("Utterance_ID", 0)),
                    "season": int(row.get("Season", 0) or 0),
                    "episode": int(row.get("Episode", 0) or 0),
                    "start_time": str(row.get("StartTime", "")).strip(),
                    "end_time": str(row.get("EndTime", "")).strip(),
                    "text": str(row.get("Utterance", "")).strip(),
                }
            )
    return records


def _feature_candidates(
    root: Path | None,
    *,
    split: str,
    sample_id: str,
) -> list[Path]:
    if root is None:
        return []
    split = split.lower()
    return [
        root / f"{sample_id}.pt",
        root / f"{sample_id}.pth",
        root / split / f"{sample_id}.pt",
        root / split / f"{sample_id}.pth",
        root / f"{split}_{sample_id}.pt",
        root / f"{split}_{sample_id}.pth",
    ]


def _load_optional_feature(root: str | Path | None, *, split: str, sample_id: str) -> torch.Tensor:
    root_path = _to_path(root)
    for candidate in _feature_candidates(root_path, split=split, sample_id=sample_id):
        if candidate.exists():
            tensor = load_tensor_feature(candidate)
            if tensor.numel() > 0:
                return tensor
    return torch.empty(0)


class MELDUtteranceDataset(Dataset):
    def __init__(
        self,
        *,
        split: str = "train",
        data_root: str | Path = "dataset/Meld/MELD-master/data/MELD",
        max_records: int | None = None,
        vocab: SimpleVocab | None = None,
        tokenizer: Callable[[str], list[str]] = tokenize_text,
        lowercase: bool = True,
        max_text_length: int | None = None,
        audio_feature_root: str | Path | None = None,
        visual_feature_root: str | Path | None = None,
        text_feature_root: str | Path | None = None,
        build_vocab_if_missing: bool = True,
    ) -> None:
        self.split = split.lower()
        self.data_root = _to_path(data_root)
        self.csv_path = _resolve_meld_csv_path(self.data_root, split)
        self.records = read_meld_csv(self.csv_path)
        if max_records is not None:
            self.records = self.records[: int(max_records)]
        self.tokenizer = tokenizer
        self.lowercase = lowercase
        self.max_text_length = max_text_length
        self.audio_feature_root = _to_path(audio_feature_root)
        self.visual_feature_root = _to_path(visual_feature_root)
        self.text_feature_root = _to_path(text_feature_root)

        if vocab is None and build_vocab_if_missing:
            vocab = SimpleVocab.build_from_texts(record["text"] for record in self.records)
        self.vocab = vocab
        self.labels = MELD_LABELS
        self.label_vocab = MELD_LABELS

    def __len__(self) -> int:
        return len(self.records)

    def _encode_text(self, text: str) -> torch.Tensor:
        if self.vocab is not None:
            tokens = self.vocab.encode(text, lowercase=self.lowercase)
        else:
            token_list = self.tokenizer(text.lower() if self.lowercase else text)
            if not token_list:
                return _empty_text_tokens()
            tokens = torch.arange(len(token_list), dtype=torch.long)
        if self.max_text_length is not None:
            tokens = tokens[: self.max_text_length]
        return tokens

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample_id = f"dia{record['dialogue_id']}_utt{record['utterance_id']}"

        # Try loading pretrained text features first (float tensor)
        pretrained_text = _load_optional_feature(
            self.text_feature_root, split=self.split, sample_id=sample_id
        ) if self.text_feature_root is not None else torch.empty(0)
        if pretrained_text.numel() > 0:
            text_tokens = pretrained_text
            if self.max_text_length is not None and text_tokens.size(0) > self.max_text_length:
                text_tokens = text_tokens[: self.max_text_length]
        else:
            text_tokens = self._encode_text(record["text"])

        audio_tokens = _load_optional_feature(self.audio_feature_root, split=self.split, sample_id=sample_id)
        visual_tokens = _load_optional_feature(self.visual_feature_root, split=self.split, sample_id=sample_id)

        meta = {
            "dataset": "MELD",
            "split": self.split,
            "csv_path": str(record["csv_path"]),
            "sample_id": sample_id,
            "dialogue_id": record["dialogue_id"],
            "utterance_id": record["utterance_id"],
            "speaker": record["speaker"],
            "season": record["season"],
            "episode": record["episode"],
            "emotion": record["emotion"],
            "sentiment": record["sentiment"],
            "start_time": record["start_time"],
            "end_time": record["end_time"],
            "text": record["text"],
        }

        return {
            "audio_tokens": audio_tokens if audio_tokens.numel() > 0 else _empty_float_tokens(),
            "text_tokens": text_tokens if text_tokens.numel() > 0 else _empty_text_tokens(),
            "visual_tokens": visual_tokens if visual_tokens.numel() > 0 else _empty_float_tokens(),
            "label": int(record["label"]),
            "meta": meta,
        }


def build_datasets(config: Mapping[str, Any]) -> dict[str, Dataset]:
    dataset_type = str(config.get("type", config.get("name", ""))).lower()
    if dataset_type == "savee":
        root = config.get("root", "dataset/SAVEE/ALL")
        common_kwargs = {
            "root": root,
            "seed": int(config.get("seed", config.get("split_seed", 1337))),
            "target_sample_rate": int(config.get("target_sample_rate", 16000)),
            "n_fft": int(config.get("n_fft", 400)),
            "hop_length": int(config.get("hop_length", 160)),
            "win_length": int(config.get("win_length", 400)),
            "n_mels": int(config.get("n_mels", 80)),
            "audio_feature_root": config.get("audio_feature_root"),
        }
        augmentations = config.get("augmentations") or {}
        if any(key in config for key in ("train_speakers", "val_speakers", "test_speakers")):
            return {
                "train": SAVEEDataset(
                    split="train",
                    speakers=config.get("train_speakers"),
                    augmentations=augmentations,
                    **common_kwargs,
                ),
                "val": SAVEEDataset(
                    split="val",
                    speakers=config.get("val_speakers"),
                    augmentations=None,
                    **common_kwargs,
                ),
                "test": SAVEEDataset(
                    split="test",
                    speakers=config.get("test_speakers"),
                    augmentations=None,
                    **common_kwargs,
                ),
            }
        if "test_speaker" in config:
            fold = build_savee_fold_splits(
                root,
                test_speaker=str(config["test_speaker"]),
                val_speaker=str(config["val_speaker"]) if config.get("val_speaker") else None,
            )
            return {
                "train": SAVEEDataset(
                    split="train",
                    speakers=[sample["speaker"] for sample in fold["train"]],
                    augmentations=augmentations,
                    **common_kwargs,
                ),
                "val": SAVEEDataset(
                    split="val",
                    speakers=[sample["speaker"] for sample in fold["val"]],
                    augmentations=None,
                    **common_kwargs,
                ),
                "test": SAVEEDataset(
                    split="test",
                    speakers=[sample["speaker"] for sample in fold["test"]],
                    augmentations=None,
                    **common_kwargs,
                ),
            }
        return {
            "train": SAVEEDataset(split="train", augmentations=augmentations, **common_kwargs),
            "val": SAVEEDataset(split="val", augmentations=None, **common_kwargs),
            "test": SAVEEDataset(split="test", augmentations=None, **common_kwargs),
        }

    if dataset_type == "meld":
        root = config.get("data_root", config.get("root", "dataset/Meld/MELD-master/data/MELD"))
        build_vocab_if_missing = bool(config.get("build_vocab_if_missing", True))
        common_kwargs = {
            "data_root": root,
            "audio_feature_root": config.get("audio_feature_root", config.get("audio_feature_dir")),
            "visual_feature_root": config.get("visual_feature_root", config.get("visual_feature_dir")),
            "text_feature_root": config.get("text_feature_root"),
            "max_text_length": config.get("max_text_length"),
        }
        train_dataset = MELDUtteranceDataset(
            split="train",
            max_records=config.get("max_train_records"),
            build_vocab_if_missing=build_vocab_if_missing,
            **common_kwargs,
        )
        vocab = train_dataset.vocab
        val_dataset = MELDUtteranceDataset(
            split="dev",
            max_records=config.get("max_val_records"),
            vocab=vocab,
            build_vocab_if_missing=False,
            **common_kwargs,
        )
        test_dataset = MELDUtteranceDataset(
            split="test",
            max_records=config.get("max_test_records"),
            vocab=vocab,
            build_vocab_if_missing=False,
            **common_kwargs,
        )
        return {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    if dataset_type == "ravdess":
        root = config.get("root", "dataset/RAVDESS")
        common_kwargs = {
            "root": root,
            "seed": int(config.get("seed", config.get("split_seed", 1337))),
            "target_sample_rate": int(config.get("target_sample_rate", 16000)),
            "n_fft": int(config.get("n_fft", 400)),
            "hop_length": int(config.get("hop_length", 160)),
            "win_length": int(config.get("win_length", 400)),
            "n_mels": int(config.get("n_mels", 80)),
            "audio_feature_root": config.get("audio_feature_root"),
        }
        augmentations = config.get("augmentations") or {}
        if any(key in config for key in ("train_actors", "val_actors", "test_actors")):
            return {
                "train": RAVDESSDataset(
                    split="train",
                    actors=config.get("train_actors"),
                    augmentations=augmentations,
                    **common_kwargs,
                ),
                "val": RAVDESSDataset(
                    split="val",
                    actors=config.get("val_actors"),
                    augmentations=None,
                    **common_kwargs,
                ),
                "test": RAVDESSDataset(
                    split="test",
                    actors=config.get("test_actors"),
                    augmentations=None,
                    **common_kwargs,
                ),
            }
        if "test_actor" in config:
            fold = build_ravdess_fold_splits(
                root,
                test_actor=config["test_actor"],
                val_actor=config.get("val_actor"),
            )
            return {
                "train": RAVDESSDataset(
                    split="train",
                    actors=fold["train_actors"],
                    augmentations=augmentations,
                    **common_kwargs,
                ),
                "val": RAVDESSDataset(
                    split="val",
                    actors=fold["val_actors"],
                    augmentations=None,
                    **common_kwargs,
                ),
                "test": RAVDESSDataset(
                    split="test",
                    actors=fold["test_actors"],
                    augmentations=None,
                    **common_kwargs,
                ),
            }
        return {
            "train": RAVDESSDataset(split="train", augmentations=augmentations, **common_kwargs),
            "val": RAVDESSDataset(split="val", augmentations=None, **common_kwargs),
            "test": RAVDESSDataset(split="test", augmentations=None, **common_kwargs),
        }

    raise ValueError(f"Unsupported dataset type: {dataset_type!r}")

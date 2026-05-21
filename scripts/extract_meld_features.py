from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gs3_icl.audio import load_wav, log_mel_spectrogram


MELD_CSV_ROOT = ROOT / "dataset" / "Meld" / "MELD-master" / "data" / "MELD"
MELD_RAW_ROOT = ROOT / "dataset" / "Meld" / "MELD.Raw.tar" / "MELD.Raw" / "MELD.Raw"
FEATURE_ROOT = ROOT / "dataset" / "Meld" / "features"

CSV_PATHS = {
    "train": MELD_CSV_ROOT / "train_sent_emo.csv",
    "dev": MELD_CSV_ROOT / "dev_sent_emo.csv",
    "test": MELD_CSV_ROOT / "test_sent_emo.csv",
}

MP4_DIRS = {
    "train": MELD_RAW_ROOT / "train.tar" / "train" / "train_splits",
    "dev": MELD_RAW_ROOT / "dev.tar" / "dev" / "dev_splits_complete",
    "test": MELD_RAW_ROOT / "test.tar" / "test" / "output_repeated_splits_test",
}


def sample_id(dialogue_id: int, utterance_id: int) -> str:
    return f"dia{dialogue_id}_utt{utterance_id}"


def read_split_rows(split: str) -> list[tuple[int, int]]:
    csv_path = CSV_PATHS[split]
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            (int(row["Dialogue_ID"]), int(row["Utterance_ID"]))
            for row in reader
        ]


def extract_audio_wav(mp4_path: Path, wav_path: Path) -> bool:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(wav_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=90, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def extract_audio_feature(mp4_path: Path, wav_path: Path) -> torch.Tensor:
    if not extract_audio_wav(mp4_path, wav_path):
        return torch.empty(0)
    waveform, sample_rate = load_wav(wav_path)
    return log_mel_spectrogram(waveform, sample_rate, n_mels=80)


def build_visual_encoder(device: str):
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


def load_frame(path: Path, device: str) -> torch.Tensor:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    array = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((array - mean) / std).to(device)


def extract_frames(mp4_path: Path, frame_dir: Path, fps: int, image_size: int) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = frame_dir / "frame_%05d.jpg"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4_path),
        "-vf",
        f"fps={fps},scale={image_size}:{image_size}",
        str(pattern),
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return sorted(frame_dir.glob("*.jpg"))


def extract_visual_feature(
    mp4_path: Path,
    frame_dir: Path,
    model,
    device: str,
    fps: int,
    image_size: int,
    max_frames: int,
) -> torch.Tensor:
    frame_paths = extract_frames(mp4_path, frame_dir, fps=fps, image_size=image_size)
    if not frame_paths:
        return torch.empty(0)
    if len(frame_paths) > max_frames:
        indices = torch.linspace(0, len(frame_paths) - 1, max_frames).long().tolist()
        frame_paths = [frame_paths[index] for index in indices]

    features = []
    with torch.no_grad():
        for frame_path in frame_paths:
            image = load_frame(frame_path, device).unsqueeze(0)
            features.append(model(image).squeeze(0).cpu())
    if not features:
        return torch.empty(0)
    return torch.stack(features, dim=0)


def process_split(args: argparse.Namespace, split: str) -> None:
    rows = read_split_rows(split)
    audio_dir = FEATURE_ROOT / "audio" / split
    visual_dir = FEATURE_ROOT / "visual" / split
    if args.audio:
        audio_dir.mkdir(parents=True, exist_ok=True)
    if args.visual:
        visual_dir.mkdir(parents=True, exist_ok=True)
        visual_model = build_visual_encoder(args.device)
    else:
        visual_model = None

    work_root = Path(tempfile.mkdtemp(prefix=f"meld_{split}_"))
    saved = skipped = failed = 0
    try:
        for dialogue_id, utterance_id in rows:
            sid = sample_id(dialogue_id, utterance_id)
            mp4_path = MP4_DIRS[split] / f"{sid}.mp4"
            if not mp4_path.exists():
                skipped += 1
                continue

            if args.audio:
                output_path = audio_dir / f"{sid}.pt"
                if output_path.exists() and not args.overwrite:
                    skipped += 1
                else:
                    feature = extract_audio_feature(mp4_path, work_root / f"{sid}.wav")
                    if feature.numel() > 0:
                        torch.save(feature, output_path)
                        saved += 1
                    else:
                        failed += 1

            if args.visual and visual_model is not None:
                output_path = visual_dir / f"{sid}.pt"
                if output_path.exists() and not args.overwrite:
                    skipped += 1
                else:
                    frame_dir = work_root / f"{sid}_frames"
                    feature = extract_visual_feature(
                        mp4_path,
                        frame_dir,
                        visual_model,
                        args.device,
                        args.visual_fps,
                        args.visual_size,
                        args.max_visual_frames,
                    )
                    if feature.numel() > 0:
                        torch.save(feature, output_path)
                        saved += 1
                    else:
                        failed += 1
                    shutil.rmtree(frame_dir, ignore_errors=True)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    print(f"{split}: saved={saved} skipped={skipped} failed={failed}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract MELD log-mel and visual frame features.")
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visual-fps", type=int, default=3)
    parser.add_argument("--visual-size", type=int, default=224)
    parser.add_argument("--max-visual-frames", type=int, default=16)
    args = parser.parse_args()

    if not args.audio and not args.visual:
        parser.error("select at least one of --audio or --visual")

    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]
    for split in splits:
        process_split(args, split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

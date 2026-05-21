from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    value = path if isinstance(path, Path) else Path(path).expanduser()
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def project_relative(path: str | Path) -> str:
    value = resolve_project_path(path)
    if value is None:
        return ""
    try:
        return value.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(value)

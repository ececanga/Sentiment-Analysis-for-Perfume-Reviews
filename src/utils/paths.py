# Date Created: 2026-03-02 11:07:58+03:00
# Date Last Modified: 2026-05-15

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LABELS_DIR = PROJECT_ROOT / "labels"
DOCS_DIR = PROJECT_ROOT / "docs"


def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def outputs_path(*parts: str) -> Path:
    return OUTPUTS_DIR.joinpath(*parts)


def labels_path(*parts: str) -> Path:
    return LABELS_DIR.joinpath(*parts)


def docs_path(*parts: str) -> Path:
    return DOCS_DIR.joinpath(*parts)

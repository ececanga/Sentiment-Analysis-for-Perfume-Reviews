#!/usr/bin/env python3
# Date Created: 2026-05-15 14:55:00 +03:00
# Date Last Modified: 2026-05-15

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path


SUPPORTED_SUFFIXES = {".py", ".md", ".yaml", ".yml"}
TODAY = date.today().isoformat()


def repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def tracked_supported_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    files: list[Path] = []
    for line in result.stdout.splitlines():
        path = root / line
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return files


def git_created_timestamp(root: Path, path: Path) -> str:
    relative_path = path.relative_to(root)
    result = subprocess.run(
        ["git", "log", "--follow", "--format=%aI", "--", str(relative_path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    commits = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if commits:
        return commits[-1].replace("T", " ")
    return TODAY


def header_block(path: Path, created_timestamp: str) -> str:
    if path.suffix.lower() == ".md":
        return (
            "<!--\n"
            f"Date Created: {created_timestamp}\n"
            f"Date Last Modified: {TODAY}\n"
            "-->\n\n"
        )
    return (
        f"# Date Created: {created_timestamp}\n"
        f"# Date Last Modified: {TODAY}\n\n"
    )


def strip_existing_markdown_header(text: str) -> str:
    lines = text.splitlines(keepends=True)
    if len(lines) < 4:
        return text
    if lines[0] != "<!--\n":
        return text
    if not lines[1].startswith("Date Created: "):
        return text
    if not lines[2].startswith("Date Last Modified: "):
        return text
    if lines[3] != "-->\n":
        return text
    stripped = lines[4:]
    while stripped and stripped[0].strip() == "":
        stripped = stripped[1:]
    return "".join(stripped)


def strip_existing_hash_header(lines: list[str]) -> list[str]:
    if len(lines) < 2:
        return lines
    if not lines[0].startswith("# Date Created: "):
        return lines
    if not lines[1].startswith("# Date Last Modified: "):
        return lines
    stripped = lines[2:]
    while stripped and stripped[0].strip() == "":
        stripped = stripped[1:]
    return stripped


def update_hash_comment_file(path: Path, created_timestamp: str) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    prefix: list[str] = []
    start = 0
    if start < len(lines) and lines[start].startswith("#!"):
        prefix.append(lines[start])
        start += 1
    if start < len(lines) and "coding" in lines[start]:
        prefix.append(lines[start])
        start += 1

    body = strip_existing_hash_header(lines[start:])
    updated = "".join(prefix) + header_block(path, created_timestamp) + "".join(body)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def update_markdown_file(path: Path, created_timestamp: str) -> bool:
    original = path.read_text(encoding="utf-8")
    body = strip_existing_markdown_header(original)
    updated = header_block(path, created_timestamp) + body
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def update_file(root: Path, path: Path) -> bool:
    created_timestamp = git_created_timestamp(root, path)
    if path.suffix.lower() == ".md":
        return update_markdown_file(path, created_timestamp)
    return update_hash_comment_file(path, created_timestamp)


def main() -> int:
    root = repo_root()
    updated_files: list[Path] = []

    for path in tracked_supported_files(root):
        if update_file(root, path):
            updated_files.append(path.relative_to(root))

    print(f"Updated {len(updated_files)} files.")
    for path in updated_files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Prepare the datasets used for GIPACount reproduction.

Currently supported:

* ``fsc147``  -- the paper's main benchmark (§4.1).  The three small metadata
  files are fetched from the official GitHub release; the image archive is large
  and lives on Google Drive, so it must be supplied either as a directly
  downloadable URL (``--image-url``) or as an already-downloaded zip
  (``--images-zip``).
* ``synthetic`` -- nothing to download (generated on the fly, used by the smoke
  test and CI).

The script is deliberately dependency-light (stdlib only) so it runs before the
project is installed.  Nothing is written outside ``--root``.

Examples::

    python scripts/download_data.py --dataset fsc147 --root data/fsc147
    python scripts/download_data.py --dataset fsc147 --root data/fsc147 \\
        --images-zip ~/Downloads/FSC147_384_V2.zip
    python scripts/download_data.py --dataset fsc147 --images-url https://.../fsc147.zip
    python scripts/download_data.py --dataset synthetic
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# Small, stable metadata files from the official FSC-147 release.
_FSC147_BASE = (
    "https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/master/data/"
)
_FSC147_FILES: dict[str, str] = {
    "Train_Test_Val_FSC_147.json": _FSC147_BASE + "Train_Test_Val_FSC_147.json",
    "annotation_FSC147_384.json": _FSC147_BASE + "annotation_FSC147_384.json",
    "ImageClasses_FSC147.txt": _FSC147_BASE + "ImageClasses_FSC147.txt",
}

#: Expected image directory inside the FSC-147 image archive.
_FSC147_IMAGE_DIR = "images_384_VarV2"


def _log(message: str) -> None:
    """Print a progress line (this script runs before logging is configured)."""
    print(message, file=sys.stderr)


def _download(url: str, destination: Path, *, force: bool = False) -> Path:
    """Download ``url`` to ``destination`` (skipping when it already exists)."""
    if destination.exists() and not force:
        _log(f"  already present: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    _log(f"  downloading {url}\n             -> {destination}")
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def _extract_zip(archive: Path, root: Path, *, force: bool = False) -> None:
    """Extract ``archive`` into ``root``, stripping the top-level directory."""
    _log(f"  extracting {archive} -> {root}")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = _strip_common_prefix(member.filename, root)
            if target is None:
                continue
            if target.exists() and not force:
                continue
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)


def _strip_common_prefix(name: str, root: Path) -> Path | None:
    """Map an archive member to a path under ``root``, dropping one wrapper dir."""
    parts = [p for p in Path(name).parts if p not in {".", "/"}]
    if not parts:
        return None
    if parts[0] in {_FSC147_IMAGE_DIR, "FSC147", "FSC147_384_V2", "data"}:
        parts = parts[1:]
    if not parts:
        return None
    return root.joinpath(*parts)


def prepare_fsc147(args: argparse.Namespace) -> int:
    """Download/prepare FSC-147 under ``args.root``.  Returns an exit code."""
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)

    _log(f"FSC-147 metadata -> {root}")
    for name, url in _FSC147_FILES.items():
        try:
            _download(url, root / name, force=args.force)
        except OSError as exc:
            _log(f"  failed: {exc}")
            return 1

    images_dir = root / _FSC147_IMAGE_DIR
    if images_dir.is_dir() and any(images_dir.iterdir()) and not args.force:
        _log(f"images already present under {images_dir}")
    elif args.images_zip is not None:
        if not args.images_zip.is_file():
            _log(f"error: --images-zip not found: {args.images_zip}")
            return 2
        _extract_zip(args.images_zip, root, force=args.force)
    elif args.images_url is not None:
        archive = root / "fsc147_images.zip"
        try:
            _download(args.images_url, archive, force=args.force)
        except OSError as exc:
            _log(f"  failed to download images: {exc}")
            return 1
        _extract_zip(archive, root, force=args.force)
    else:
        _log(
            "\nThe image archive is not fetched automatically (it is large and is hosted\n"
            "on Google Drive, whose links are unstable). Download it manually and either\n"
            "unzip it into the root, or re-run this script with --images-zip:\n\n"
            "  https://github.com/cvlab-stonybrook/LearningToCountEverything\n"
            "  (file: FSC147_384_V2.zip)  ->  images_384_VarV2/\n"
        )

    _log("\nverification:")
    for name in (*_FSC147_FILES, _FSC147_IMAGE_DIR):
        path = root / name
        _log(f"  [{'ok' if path.exists() else 'missing'}] {path}")
    return 0


def prepare_synthetic(args: argparse.Namespace) -> int:
    """Synthetic data needs no preparation."""
    _log("synthetic dataset: generated on the fly; nothing to download.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for this script."""
    parser = argparse.ArgumentParser(description="Prepare GIPACount datasets.")
    parser.add_argument("--dataset", choices=["fsc147", "synthetic"], default="fsc147")
    parser.add_argument("--root", type=Path, default=Path("data/fsc147"))
    parser.add_argument("--images-zip", type=Path, default=None, help="Local FSC-147 image zip.")
    parser.add_argument("--images-url", default=None, help="Direct URL to the image archive.")
    parser.add_argument("--force", action="store_true", help="Re-download/re-extract.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse arguments and prepare the requested dataset."""
    args = build_parser().parse_args(argv)
    if args.dataset == "synthetic":
        return prepare_synthetic(args)
    return prepare_fsc147(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

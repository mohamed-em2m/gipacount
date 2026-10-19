"""ECR (exemplar-consistent response ratio) evaluation set, Eq. (21).

The composite evaluation set is built exactly like the CCSS training collages —
four images stitched into a 2x2 layout with an anchor exemplar defining the
target category — except that no category filtering is applied to the density
targets and the set of exemplar-consistent regions ``Ω+`` is recorded so that

    ECR = (1/N_c) Σ_i  Σ_{Ω+_i} D̂ / (Σ_{Ω_i} D̂ + ε₀)

can be computed.  ``Ω+`` is the union of quadrants whose category matches the
anchor's; the remaining quadrants form the complement, so only ECR is reported.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Dataset

from gipacount.config import DataConfig
from gipacount.data.collate import CountingCollator
from gipacount.data.raw import RawSample
from gipacount.metrics import exemplar_consistent_response_ratio

__all__ = ["build_ecr_loader", "exemplar_consistent_response_ratio"]


def build_ecr_loader(
    cfg: DataConfig,
    dataset: Dataset[RawSample],
    *,
    n_composites: int | None = None,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader[Any]:
    """Loader that always emits one 2x2 composite per batch.

    Args:
        cfg: Data configuration (the CCSS grid defines the collage size).
        dataset: The split to draw images from.
        n_composites: Number of composites requested; the loader is sized so
            that one pass yields at least this many (rounded to a multiple of
            the collage size).  ``None`` uses the whole split.
        seed: Seed for the quadrant assignment.
        num_workers: Loader workers (0 keeps everything in-process).

    Returns:
        A ``DataLoader`` yielding single-composite :class:`CountingBatch` objects
        with ``positive_mask`` set.
    """
    n_cells = cfg.ccss.grid**2
    available = (len(dataset) // n_cells) * n_cells  # type: ignore[arg-type]
    wanted = available if n_composites is None else min(available, n_composites * n_cells)
    wanted = max(n_cells, min(wanted, available))

    subset: Dataset[RawSample] = torch_subset(dataset, wanted)
    collator = CountingCollator(cfg, train=False, seed=seed, force_mix=True)
    return DataLoader(
        subset,
        batch_size=n_cells,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=True,
    )


def torch_subset(dataset: Dataset[RawSample], length: int) -> Dataset[RawSample]:
    """A prefix ``Subset`` of ``dataset`` (kept import-light for typing)."""
    from torch.utils.data import Subset

    return Subset(dataset, list(range(min(length, len(dataset)))))  # type: ignore[arg-type]

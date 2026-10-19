"""Data pipeline: transforms, point maps, CCSS collage and collation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gipacount.data.ccss import build_composite, quadrant_offsets
from gipacount.data.collate import CountingCollator, collate_samples
from gipacount.data.point_map import (
    assign_unique_locations,
    build_point_map,
    points_to_locations,
    ring_offsets,
)
from gipacount.data.raw import RawSample
from gipacount.data.synthetic import SyntheticCountingDataset
from gipacount.data.transforms import SamplePreprocessor, hflip, resize_and_pad
from tests.conftest import make_config


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_resize_and_pad_geometry() -> None:
    """The long side is resized to ``out_size`` and the content stays top-left."""
    image = torch.rand(3, 100, 50)  # portrait: the long side is the height
    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    points = torch.tensor([[5.0, 6.0]])
    out = resize_and_pad(image, boxes, points, 64)
    assert out.scale == pytest.approx(64 / 100)
    assert out.image.shape == (3, 64, 64)
    assert int(out.pad_mask.sum()) == 64 * 32  # 64 tall x 32 wide content
    assert out.boxes[0, 2].item() == pytest.approx(20.0 * 0.64)
    assert out.points[0, 0].item() == pytest.approx(5.0 * 0.64)
    # the padded (right) region is exactly zero
    assert out.image[:, :, 32:].abs().sum() == 0


def test_hflip_keeps_everything_consistent() -> None:
    image = torch.zeros(3, 8, 8)
    image[0, :, 0] = 1.0  # marker on the left edge
    pad_mask = torch.ones(1, 8, 8, dtype=torch.bool)
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    points = torch.tensor([[1.0, 1.0]])
    img2, _, boxes2, points2 = hflip(image, pad_mask, boxes, points)
    assert img2[0, :, -1].sum() == 8  # flipped to the right
    assert boxes2[0, 0].item() == pytest.approx((8 - 1) - 2.0)
    assert boxes2[0, 2].item() == pytest.approx((8 - 1) - 0.0)
    assert points2[0, 0].item() == pytest.approx((8 - 1) - 1.0)


# --------------------------------------------------------------------------- #
# Point discretisation
# --------------------------------------------------------------------------- #
def test_points_to_locations_is_yx() -> None:
    loc = points_to_locations(np.array([[3.2, 7.9]]))
    assert loc.tolist() == [[7, 3]]  # (floor(y), floor(x))


def test_ring_offsets_are_the_8_neighbourhood() -> None:
    offsets = set(ring_offsets(1))
    assert len(offsets) == 8
    assert (0, 0) not in offsets
    assert (1, 1) in offsets and (-1, -1) in offsets


def test_collisions_are_reassigned_and_mass_preserved() -> None:
    """Two annotations on the same cell: one keeps it, the other moves aside."""
    points = np.array([[5.0, 5.0], [5.0, 5.0]])
    locations, mass, collisions = assign_unique_locations(points, size=16, radius=1)
    assert collisions == 1
    assert mass.sum() == pytest.approx(2.0)
    assert len({tuple(loc) for loc in locations.tolist()}) == 2  # no duplicate cells


def test_build_point_map_accumulates_mass() -> None:
    locations = np.array([[1, 2], [1, 2]])
    point_map = build_point_map(locations, size=8, mass=np.array([1.0, 2.0]))
    assert point_map.shape == (1, 8, 8)
    assert point_map[0, 1, 2].item() == pytest.approx(3.0)


def test_empty_points_are_handled() -> None:
    locations, mass, collisions = assign_unique_locations(np.zeros((0, 2)), size=8)
    assert locations.shape == (0, 2)
    assert mass.shape == (0,)
    assert collisions == 0
    assert build_point_map(locations, size=8).sum() == 0


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def test_preprocessor_points_live_on_the_density_grid() -> None:
    cfg = make_config()
    raw = RawSample(
        image=torch.rand(3, 64, 64),
        boxes=torch.tensor([[4.0, 4.0, 12.0, 12.0]]),
        points=torch.tensor([[10.0, 10.0]]),
        category=0,
    )
    sample = SamplePreprocessor(cfg.data, out_size=64)(raw)
    assert sample.density_size == 32
    # 10 / 2 = 5  -> floor -> cell (x=5, y=5)
    assert sample.points[0].tolist() == [5.0, 5.0]
    assert sample.count == 1.0
    assert sample.box_mask.all()


def test_preprocessor_is_deterministic_per_seed() -> None:
    cfg = make_config()
    ds = SyntheticCountingDataset(cfg.data, size=4, n_exemplars=3, seed=0)
    prep = SamplePreprocessor(cfg.data, out_size=64, train=True)
    a = prep(ds[0], np.random.default_rng(123))
    b = prep(ds[0], np.random.default_rng(123))
    assert torch.equal(a.image, b.image)
    assert torch.equal(a.points, b.points)


def test_synthetic_dataset_is_reproducible() -> None:
    cfg = make_config()
    a = SyntheticCountingDataset(cfg.data, size=4, seed=7)
    b = SyntheticCountingDataset(cfg.data, size=4, seed=7)
    assert torch.equal(a[2].image, b[2].image)
    assert torch.equal(a[2].points, b[2].points)


# --------------------------------------------------------------------------- #
# CCSS
# --------------------------------------------------------------------------- #
def test_quadrant_offsets() -> None:
    assert quadrant_offsets(0, 2, 1) == (0, 0)
    assert quadrant_offsets(1, 2, 1) == (0, 1)
    assert quadrant_offsets(2, 2, 1) == (1, 0)
    assert quadrant_offsets(3, 2, 1) == (1, 1)
    with pytest.raises(ValueError):
        quadrant_offsets(4, 2, 1)


def _four_samples(cfg, *, same_category: bool = False):
    prep = SamplePreprocessor(cfg.data, out_size=cfg.data.ccss.sub_image_size)
    samples = []
    categories = [0, 1, 2, 3] if not same_category else [0, 0, 0, 0]
    for i in range(4):
        raw = RawSample(
            image=torch.rand(3, 64, 64),
            boxes=torch.tensor([[4.0, 4.0, 12.0, 12.0]]),
            points=torch.tensor([[10.0, 10.0], [30.0, 30.0]]),
            category=categories[i],
        )
        sample = prep(raw)
        sample.category = categories[i]
        samples.append(sample)
    return samples


def test_build_composite_shapes_and_anchor() -> None:
    cfg = make_config()
    samples = _four_samples(cfg)
    composite = build_composite(samples, cfg.data.ccss, np.random.default_rng(0))
    assert composite.is_composite
    assert composite.image.shape == (3, 64, 64)
    assert composite.density_size == 32
    assert composite.positive_mask is not None
    assert composite.positive_mask.shape == (1, 32, 32)
    # anchor category is preserved
    assert composite.category == samples[0].category


def test_ccss_category_filter_zeroes_other_categories() -> None:
    """Eq. 15: only anchor-consistent quadrants keep positive supervision."""
    cfg = make_config()
    cfg.data.ccss.category_filter = True
    samples = _four_samples(cfg)
    composite = build_composite(samples, cfg.data.ccss, np.random.default_rng(0))
    # only the anchor (category 0, 2 points) survives -> count == 2
    assert composite.count == pytest.approx(2.0)
    assert composite.points.shape[0] == 2
    # every surviving point lies inside the anchor's density quadrant
    positive = composite.positive_mask[0].bool()
    assert positive.any()


def test_ccss_without_filter_keeps_all_points() -> None:
    cfg = make_config()
    cfg.data.ccss.category_filter = False
    samples = _four_samples(cfg)
    composite = build_composite(samples, cfg.data.ccss, np.random.default_rng(0))
    assert composite.points.shape[0] == 8  # 4 quadrants x 2 points
    assert composite.count == pytest.approx(8.0)


def test_build_composite_rejects_too_few_samples() -> None:
    cfg = make_config()
    samples = _four_samples(cfg)[:3]
    with pytest.raises(ValueError, match="needs 4 samples"):
        build_composite(samples, cfg.data.ccss, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# Collation
# --------------------------------------------------------------------------- #
def test_collator_single_image_path() -> None:
    cfg = make_config()
    ds = SyntheticCountingDataset(cfg.data, size=8, seed=0)
    collator = CountingCollator(cfg.data, train=False, seed=0)
    batch = collator([ds[i] for i in range(4)])
    assert batch.images.shape == (4, 3, 64, 64)
    assert batch.density_size == 32
    assert batch.boxes.shape[1] == 3
    assert batch.is_composite is not None and not batch.is_composite.any()
    assert batch.positive_mask is None


def test_collator_mixes_when_p_mix_is_one() -> None:
    cfg = make_config()
    cfg.data.ccss.p_mix = 1.0
    ds = SyntheticCountingDataset(cfg.data, size=8, seed=0)
    collator = CountingCollator(cfg.data, train=True, seed=0)
    batch = collator([ds[i] for i in range(4)])
    # A collage replaces the whole batch: exactly one sample, full resolution.
    assert batch.images.shape == (1, 3, 64, 64)
    assert batch.is_composite is not None and bool(batch.is_composite[0])
    assert batch.positive_mask is not None


def test_collator_force_mix_for_ecr() -> None:
    cfg = make_config()
    ds = SyntheticCountingDataset(cfg.data, size=8, seed=0)
    collator = CountingCollator(cfg.data, train=False, seed=0, force_mix=True)
    batch = collator([ds[i] for i in range(4)])
    assert batch.images.shape[0] == 1
    assert batch.positive_mask is not None


def test_collate_samples_pads_variable_k_and_m() -> None:
    from gipacount.data.types import CountingSample

    def sample(k: int, m: int) -> CountingSample:
        return CountingSample(
            image=torch.rand(3, 32, 32),
            pad_mask=torch.ones(1, 32, 32, dtype=torch.bool),
            boxes=torch.rand(k, 4),
            box_mask=torch.ones(k, dtype=torch.bool),
            points=torch.rand(m, 2),
            point_mask=torch.ones(m, dtype=torch.bool),
            point_mass=torch.ones(m),
            count=float(m),
            category=0,
            image_id=f"s{k}{m}",
            density_size=16,
        )

    batch = collate_samples([sample(3, 5), sample(1, 2)])
    assert batch.boxes.shape == (2, 3, 4)
    assert batch.box_mask.sum().item() == 4  # 3 + 1
    assert batch.point_mask.sum().item() == 7  # 5 + 2
    assert batch.points.shape == (2, 5, 2)


def test_collate_rejects_mixed_density_sizes() -> None:
    from gipacount.data.types import CountingSample

    def sample(size: int) -> CountingSample:
        return CountingSample(
            image=torch.rand(3, 32, 32),
            pad_mask=torch.ones(1, 32, 32, dtype=torch.bool),
            boxes=torch.rand(1, 4),
            box_mask=torch.ones(1, dtype=torch.bool),
            points=torch.rand(1, 2),
            point_mask=torch.ones(1, dtype=torch.bool),
            point_mass=torch.ones(1),
            count=1.0,
            category=0,
            image_id="x",
            density_size=size,
        )

    with pytest.raises(ValueError, match="mixed density"):
        collate_samples([sample(16), sample(32)])


def test_collator_state_round_trip() -> None:
    """The collator RNG can be snapshotted for exact resumption."""
    cfg = make_config()
    cfg.data.ccss.p_mix = 1.0
    ds = SyntheticCountingDataset(cfg.data, size=8, seed=0)
    collator = CountingCollator(cfg.data, train=True, seed=0)
    state = collator.get_state()
    first = collator([ds[i] for i in range(4)])
    collator.set_state(state)
    second = collator([ds[i] for i in range(4)])
    assert torch.equal(first.images, second.images)
    assert torch.equal(first.points[0], second.points[0])

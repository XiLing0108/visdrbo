"""Generate a stratified synthetic scatter plot dataset.

The generated JSON is compatible with ``plot_umap.py``: every record contains
``points``, ``labels`` and ``image_count``.  A ``metadata`` field records the
actual factors used to create the sample.  The default run creates 2,000 plots
with balanced coverage of the five visual attributes described in the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


SHAPE_TYPES = (
    "circular",
    "elliptical",
    "elongated",
    "crescent",
    "ring",
    "spiral",
    "multimodal",
)
POSITION_PRESETS = {
    "center": (0.50, 0.50),
    "top_left": (0.25, 0.75),
    "top_right": (0.75, 0.75),
    "bottom_left": (0.25, 0.25),
    "bottom_right": (0.75, 0.25),
}
SCENARIOS = ("clean", "noise", "outliers", "non_cluster", "mixed")
SPREAD_RANGE = (0.25, 4.00)
CLARITY_RANGE = (0.35, 1.60)
DEFAULT_NUM_SAMPLES = 2_000


def _balanced_categories(values: Sequence, count: int,
                         rng: np.random.Generator) -> list:
    """Return a shuffled, near-equal allocation of categorical values."""
    repeated = np.resize(np.asarray(values, dtype=object), count)
    return repeated[rng.permutation(count)].tolist()


def _stratified_continuous(low: float, high: float, count: int,
                           rng: np.random.Generator) -> np.ndarray:
    """One-dimensional Latin-hypercube sample over a closed range."""
    strata = (rng.permutation(count) + rng.random(count)) / count
    return low + (high - low) * strata


def stratified_attributes(count: int, rng: np.random.Generator) -> list[dict]:
    """Sample balanced plot-level attributes across the paper's five factors.

    Categorical attributes are allocated as evenly as possible. Spread and
    separation are sampled once in every equal-width stratum. Random rotations
    and per-cluster point counts continue to add within-stratum variation.
    """
    if count < 1:
        raise ValueError("count must be positive")
    cluster_counts = _balanced_categories(range(1, 11), count, rng)
    cluster_shapes = _balanced_categories(
        SHAPE_TYPES, sum(cluster_counts), rng
    )
    positions = _balanced_categories(tuple(POSITION_PRESETS), count, rng)
    scenarios = _balanced_categories(SCENARIOS, count, rng)
    spreads = _stratified_continuous(*SPREAD_RANGE, count, rng)
    separations = _stratified_continuous(*CLARITY_RANGE, count, rng)
    result = []
    shape_offset = 0
    for i in range(count):
        k = int(cluster_counts[i])
        shapes = cluster_shapes[shape_offset:shape_offset + k]
        shape_offset += k
        result.append({
            "cluster_count": k,
            "spread": float(spreads[i]),
            "separation": float(separations[i]),
            "cluster_shapes": [str(shape) for shape in shapes],
            "position_name": str(positions[i]),
            "position": POSITION_PRESETS[positions[i]],
            "scenario": str(scenarios[i]),
        })
    return result


def _rotation_matrix(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def gaussian_cluster(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Circular Gaussian cluster; ``scale`` controls its tightness."""
    return rng.standard_normal((n, 2)) * scale


def shaped_cluster(
    n: int, shape: str, scale: float, rng: np.random.Generator
) -> np.ndarray:
    """Create a zero-centred cluster from one of the seven shape prototypes."""
    points, _ = shaped_cluster_with_attributes(n, shape, scale, rng)
    return points


def shaped_cluster_with_attributes(
    n: int, shape: str, scale: float, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Generate a cluster and its continuous generative shape attributes.

    The attributes describe the parameters used to generate the shape.  They
    are not post-hoc measurements, so they remain stable and interpretable.
    """
    rotation = float(rng.uniform(0, 2 * np.pi))
    attributes = {
        "aspect_ratio": 1.0,
        "curvature": 0.0,
        "convexity": 1.0,
        "number_of_modes": 1,
        "hole_ratio": 0.0,
        "rotation_radians": rotation,
    }

    if shape == "circular":
        points = gaussian_cluster(n, scale, rng)
    elif shape == "elliptical":
        axis_ratio = float(rng.uniform(1.5, 3.5))
        points = rng.standard_normal((n, 2)) * np.array([scale * axis_ratio, scale])
        attributes["aspect_ratio"] = axis_ratio
    elif shape == "elongated":
        axis_ratio = float(rng.uniform(5.0, 10.0))
        bend = float(rng.uniform(0.0, 0.12))
        along = rng.normal(0, axis_ratio * scale, n)
        across = rng.normal(0, scale, n) + bend * along ** 2 / (axis_ratio * scale)
        points = np.column_stack((along, across))
        attributes.update({
            "aspect_ratio": axis_ratio,
            "curvature": bend,
            "convexity": float(1.0 - 0.5 * bend),
        })
    elif shape == "crescent":
        angular_span = float(rng.uniform(1.15 * np.pi, 1.85 * np.pi))
        mean_radius = float(rng.uniform(2.2, 3.2) * scale)
        thickness = float(rng.uniform(0.12, 0.30) * mean_radius)
        theta = rng.uniform(-angular_span / 2, angular_span / 2, n)
        radius = mean_radius + rng.normal(0, thickness, n)
        points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
        attributes.update({
            "aspect_ratio": float(angular_span / np.pi),
            "curvature": float(1.0 / mean_radius),
            "convexity": float(np.clip(1.0 - angular_span / (2 * np.pi), 0, 1)),
            "angular_span_radians": angular_span,
            "thickness": thickness,
        })
    elif shape == "ring":
        mean_radius = float(rng.uniform(2.0, 3.0) * scale)
        ring_width = float(rng.uniform(0.10, 0.28) * mean_radius)
        completeness = float(rng.uniform(0.82, 1.0))
        angular_span = 2 * np.pi * completeness
        theta = rng.uniform(0, 2 * np.pi, n)
        if completeness < 0.999:
            theta = rng.uniform(-angular_span / 2, angular_span / 2, n)
        radius = np.maximum(0, rng.normal(mean_radius, ring_width, n))
        points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
        inner_radius = max(0.0, mean_radius - 2 * ring_width)
        outer_radius = mean_radius + 2 * ring_width
        attributes.update({
            "convexity": float(0.35 + 0.25 * (1 - completeness)),
            "hole_ratio": float(inner_radius / outer_radius),
            "completeness": completeness,
            "ring_width": ring_width,
        })
    elif shape == "spiral":
        turns = float(rng.uniform(1.25, 3.0))
        theta = rng.uniform(0, turns * 2 * np.pi, n)
        spacing = float(rng.uniform(0.22, 0.45) * scale)
        radius = spacing * theta + rng.normal(0, 0.16 * scale, n)
        points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
        attributes.update({
            "aspect_ratio": float(rng.uniform(1.0, 1.25)),
            "curvature": float(turns),
            "convexity": float(1.0 / (1.0 + turns)),
            "number_of_turns": turns,
            "arm_spacing": spacing,
        })
        points[:, 1] *= attributes["aspect_ratio"]
    elif shape == "multimodal":
        modes = int(rng.integers(2, 6))
        mode_distance = float(rng.uniform(1.5, 3.0) * scale)
        angles = rng.uniform(0, 2 * np.pi, modes)
        distances = rng.uniform(0.6, 1.0, modes) * mode_distance
        mode_centres = np.column_stack((np.cos(angles), np.sin(angles))) * distances[:, None]
        weights = rng.dirichlet(np.ones(modes))
        assignments = rng.choice(modes, size=n, p=weights)
        mode_scales = rng.uniform(0.30, 0.70, modes) * scale
        points = mode_centres[assignments] + rng.normal(0, mode_scales[assignments, None], (n, 2))
        attributes.update({
            "convexity": float(rng.uniform(0.45, 0.75)),
            "number_of_modes": modes,
            "mode_distance": mode_distance,
            "mode_weights": weights.tolist(),
        })
    else:
        raise ValueError(f"Unknown shape {shape!r}; choose from {SHAPE_TYPES}")

    points = points @ _rotation_matrix(rotation).T
    # Centre by the realised mean so centre placement remains exact.
    return points - points.mean(axis=0, keepdims=True), attributes


def cluster_radius(points: np.ndarray, quantile: float = 0.95) -> float:
    """Robust cluster radius used when enforcing inter-cluster clarity."""
    return float(np.quantile(np.linalg.norm(points, axis=1), quantile))


def place_cluster_centres(
    radii: Sequence[float], clarity_factor: float, rng: np.random.Generator,
    max_attempts: int = 10_000,
) -> np.ndarray:
    """Place centres so d(i,j) >= clarity_factor * (radius_i + radius_j)."""
    if clarity_factor < 0:
        raise ValueError("clarity_factor must be non-negative")
    centres = [np.zeros(2)]
    for i in range(1, len(radii)):
        for _ in range(max_attempts):
            anchor = int(rng.integers(0, i))
            angle = rng.uniform(0, 2 * np.pi)
            required = clarity_factor * (radii[i] + radii[anchor])
            # Stay near the requested boundary.  In particular, factors below
            # one should create real overlap instead of merely permitting it.
            distance = required * rng.uniform(1.0, 1.18)
            candidate = centres[anchor] + distance * np.array(
                [np.cos(angle), np.sin(angle)]
            )
            valid = all(
                np.linalg.norm(candidate - centres[j])
                >= clarity_factor * (radii[i] + radii[j])
                for j in range(i)
            )
            if valid:
                centres.append(candidate)
                break
        else:
            raise RuntimeError("Could not place cluster centres; reduce K or clarity")
    return np.asarray(centres)


def translate_to(points: np.ndarray, target: Sequence[float]) -> np.ndarray:
    """Translate all points so their bounding-box centre equals ``target``."""
    target_array = np.asarray(target, dtype=float)
    if target_array.shape != (2,):
        raise ValueError("target position must contain exactly two numbers")
    current = (points.min(axis=0) + points.max(axis=0)) / 2
    return points + target_array - current


def generate_scatter(
    rng: np.random.Generator,
    k: int | None = None,
    points_per_cluster: int | tuple[int, int] = (80, 180),
    tightness: float | tuple[float, float] = (0.25, 4.00),
    clarity_factor: float | tuple[float, float] = (0.35, 1.60),
    shapes: Sequence[str] | None = None,
    position: Sequence[float] = (0.5, 0.5),
    scenario: str = "clean",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate one scatter plot and return points, labels and factor metadata."""
    k = int(rng.integers(1, 11)) if k is None else int(k)
    if not 1 <= k <= 10:
        raise ValueError("k must be between 1 and 10")
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")

    if shapes is not None and len(shapes) not in (1, k):
        raise ValueError("shapes must have length 1 or k")
    selected_shapes = (
        [str(rng.choice(SHAPE_TYPES)) for _ in range(k)]
        if shapes is None else list(shapes) * k if len(shapes) == 1 else list(shapes)
    )

    def sample_value(value, integer=False):
        if isinstance(value, tuple):
            low, high = value
            return int(rng.integers(low, high + 1)) if integer else float(rng.uniform(low, high))
        return int(value) if integer else float(value)

    def sample_spread(value):
        """Sample spread with moderate bias toward loose (low-tightness) clusters."""
        if isinstance(value, tuple):
            low, high = map(float, value)
            if low <= 0 or high < low:
                raise ValueError("tightness/spread range must satisfy 0 < MIN <= MAX")
            # sqrt(U) has more mass near 1, producing more large-spread clusters.
            return low + (high - low) * np.sqrt(rng.random())
        if float(value) <= 0:
            raise ValueError("tightness/spread must be positive")
        return float(value)

    def sample_clarity(value):
        """Sample one image-level clarity factor; values below 1 allow overlap."""
        if isinstance(value, tuple):
            low, high = map(float, value)
            if low < 0 or high < low:
                raise ValueError("clarity range must satisfy 0 <= MIN <= MAX")
            return float(rng.uniform(low, high))
        if float(value) < 0:
            raise ValueError("clarity must be non-negative")
        return float(value)

    counts = [sample_value(points_per_cluster, integer=True) for _ in range(k)]
    scales = [sample_spread(tightness) for _ in range(k)]
    sampled_clarity = sample_clarity(clarity_factor)
    # An intuitive score in (0, 1]: higher means tighter.  Spread is the actual
    # standard-deviation-like parameter used by the shape generators.
    tightness_scores = [1.0 / (1.0 + scale) for scale in scales]
    generated_clusters = [
        shaped_cluster_with_attributes(n, shape, scale, rng)
        for n, shape, scale in zip(counts, selected_shapes, scales)
    ]
    local = [cluster for cluster, _ in generated_clusters]
    shape_attributes = [attributes for _, attributes in generated_clusters]
    raw_radii = [cluster_radius(cluster) for cluster in local]

    # Standardise shape prototypes first, then assign an explicit visual radius.
    # This prevents whole-image normalisation from cancelling the spread factor.
    standardised = [
        cluster / max(radius, 1e-12)
        for cluster, radius in zip(local, raw_radii)
    ]
    spread_levels = [
        float(np.clip((scale - 0.25) / (4.00 - 0.25), 0, 1))
        for scale in scales
    ]
    target_radii = [0.04 + 0.24 * level for level in spread_levels]
    centres = place_cluster_centres(target_radii, sampled_clarity, rng)

    cluster_points = [
        cluster * radius + centre
        for cluster, radius, centre in zip(standardised, target_radii, centres)
    ]
    points = np.vstack(cluster_points)
    labels = np.concatenate([np.full(n, i, dtype=int) for i, n in enumerate(counts)])

    # Preserve the requested radii unless the complete multi-cluster layout
    # would leave the canvas. In that case scale it uniformly while keeping
    # every point within [0, 1] at the requested global position.
    span = np.ptp(points, axis=0)
    target = np.asarray(position, dtype=float)
    available = 2 * np.minimum(target, 1 - target)
    per_axis_scale = np.divide(
        available, span, out=np.full(2, np.inf), where=span > 0
    )
    layout_scale = min(1.0, 0.95 / max(float(span.max()), 1e-12),
                       float(np.min(per_axis_scale)))
    points *= layout_scale
    points = translate_to(points, position)

    # Add unlabelled background points after placing the real clusters, so
    # cluster radii and clarity keep their original meaning. Label -1 is
    # reserved for these points and is rendered in neutral gray.
    base_count = len(points)
    extras = []
    extra_types = []
    if scenario in ("noise", "mixed"):
        n = max(1, round(base_count * 0.10))
        extras.append(rng.uniform(0, 1, (n, 2)))
        extra_types.extend(["noise"] * n)
    if scenario in ("outliers", "mixed"):
        n = max(1, round(base_count * 0.03))
        # Place isolated points near the canvas edges, away from the cluster
        # core whenever possible.
        candidates = rng.uniform(0, 1, (max(n * 30, 100), 2))
        distance = np.min(np.linalg.norm(candidates[:, None] - points[None], axis=2), axis=1)
        extras.append(candidates[np.argsort(distance)[-n:]])
        extra_types.extend(["outlier"] * n)
    structure = None
    if scenario in ("non_cluster", "mixed"):
        n = max(2, round(base_count * 0.25))
        structure = str(rng.choice(("line", "wave", "uniform")))
        t = rng.uniform(0.08, 0.92, n)
        if structure == "line":
            extra = np.column_stack((t, 0.5 + rng.normal(0, 0.012, n)))
        elif structure == "wave":
            extra = np.column_stack((t, 0.5 + 0.22 * np.sin(4 * np.pi * t) + rng.normal(0, 0.012, n)))
        else:
            extra = rng.uniform(0.05, 0.95, (n, 2))
        extras.append(np.clip(extra, 0, 1))
        extra_types.extend(["non_cluster"] * n)
    if extras:
        points = np.vstack((points, *extras))
        labels = np.concatenate((labels, np.full(len(extra_types), -1, dtype=int)))

    metadata = {
        "cluster_count": k,
        "shapes": selected_shapes,
        "shape_attributes": shape_attributes,
        "spread_scales": scales,
        "tightness_scores": tightness_scores,
        "target_cluster_radii": target_radii,
        "effective_cluster_radii": [radius * layout_scale for radius in target_radii],
        "layout_scale": layout_scale,
        "clarity_factor": sampled_clarity,
        "clarity_range": (
            list(map(float, clarity_factor))
            if isinstance(clarity_factor, tuple)
            else [float(clarity_factor), float(clarity_factor)]
        ),
        "position": list(map(float, position)),
        "points_per_cluster": counts,
        "cluster_radii_before_normalization": raw_radii,
        "scenario": scenario,
        "background_point_types": extra_types,
        "noise_count": extra_types.count("noise"),
        "outlier_count": extra_types.count("outlier"),
        "non_cluster_count": extra_types.count("non_cluster"),
        "non_cluster_structure": structure,
    }
    return points, labels, metadata


def save_plot(points: np.ndarray, labels: np.ndarray, path: Path) -> None:
    # Keep matplotlib optional when callers only need JSON data.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    clustered = labels >= 0
    ax.scatter(points[clustered, 0], points[clustered, 1], c=labels[clustered],
               cmap="tab10", vmin=0, vmax=9, s=10, alpha=1.0)
    if np.any(~clustered):
        ax.scatter(points[~clustered, 0], points[~clustered, 1],
                   color="gray", s=10, alpha=1.0)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=Path("synthetic_scatter"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--points", type=int, nargs=2, metavar=("MIN", "MAX"), default=(80, 180))
    parser.add_argument("--no-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("num-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []
    attributes = stratified_attributes(args.num_samples, rng)
    for image_count in range(args.num_samples):
        factors = attributes[image_count]
        points, labels, metadata = generate_scatter(
            rng=rng,
            k=factors["cluster_count"],
            points_per_cluster=tuple(args.points),
            tightness=factors["spread"],
            clarity_factor=factors["separation"],
            shapes=factors["cluster_shapes"],
            position=factors["position"],
            scenario=factors["scenario"],
        )
        metadata.update({
            "stratified_attributes": factors,
            "random_seed": args.seed,
        })
        records.append({
            "image_count": image_count,
            "points": points.tolist(),
            "labels": labels.tolist(),
            "metadata": metadata,
        })
        if not args.no_images:
            save_plot(points, labels, args.output_dir / f"scatter_{image_count:05d}.png")

    output_json = args.output_dir / "synthetic_scatter.json"
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print(f"Generated {len(records)} samples in {args.output_dir}")


if __name__ == "__main__":
    main()

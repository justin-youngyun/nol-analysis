#!/usr/bin/env python
"""
NOL (Novel Object Location) analysis from DeepLabCut pose tracking.

Reads DeepLabCut tracking output (.csv or .h5) and computes, per animal:
  - the discrimination index (DI) for the novel vs familiar object location
  - a set of alternative learning metrics (time-binned DI, cumulative DI,
    bout-count DI, occupancy DI, first-approach and approach latency)
  - exploration and locomotion metrics (thigmotaxis, distance, velocity,
    immobility, centre entries, path tortuosity)
  - quality-control flags (low explorer, corner detection, bounds check)

It then writes a per-animal table, a cohort summary that tests whether the
group discriminates the novel location above chance, and a summary figure.

Method notes:
  * Distortion correction: projective homography from the four arena corners
    A, B, C, D. A=(0,0) top-left, B=(w,0) top-right, C=(w,h) bottom-right,
    D=(0,h) bottom-left.
  * Body centroid: NaN-safe mean of {Neck, Back1, Back2, Back3}.
  * Interaction with an object: nose within `interact_radius` of the object
    edge AND head angle (neck to nose vs nose to object) <= `interact_angle_deg`.
  * DI = (t_novel - t_familiar) / (t_novel + t_familiar), object B is novel.
  * Zones (per frame):
      - near_object: centroid within `object_exclusion_radius_cm` of either
        object centre (excludes object vicinity from the wall/centre split)
      - wall: within `wall_buffer_cm` of any arena wall AND not near_object
      - centre: neither
  * Thigmotaxis = wall / (wall + centre), so object-time is out of the denom.
  * Locomotion: frame-to-frame displacement, summed, NaN frames interpolated.
  * Immobility: velocity < `immobile_speed_cm_s`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import median_abs_deviation

try:
    from statsmodels.stats.diagnostic import lilliefors
    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------


@dataclass
class Config:
    input_dir: Path
    output_dir: Path

    maze_w: float = 52.07
    maze_h: float = 68.58
    pixel_to_meter: float = 0.0007895
    frame_rate: float = 60.0

    # Edge-point / interaction
    bottle_radius: float = 2.5          # cm, used for edge-point vector
    interact_radius: float = 2.5        # cm from object edge
    interact_angle_deg: float = 60.0

    # Exclusion criteria
    min_total_interact_sec: float = 2.0
    min_bouts_per_obj: int = 3
    min_bout_sec: float = 0.3

    # Thigmotaxis / locomotion
    wall_buffer_cm: float = 8.0
    object_exclusion_radius_cm: float = 5.0  # from object centre
    immobile_speed_cm_s: float = 2.0
    low_explorer_distance_m: float = 3.0
    low_explorer_immobile_pct: float = 80.0

    # Time window + stats
    time_window_sec: tuple[float, float] | None = (120.0, 600.0)
    k_rz: float = 2.5
    n_bootstrap: int = 5000
    bootstrap_ci: float = 0.95
    alpha: float = 0.05
    eps_t: float = 1e-6

    # DLC
    likelihood_cutoff: float = 0.4

    # Output behavior (for programmatic callers)
    save_outputs: bool = True
    verbose: bool = True

    @property
    def pixel_to_cm(self) -> float:
        return self.pixel_to_meter * 100.0

    @property
    def min_bout_frames(self) -> int:
        return max(1, round(self.min_bout_sec * self.frame_rate))

    @property
    def world_corners(self) -> np.ndarray:
        # A=(0,0) top-left, B=(w,0) top-right, C=(w,h) bottom-right, D=(0,h)
        return np.array(
            [[0.0, 0.0], [self.maze_w, 0.0], [self.maze_w, self.maze_h], [0.0, self.maze_h]],
            dtype=float,
        )


# ----------------------------------------------------------------------------
# DeepLabCut loading (.csv or .h5)
# ----------------------------------------------------------------------------


REQUIRED_BODYPARTS = [
    "Nose",
    "Neck",
    "Back1",
    "Back2",
    "Back3",
    "Tailbase",
    "ObjA_S",
    "ObjB_Novel",
    "A",
    "B",
    "C",
    "D",
]


def _read_dlc_frame(path: Path) -> pd.DataFrame:
    """Read a DeepLabCut tracking file into a DataFrame with MultiIndex columns.

    Supports the two native DeepLabCut export formats:
      - .h5  (read with pandas / pytables)
      - .csv (three header rows: scorer, bodyparts, coords)
    """
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        df = pd.read_hdf(path)
    elif suffix == ".csv":
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    else:
        raise ValueError(f"{path.name}: unsupported extension {suffix}")
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(f"{path.name}: expected MultiIndex columns")
    return df


def load_dlc(path: Path, likelihood_cutoff: float) -> dict[str, np.ndarray]:
    """Load a DeepLabCut file into a dict of <bp>_x, <bp>_y, <bp>_lik, <bp>_mid.

    Low-likelihood frames are NaN-masked in x/y.
    """
    df = _read_dlc_frame(path)
    if df.columns.nlevels == 3:
        df.columns = df.columns.droplevel(0)

    out: dict[str, np.ndarray] = {}
    for bp in REQUIRED_BODYPARTS:
        if (bp, "x") not in df.columns:
            out[f"{bp}_x"] = np.array([])
            out[f"{bp}_y"] = np.array([])
            out[f"{bp}_lik"] = np.array([])
            out[f"{bp}_mid"] = np.array([np.nan, np.nan])
            continue
        x = df[(bp, "x")].to_numpy(dtype=float)
        y = df[(bp, "y")].to_numpy(dtype=float)
        lik = (
            df[(bp, "likelihood")].to_numpy(dtype=float)
            if (bp, "likelihood") in df.columns
            else np.ones_like(x)
        )
        mask = lik < likelihood_cutoff
        x_m = x.copy()
        y_m = y.copy()
        x_m[mask] = np.nan
        y_m[mask] = np.nan
        out[f"{bp}_x"] = x_m
        out[f"{bp}_y"] = y_m
        out[f"{bp}_lik"] = lik
        out[f"{bp}_mid"] = np.array([np.nanmedian(x_m), np.nanmedian(y_m)])

    return out


# ----------------------------------------------------------------------------
# Geometry: perspective correction
# ----------------------------------------------------------------------------


def fit_projective(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if HAVE_CV2:
        return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    return _dlt_homography(src, dst)


def _dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    assert src.shape == (4, 2) and dst.shape == (4, 2)
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A, dtype=float)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(pts)
    ones = np.ones((pts.shape[0], 1))
    homog = np.hstack([pts, ones]) @ H.T
    return homog[:, :2] / homog[:, 2:3]


# ----------------------------------------------------------------------------
# Per-animal data container
# ----------------------------------------------------------------------------


@dataclass
class AnimalResult:
    animal_id: str
    file_name: str
    folder: str
    n_frames: int
    # NOL fields
    dist_a: np.ndarray = field(default_factory=lambda: np.array([]))
    dist_b: np.ndarray = field(default_factory=lambda: np.array([]))
    angles_a: np.ndarray = field(default_factory=lambda: np.array([]))
    angles_b: np.ndarray = field(default_factory=lambda: np.array([]))
    nose_cm: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    obj_a_cm: np.ndarray = field(default_factory=lambda: np.array([np.nan, np.nan]))
    obj_b_cm: np.ndarray = field(default_factory=lambda: np.array([np.nan, np.nan]))
    corners_ok: bool = False
    coords_in_bounds: bool = True
    # Exploration/locomotion fields (populated by compute_exploration_metrics)
    time_wall_pct: float = float("nan")
    time_center_pct: float = float("nan")
    time_near_object_pct: float = float("nan")
    distance_m: float = float("nan")
    mean_velocity_cm_s: float = float("nan")
    immobile_pct: float = float("nan")
    center_entries: int = 0
    path_tortuosity: float = float("nan")
    low_explorer_flag: bool = False


# ----------------------------------------------------------------------------
# Core computation helpers
# ----------------------------------------------------------------------------


def compute_frame_window(n: int, fr: float, tw: tuple[float, float] | None) -> np.ndarray:
    if tw is None:
        return np.ones(n, dtype=bool)
    t = np.arange(n) / fr
    return (t >= tw[0]) & (t <= tw[1])


def calculate_angle(neck_to_nose: np.ndarray, nose_to_obj: np.ndarray) -> np.ndarray:
    dot = np.sum(neck_to_nose * nose_to_obj, axis=1)
    norm_nn = np.linalg.norm(neck_to_nose, axis=1)
    norm_no = np.linalg.norm(nose_to_obj, axis=1)
    denom = np.maximum(norm_nn * norm_no, np.finfo(float).eps)
    cos_theta = np.clip(dot / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def normalize_rows(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, np.finfo(float).eps)


def nan_safe_mean_of_points(points: list[np.ndarray]) -> np.ndarray:
    """Per-frame mean of a list of (N,2) arrays, ignoring NaN coordinates.

    If all points are NaN for a frame, the result is (NaN, NaN) for that frame.
    """
    stacked = np.stack(points, axis=0)  # (K, N, 2)
    return np.nanmean(stacked, axis=0)


def compute_exploration_metrics(
    centroid_cm: np.ndarray,
    obj_a_cm: np.ndarray,
    obj_b_cm: np.ndarray,
    cfg: Config,
) -> dict[str, Any]:
    """Thigmotaxis + locomotion + QC metrics.

    Args:
        centroid_cm: (N, 2) array of body centroid positions in cm.
        obj_a_cm, obj_b_cm: (2,) object centre positions in cm.
        cfg: Config.

    Returns:
        Dict with keys: time_wall_pct, time_center_pct, time_near_object_pct,
            distance_m, mean_velocity_cm_s, immobile_pct, center_entries,
            path_tortuosity, low_explorer_flag.
    """
    nan = float("nan")
    out = {
        "time_wall_pct": nan,
        "time_center_pct": nan,
        "time_near_object_pct": nan,
        "distance_m": nan,
        "mean_velocity_cm_s": nan,
        "immobile_pct": nan,
        "center_entries": 0,
        "path_tortuosity": nan,
        "low_explorer_flag": False,
    }
    if centroid_cm.size == 0:
        out["low_explorer_flag"] = True
        return out

    x = centroid_cm[:, 0]
    y = centroid_cm[:, 1]
    valid = ~np.isnan(x) & ~np.isnan(y)

    # Zone classification
    # Near-object: within exclusion radius of either object centre
    if not np.any(np.isnan(obj_a_cm)):
        d_a = np.hypot(x - obj_a_cm[0], y - obj_a_cm[1])
    else:
        d_a = np.full_like(x, np.inf)
    if not np.any(np.isnan(obj_b_cm)):
        d_b = np.hypot(x - obj_b_cm[0], y - obj_b_cm[1])
    else:
        d_b = np.full_like(x, np.inf)
    near_object = (d_a <= cfg.object_exclusion_radius_cm) | (d_b <= cfg.object_exclusion_radius_cm)

    # Wall: within buffer of any wall
    wb = cfg.wall_buffer_cm
    wall_any = (x <= wb) | (x >= cfg.maze_w - wb) | (y <= wb) | (y >= cfg.maze_h - wb)

    # Final mutually-exclusive classification
    in_near_obj = near_object & valid
    in_wall = wall_any & ~near_object & valid
    in_center = ~wall_any & ~near_object & valid

    n_valid = int(valid.sum())
    n_near = int(in_near_obj.sum())
    n_wall = int(in_wall.sum())
    n_center = int(in_center.sum())
    n_non_obj = n_wall + n_center

    if n_valid > 0:
        out["time_near_object_pct"] = 100.0 * n_near / n_valid
    if n_non_obj > 0:
        out["time_wall_pct"] = 100.0 * n_wall / n_non_obj
        out["time_center_pct"] = 100.0 * n_center / n_non_obj

    # Locomotion: frame-to-frame displacement with gap handling
    # Use linearly-interpolated trajectory for distance calc (avoids NaN gaps)
    x_interp = pd.Series(x).interpolate(limit=int(cfg.frame_rate)).to_numpy()
    y_interp = pd.Series(y).interpolate(limit=int(cfg.frame_rate)).to_numpy()
    dx = np.diff(x_interp)
    dy = np.diff(y_interp)
    step = np.hypot(dx, dy)
    step = step[~np.isnan(step)]
    # Velocity per frame (cm/s)
    v_cm_s = step * cfg.frame_rate

    if step.size > 0:
        dist_cm = float(np.sum(step))
        out["distance_m"] = dist_cm / 100.0
        # Mean velocity over moving frames only (velocity >= immobile threshold)
        moving = v_cm_s >= cfg.immobile_speed_cm_s
        if moving.any():
            out["mean_velocity_cm_s"] = float(np.mean(v_cm_s[moving]))
        out["immobile_pct"] = 100.0 * float(np.sum(~moving)) / float(moving.size)

        # Path tortuosity = total distance / net displacement (start to end)
        start_ok = valid[:1].any() or ~np.isnan(x_interp[0])
        end_ok = valid[-1:].any() or ~np.isnan(x_interp[-1])
        if start_ok and end_ok:
            net = float(np.hypot(x_interp[-1] - x_interp[0], y_interp[-1] - y_interp[0]))
            if net > 1e-3:
                out["path_tortuosity"] = dist_cm / net

    # Center entries: count bouts of being in_center lasting >= 0.3s
    if in_center.size > 0:
        padded = np.concatenate([[False], in_center.astype(bool), [False]])
        edges = np.diff(padded.astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1
        lengths = ends - starts + 1
        out["center_entries"] = int(np.sum(lengths >= max(1, round(0.3 * cfg.frame_rate))))

    # Low-explorer flag
    flag = False
    if not np.isnan(out["distance_m"]) and out["distance_m"] < cfg.low_explorer_distance_m:
        flag = True
    if not np.isnan(out["immobile_pct"]) and out["immobile_pct"] > cfg.low_explorer_immobile_pct:
        flag = True
    out["low_explorer_flag"] = flag

    return out


def process_animal(path: Path, cfg: Config) -> AnimalResult:
    data = load_dlc(path, cfg.likelihood_cutoff)
    n_total = len(data["Nose_x"])
    result = AnimalResult(
        animal_id=animal_id_from_path(path),
        file_name=path.name,
        folder=str(path.parent),
        n_frames=n_total,
    )
    if n_total == 0:
        return result

    win = compute_frame_window(n_total, cfg.frame_rate, cfg.time_window_sec)

    # Stack NOL body points (windowed)
    nose_px = np.column_stack([data["Nose_x"][win], data["Nose_y"][win]])
    neck_px = np.column_stack([data["Neck_x"][win], data["Neck_y"][win]])

    # Stack body centroid points (Neck + Back1 + Back2 + Back3)
    body_pts_px = [
        np.column_stack([data[f"{bp}_x"][win], data[f"{bp}_y"][win]])
        for bp in ("Neck", "Back1", "Back2", "Back3")
    ]
    body_centroid_px = nan_safe_mean_of_points(body_pts_px)

    corners_px = np.array([data["A_mid"], data["B_mid"], data["C_mid"], data["D_mid"]])
    if np.any(np.isnan(corners_px)):
        if cfg.verbose:
            print(f"[WARN] {path.name}: missing corner midpoints, skipping distortion correction")
        return result

    H = fit_projective(corners_px, cfg.world_corners)
    result.corners_ok = True

    nose_cm = apply_homography(H, nose_px)
    neck_cm = apply_homography(H, neck_px)
    centroid_cm = apply_homography(H, body_centroid_px)
    obj_a_cm = apply_homography(H, data["ObjA_S_mid"].reshape(1, 2))[0]
    obj_b_cm = apply_homography(H, data["ObjB_Novel_mid"].reshape(1, 2))[0]

    result.nose_cm = nose_cm
    result.obj_a_cm = obj_a_cm
    result.obj_b_cm = obj_b_cm

    # Bounds check on nose
    xs, ys = nose_cm[:, 0], nose_cm[:, 1]
    coords_ok = np.all(
        (xs[~np.isnan(xs)] >= -5)
        & (xs[~np.isnan(xs)] <= cfg.maze_w + 5)
        & (ys[~np.isnan(ys)] >= -5)
        & (ys[~np.isnan(ys)] <= cfg.maze_h + 5)
    ) if np.any(~np.isnan(xs)) else True
    result.coords_in_bounds = bool(coords_ok)

    # NOL angle / distance vectors (edge-point formulation)
    def edge_point(nose_pts: np.ndarray, centre: np.ndarray) -> np.ndarray:
        return centre + cfg.bottle_radius * normalize_rows(nose_pts - centre)

    neck_to_nose = nose_cm - neck_cm
    obj_a_edge = edge_point(nose_cm, obj_a_cm)
    obj_b_edge = edge_point(nose_cm, obj_b_cm)
    nose_to_a = obj_a_edge - nose_cm
    nose_to_b = obj_b_edge - nose_cm

    angles_a_full = calculate_angle(neck_to_nose, nose_to_a)
    angles_b_full = calculate_angle(neck_to_nose, nose_to_b)
    dist_a_full = np.linalg.norm(nose_to_a, axis=1)
    dist_b_full = np.linalg.norm(nose_to_b, axis=1)

    # Keep frames where nose/obj vectors are well-defined
    valid_angle = ~np.any(np.isnan(neck_to_nose), axis=1)
    result.angles_a = angles_a_full[valid_angle & ~np.isnan(dist_a_full)]
    result.angles_b = angles_b_full[valid_angle & ~np.isnan(dist_b_full)]
    result.dist_a = dist_a_full[valid_angle & ~np.isnan(dist_a_full)]
    result.dist_b = dist_b_full[valid_angle & ~np.isnan(dist_b_full)]

    # Exploration / locomotion on the body centroid trajectory
    expl = compute_exploration_metrics(centroid_cm, obj_a_cm, obj_b_cm, cfg)
    for k, v in expl.items():
        setattr(result, k, v)

    return result


# ----------------------------------------------------------------------------
# Discrimination Index + bout counting
# ----------------------------------------------------------------------------


def count_bouts(mask: np.ndarray, min_len: int) -> int:
    if mask.size == 0:
        return 0
    m = np.asarray(mask, dtype=bool)
    padded = np.concatenate([[False], m, [False]])
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    lengths = ends - starts + 1
    return int(np.sum(lengths >= max(1, min_len)))


def compute_di(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    ang_a: np.ndarray,
    ang_b: np.ndarray,
    cfg: Config,
) -> tuple[float, float, float, int, int]:
    nan = float("nan")
    if dist_a.size == 0 or dist_b.size == 0 or ang_a.size == 0 or ang_b.size == 0:
        return (nan, nan, nan, 0, 0)
    nA = min(dist_a.size, ang_a.size)
    nB = min(dist_b.size, ang_b.size)
    if nA == 0 or nB == 0:
        return (nan, nan, nan, 0, 0)

    dA = dist_a[:nA]
    aA = ang_a[:nA]
    dB = dist_b[:nB]
    aB = ang_b[:nB]

    is_a = (dA <= cfg.interact_radius) & (aA <= cfg.interact_angle_deg)
    is_b = (dB <= cfg.interact_radius) & (aB <= cfg.interact_angle_deg)

    tA = float(is_a.sum() / cfg.frame_rate)
    tB = float(is_b.sum() / cfg.frame_rate)
    n_bout_a = count_bouts(is_a, cfg.min_bout_frames)
    n_bout_b = count_bouts(is_b, cfg.min_bout_frames)

    if (tA + tB) < cfg.min_total_interact_sec:
        return (nan, nan, nan, n_bout_a, n_bout_b)
    if n_bout_a < cfg.min_bouts_per_obj or n_bout_b < cfg.min_bouts_per_obj:
        return (nan, nan, nan, n_bout_a, n_bout_b)
    if (tA + tB) <= 0:
        return (nan, nan, nan, n_bout_a, n_bout_b)

    return ((tB - tA) / (tA + tB), tA, tB, n_bout_a, n_bout_b)


# ----------------------------------------------------------------------------
# Alternative learning metrics (time-binned DI, first-approach, occupancy, etc.)
# ----------------------------------------------------------------------------

def _interaction_masks(dist_a, dist_b, ang_a, ang_b, cfg):
    nA = min(dist_a.size, ang_a.size)
    nB = min(dist_b.size, ang_b.size)
    is_a = (dist_a[:nA] <= cfg.interact_radius) & (ang_a[:nA] <= cfg.interact_angle_deg)
    is_b = (dist_b[:nB] <= cfg.interact_radius) & (ang_b[:nB] <= cfg.interact_angle_deg)
    return is_a, is_b


def _di_in_window(is_a, is_b, start_frame, end_frame, fr):
    sa = is_a[start_frame:end_frame].sum()
    sb = is_b[start_frame:end_frame].sum()
    tA = sa / fr
    tB = sb / fr
    if (tA + tB) <= 0:
        return float("nan"), tA, tB
    return (tB - tA) / (tA + tB), tA, tB


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral, compatible across NumPy versions."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


def compute_alternative_metrics(dist_a, dist_b, ang_a, ang_b, cfg, occupancy_radius_cm=6.0):
    """Time-binned DI, first approach, latency, cumulative DI, bout-count DI, occupancy DI.

    Note: object B is novel by convention (ObjB_Novel), object A is familiar.
    """
    nan = float("nan")
    out = {
        "DI_first2min": nan, "DI_first5min": nan,
        "DI_min0_1": nan, "DI_min1_2": nan, "DI_min2_3": nan, "DI_min3_4": nan, "DI_min4_5": nan,
        "first_approach": "",       # "novel"|"familiar"|""
        "first_approach_novel": nan,  # 1 if novel was first, 0 if familiar, NaN if neither
        "latency_first_novel_sec": nan,
        "latency_first_familiar_sec": nan,
        "DI_bouts": nan,
        "occupancy_DI": nan,        # (zone_B - zone_A) / (zone_A + zone_B) with no angle gate
        "AUC_cumDI_5min": nan,      # area under cumulative DI(t) curve for first 5 min
    }
    # cumulative DI sampled every 30s for first 5min (t=30..300)
    for t in range(30, 301, 30):
        out[f"cumDI_t{t}s"] = nan

    if dist_a.size == 0 or dist_b.size == 0 or ang_a.size == 0 or ang_b.size == 0:
        return out

    is_a, is_b = _interaction_masks(dist_a, dist_b, ang_a, ang_b, cfg)
    fr = cfg.frame_rate
    n = min(is_a.size, is_b.size)
    if n == 0:
        return out
    is_a = is_a[:n]; is_b = is_b[:n]

    # --- Time-binned DI (1-min bins for first 5 min) ---
    f120 = int(min(120 * fr, n))
    f300 = int(min(300 * fr, n))
    out["DI_first2min"], _, _ = _di_in_window(is_a, is_b, 0, f120, fr)
    out["DI_first5min"], _, _ = _di_in_window(is_a, is_b, 0, f300, fr)
    for i in range(5):
        s = int(min((i * 60) * fr, n))
        e = int(min(((i + 1) * 60) * fr, n))
        di, _, _ = _di_in_window(is_a, is_b, s, e, fr)
        out[f"DI_min{i}_{i+1}"] = di

    # --- Cumulative DI sampled every 30s; AUC over 5 min ---
    cumDI_vals = []
    for t in range(30, 301, 30):
        f = int(min(t * fr, n))
        di, _, _ = _di_in_window(is_a, is_b, 0, f, fr)
        out[f"cumDI_t{t}s"] = di
        cumDI_vals.append((t, di))
    valid = [(t, v) for t, v in cumDI_vals if not np.isnan(v)]
    if len(valid) >= 2:
        ts = np.array([v[0] for v in valid], dtype=float)
        vs = np.array([v[1] for v in valid], dtype=float)
        out["AUC_cumDI_5min"] = _trapz(vs, ts) / (ts[-1] - ts[0])  # mean DI over time

    # --- Bout-count DI ---
    nBa = count_bouts(is_a, cfg.min_bout_frames)
    nBb = count_bouts(is_b, cfg.min_bout_frames)
    if (nBa + nBb) > 0:
        out["DI_bouts"] = (nBb - nBa) / (nBa + nBb)

    # --- First approach (no angle gate; just body-distance to either object) ---
    # Use the wider zone-occupancy radius so we capture approach even before head-orientation
    nA_dist = dist_a.size
    nB_dist = dist_b.size
    nd = min(nA_dist, nB_dist)
    in_a_zone = dist_a[:nd] <= occupancy_radius_cm
    in_b_zone = dist_b[:nd] <= occupancy_radius_cm
    first_a = np.argmax(in_a_zone) if in_a_zone.any() else None
    first_b = np.argmax(in_b_zone) if in_b_zone.any() else None
    if first_a is not None:
        out["latency_first_familiar_sec"] = float(first_a / fr)
    if first_b is not None:
        out["latency_first_novel_sec"] = float(first_b / fr)
    if first_a is not None or first_b is not None:
        if first_a is None:
            out["first_approach"] = "novel"; out["first_approach_novel"] = 1.0
        elif first_b is None:
            out["first_approach"] = "familiar"; out["first_approach_novel"] = 0.0
        else:
            if first_b < first_a:
                out["first_approach"] = "novel"; out["first_approach_novel"] = 1.0
            elif first_a < first_b:
                out["first_approach"] = "familiar"; out["first_approach_novel"] = 0.0
            else:
                out["first_approach"] = "tie"; out["first_approach_novel"] = 0.5

    # --- Occupancy DI: distance-only zone, no angle gate ---
    occ_a_sec = float(in_a_zone.sum() / fr)
    occ_b_sec = float(in_b_zone.sum() / fr)
    if (occ_a_sec + occ_b_sec) > 0:
        out["occupancy_DI"] = (occ_b_sec - occ_a_sec) / (occ_a_sec + occ_b_sec)

    return out


def extract_bout_records(dist_a, dist_b, ang_a, ang_b, cfg, animal_id, file_name):
    """Per-bout records for mixed-effects analysis.

    Returns list of dicts: one row per bout (object x bout). object is 'novel' or 'familiar'.
    """
    if dist_a.size == 0 or dist_b.size == 0:
        return []
    is_a, is_b = _interaction_masks(dist_a, dist_b, ang_a, ang_b, cfg)

    def runs(mask, label):
        if mask.size == 0:
            return []
        m = np.asarray(mask, dtype=bool)
        padded = np.concatenate([[False], m, [False]])
        edges = np.diff(padded.astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1
        out = []
        for s, e in zip(starts, ends):
            length = e - s + 1
            if length < cfg.min_bout_frames:
                continue
            out.append({
                "AnimalID": animal_id, "FileName": file_name,
                "Object": label, "object_novel": 1 if label == "novel" else 0,
                "bout_start_frame": int(s), "bout_end_frame": int(e),
                "bout_start_sec": float(s / cfg.frame_rate),
                "bout_duration_sec": float(length / cfg.frame_rate),
                "bout_n_frames": int(length),
            })
        return out

    return runs(is_a, "familiar") + runs(is_b, "novel")


# ----------------------------------------------------------------------------
# Input discovery + processing
# ----------------------------------------------------------------------------


def animal_id_from_path(path: Path) -> str:
    """Derive a short animal id from a DeepLabCut file name.

    Strips the DeepLabCut model suffix (everything from 'DLC' onward) and the
    '_filtered' flag so raw and filtered exports collapse to the same id.
    """
    stem = path.stem
    for marker in ("DLC", "DeepCut"):
        idx = stem.find(marker)
        if idx > 0:
            stem = stem[:idx]
            break
    if stem.endswith("_filtered"):
        stem = stem[: -len("_filtered")]
    return stem


def discover_files(root: Path) -> list[Path]:
    """Find DeepLabCut tracking files under root, preferring filtered exports.

    When both a raw and a '_filtered' export exist for the same base name, the
    filtered one is used.
    """
    files: list[Path] = []
    for ext in ("*.csv", "*.h5", "*.hdf5"):
        files.extend(root.rglob(ext))
    by_base: dict[str, Path] = {}
    for f in sorted(files):
        stem = f.stem
        is_f = stem.endswith("_filtered")
        base = stem[: -len("_filtered")] if is_f else stem
        if base not in by_base or is_f:
            by_base[base] = f
    return sorted(by_base.values())


def process_directory(root: Path, cfg: Config) -> list[AnimalResult]:
    results: list[AnimalResult] = []
    for f in discover_files(root):
        try:
            results.append(process_animal(f, cfg))
        except Exception as exc:  # noqa: BLE001
            if cfg.verbose:
                print(f"[ERROR] {f.name}: {exc}")
    return results


# ----------------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------------


def robust_z_scores(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    med = np.nanmedian(x)
    mad_v = median_abs_deviation(x[~np.isnan(x)], scale=1.0)
    sigma = 1.4826 * mad_v
    if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
        q75, q25 = np.nanpercentile(x, [75, 25])
        sigma = (q75 - q25) / 1.349
    if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
        sigma = np.nanstd(x, ddof=0)
    if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
        return np.zeros_like(x)
    return (x - med) / sigma


def hedges_g_vs_zero(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = vals.size
    if n < 2:
        return float("nan")
    sd = np.std(vals, ddof=1)
    if sd <= 0:
        return 0.0
    d_raw = float(np.mean(vals) / sd)
    j = 1 - 3 / (4 * (n - 1) - 1)
    return d_raw * j


def bootstrap_ci_mean(vals: np.ndarray, ci: float, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size < 3:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot_means = np.mean(vals[idx], axis=1)
    lo = np.quantile(boot_means, (1 - ci) / 2)
    hi = np.quantile(boot_means, 1 - (1 - ci) / 2)
    return (float(lo), float(hi))


def lilliefors_or_ks(x: np.ndarray) -> tuple[float, bool]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 4:
        return (float("nan"), False)
    if HAVE_STATSMODELS:
        _, p = lilliefors(x, dist="norm")
        return (float(p), p < 0.05)
    std = np.std(x, ddof=1)
    if std == 0:
        return (1.0, False)
    z = (x - np.mean(x)) / std
    _, p = stats.kstest(z, "norm")
    return (float(p), p < 0.05)


# ----------------------------------------------------------------------------
# Programmatic analysis, returns a dict, can write to disk
# ----------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    per_animal: pd.DataFrame
    summary: dict[str, Any]
    bouts_tbl: pd.DataFrame
    params: dict[str, Any]


def compute_analysis(cfg: Config) -> AnalysisResult:
    rng = np.random.default_rng(42)
    vlog = (lambda *a, **k: print(*a, **k)) if cfg.verbose else (lambda *a, **k: None)

    animals = process_directory(cfg.input_dir, cfg)
    vlog(f"Loaded {len(animals)} tracking files from {cfg.input_dir}")

    # -------- Per-animal DI + exploration + alternative metrics --------
    rows: list[dict[str, Any]] = []
    bout_rows: list[dict[str, Any]] = []
    for a in animals:
        di, tA, tB, nA, nB = compute_di(a.dist_a, a.dist_b, a.angles_a, a.angles_b, cfg)
        alt = compute_alternative_metrics(a.dist_a, a.dist_b, a.angles_a, a.angles_b, cfg)
        bout_rows.extend(extract_bout_records(a.dist_a, a.dist_b, a.angles_a, a.angles_b,
                                              cfg, a.animal_id, a.file_name))
        t_interact = (tA + tB) if not (np.isnan(tA) or np.isnan(tB)) else float("nan")
        # Log-ratio of novel to familiar exploration time
        if not np.isnan(tA) and not np.isnan(tB):
            lr = float(np.log((tB + cfg.eps_t) / (tA + cfg.eps_t)))
        else:
            lr = float("nan")
        rows.append(
            {
                "AnimalID": a.animal_id,
                "FileName": a.file_name,
                "Folder": a.folder,
                "n_frames": a.n_frames,
                "DI": di,
                "LR": lr,
                "tA_sec": tA,
                "tB_sec": tB,
                "tInteract_sec": t_interact,
                "nBoutA": nA,
                "nBoutB": nB,
                "Pass_DI": not np.isnan(di),
                # Exploration metrics
                "time_wall_pct": a.time_wall_pct,
                "time_center_pct": a.time_center_pct,
                "time_near_object_pct": a.time_near_object_pct,
                "distance_m": a.distance_m,
                "mean_velocity_cm_s": a.mean_velocity_cm_s,
                "immobile_pct": a.immobile_pct,
                "center_entries": a.center_entries,
                "path_tortuosity": a.path_tortuosity,
                "low_explorer": a.low_explorer_flag,
                "corners_ok": a.corners_ok,
                "coords_in_bounds": a.coords_in_bounds,
                # Alternative learning metrics
                **alt,
            }
        )
    per_animal = pd.DataFrame(rows)
    bouts_tbl = pd.DataFrame(bout_rows)

    # -------- Robust-z outlier flag on DI across the cohort --------
    if not per_animal.empty:
        per_animal["is_outlier"] = False
        mask = per_animal["DI"].notna()
        if mask.sum() >= 3:
            rz = robust_z_scores(per_animal.loc[mask, "DI"].to_numpy())
            out_idx = per_animal.index[mask][np.abs(rz) > cfg.k_rz]
            per_animal.loc[out_idx, "is_outlier"] = True

    # -------- Cohort summary: does the group discriminate above chance? --------
    summary = summarize_cohort(per_animal, cfg, rng)

    params = {
        "input_dir": str(cfg.input_dir),
        "time_window_sec": list(cfg.time_window_sec) if cfg.time_window_sec else None,
        "frame_rate": cfg.frame_rate,
        "maze_w": cfg.maze_w,
        "maze_h": cfg.maze_h,
        "interact_radius": cfg.interact_radius,
        "interact_angle_deg": cfg.interact_angle_deg,
        "bottle_radius": cfg.bottle_radius,
        "min_total_interact_sec": cfg.min_total_interact_sec,
        "min_bouts_per_obj": cfg.min_bouts_per_obj,
        "min_bout_sec": cfg.min_bout_sec,
        "wall_buffer_cm": cfg.wall_buffer_cm,
        "object_exclusion_radius_cm": cfg.object_exclusion_radius_cm,
        "immobile_speed_cm_s": cfg.immobile_speed_cm_s,
        "likelihood_cutoff": cfg.likelihood_cutoff,
        "k_rz": cfg.k_rz,
        "n_bootstrap": cfg.n_bootstrap,
    }

    return AnalysisResult(
        per_animal=per_animal,
        summary=summary,
        bouts_tbl=bouts_tbl,
        params=params,
    )


def summarize_cohort(per_animal: pd.DataFrame, cfg: Config, rng: np.random.Generator) -> dict[str, Any]:
    """Descriptive summary + a one-sample test of DI against chance (zero)."""
    summary: dict[str, Any] = {
        "n_files": int(len(per_animal)),
        "n_pass_di": 0,
        "n_outliers": 0,
    }
    if per_animal.empty:
        return summary

    included = per_animal.loc[~per_animal.get("is_outlier", False) & per_animal["DI"].notna()]
    vals = included["DI"].to_numpy(dtype=float)
    summary["n_pass_di"] = int(per_animal["DI"].notna().sum())
    summary["n_outliers"] = int(per_animal.get("is_outlier", pd.Series(dtype=bool)).sum())
    summary["n_included"] = int(vals.size)

    if vals.size >= 1:
        summary["mean_DI"] = float(np.mean(vals))
        summary["sd_DI"] = float(np.std(vals, ddof=1)) if vals.size >= 2 else float("nan")
        summary["sem_DI"] = (
            float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size >= 2 else float("nan")
        )
        summary["median_DI"] = float(np.median(vals))
    if vals.size >= 2:
        t_stat, p0 = stats.ttest_1samp(vals, 0.0)
        summary["ttest_vs_chance"] = {"t": float(t_stat), "p": float(p0)}
        summary["hedges_g_vs_chance"] = hedges_g_vs_zero(vals)
        w_stat, w_p = stats.wilcoxon(vals) if np.any(vals != 0) else (float("nan"), float("nan"))
        summary["wilcoxon_vs_chance"] = {"stat": float(w_stat), "p": float(w_p)}
    if vals.size >= 3:
        lo, hi = bootstrap_ci_mean(vals, cfg.bootstrap_ci, cfg.n_bootstrap, rng)
        summary["bootstrap_ci_mean_DI"] = {"lo": lo, "hi": hi, "ci": cfg.bootstrap_ci}
    p_norm, non_normal = lilliefors_or_ks(vals)
    summary["normality"] = {"p": p_norm, "non_normal": bool(non_normal)}

    return summary


# ----------------------------------------------------------------------------
# Save wrapper
# ----------------------------------------------------------------------------


def run_and_save(cfg: Config) -> AnalysisResult:
    res = compute_analysis(cfg)
    if not cfg.save_outputs:
        return res
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_csv = cfg.output_dir / f"NOL_per_animal_{ts}.csv"
    res.per_animal.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    if not res.bouts_tbl.empty:
        out_bouts = cfg.output_dir / f"NOL_bouts_{ts}.csv"
        res.bouts_tbl.to_csv(out_bouts, index=False)
        print(f"Saved: {out_bouts}")

    out_json = cfg.output_dir / f"NOL_summary_{ts}.json"
    out_json.write_text(json.dumps({"params": res.params, "summary": res.summary}, indent=2, default=float))
    print(f"Saved: {out_json}")

    # Figure (built from the per-animal table)
    try:
        from make_nol_figures import plot_summary
        fig_path = cfg.output_dir / f"NOL_summary_{ts}.png"
        plot_summary(res.per_animal, fig_path, title="NOL discrimination summary")
        print(f"Saved: {fig_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] figure step skipped: {exc}")

    return res


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args(argv: list[str]) -> Config:
    p = argparse.ArgumentParser(description="NOL discrimination-index analysis from DeepLabCut output.")
    p.add_argument("--input-dir", required=True, help="Directory of DeepLabCut .csv/.h5 files (searched recursively).")
    p.add_argument("--output-dir", required=True, help="Directory for the per-animal table, summary, and figure.")
    p.add_argument("--frame-rate", type=float, default=60.0)
    p.add_argument("--maze-w", type=float, default=52.07)
    p.add_argument("--maze-h", type=float, default=68.58)
    p.add_argument("--bottle-radius", type=float, default=2.5)
    p.add_argument("--interact-radius", type=float, default=2.5)
    p.add_argument("--interact-angle-deg", type=float, default=60.0)
    p.add_argument("--min-total-interact-sec", type=float, default=2.0)
    p.add_argument("--min-bouts-per-obj", type=int, default=3)
    p.add_argument("--min-bout-sec", type=float, default=0.3)
    p.add_argument("--wall-buffer-cm", type=float, default=8.0)
    p.add_argument("--object-exclusion-radius-cm", type=float, default=5.0)
    p.add_argument("--immobile-speed-cm-s", type=float, default=2.0)
    p.add_argument("--low-explorer-distance-m", type=float, default=3.0)
    p.add_argument("--low-explorer-immobile-pct", type=float, default=80.0)
    p.add_argument("--time-window-sec", default="120,600", help="Start,end seconds, or 'none' for all frames.")
    p.add_argument("--likelihood-cutoff", type=float, default=0.4)
    p.add_argument("--n-bootstrap", type=int, default=5000)

    ns = p.parse_args(argv)
    if ns.time_window_sec.strip().lower() in {"none", "all", ""}:
        tw: tuple[float, float] | None = None
    else:
        parts = [float(x) for x in ns.time_window_sec.split(",")]
        tw = (parts[0], parts[1])

    return Config(
        input_dir=Path(ns.input_dir).resolve(),
        output_dir=Path(ns.output_dir).resolve(),
        frame_rate=ns.frame_rate,
        maze_w=ns.maze_w,
        maze_h=ns.maze_h,
        bottle_radius=ns.bottle_radius,
        interact_radius=ns.interact_radius,
        interact_angle_deg=ns.interact_angle_deg,
        min_total_interact_sec=ns.min_total_interact_sec,
        min_bouts_per_obj=ns.min_bouts_per_obj,
        min_bout_sec=ns.min_bout_sec,
        wall_buffer_cm=ns.wall_buffer_cm,
        object_exclusion_radius_cm=ns.object_exclusion_radius_cm,
        immobile_speed_cm_s=ns.immobile_speed_cm_s,
        low_explorer_distance_m=ns.low_explorer_distance_m,
        low_explorer_immobile_pct=ns.low_explorer_immobile_pct,
        time_window_sec=tw,
        likelihood_cutoff=ns.likelihood_cutoff,
        n_bootstrap=ns.n_bootstrap,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv if argv is not None else sys.argv[1:])
    if not HAVE_CV2:
        print("[INFO] OpenCV not installed, using NumPy DLT homography")
    res = run_and_save(cfg)
    s = res.summary
    if "mean_DI" in s:
        line = f"n={s.get('n_included', 0)} included, mean DI={s['mean_DI']:+.3f}"
        if "ttest_vs_chance" in s:
            line += f", one-sample t vs chance p={s['ttest_vs_chance']['p']:.4f}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

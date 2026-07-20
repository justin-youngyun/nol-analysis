#!/usr/bin/env python
"""
Synthetic DeepLabCut-style tracking generator for the NOL analysis.

This writes fake pose-tracking tables so the analysis can be run and inspected
without any real behavior data. Each animal gets one CSV in the same layout a
DeepLabCut export uses: three header rows (scorer, bodyparts, coords) and one
column triple (x, y, likelihood) per tracked point.

The tracked points match what nol_analysis.py expects:
  body:    Nose, Neck, Back1, Back2, Back3, Tailbase
  objects: ObjA_S (familiar), ObjB_Novel (novel)
  corners: A, B, C, D

Each animal wanders the arena and makes a set of scripted investigation bouts
at the two object locations, with more time at the novel location on average,
so the discrimination index is computable and tends to be positive. Coordinates
are written in pixels (a mild perspective is applied to the arena corners) so
the analysis exercises its homography correction to recover centimetres.

Nothing here is real data. The numbers are drawn from random number generators.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCORER = "DLC_synthetic"
BODYPARTS = [
    "Nose", "Neck", "Back1", "Back2", "Back3", "Tailbase",
    "ObjA_S", "ObjB_Novel", "A", "B", "C", "D",
]
BODY_CHAIN = ["Neck", "Back1", "Back2", "Back3", "Tailbase"]
CHAIN_OFFSET_CM = {"Neck": 2.0, "Back1": 4.0, "Back2": 6.0, "Back3": 8.0, "Tailbase": 11.0}


# ----------------------------------------------------------------------------
# Small geometry helpers (kept local so this generator has no project imports)
# ----------------------------------------------------------------------------


def dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A, dtype=float)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(pts)
    ones = np.ones((pts.shape[0], 1))
    homog = np.hstack([pts, ones]) @ H.T
    return homog[:, :2] / homog[:, 2:3]


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


# ----------------------------------------------------------------------------
# Trajectory construction (all in arena centimetres)
# ----------------------------------------------------------------------------


def _safe_waypoint(rng, w, h, objs, wall_bias=0.6):
    """Pick a target point, biased toward the walls, kept clear of objects."""
    for _ in range(50):
        if rng.random() < wall_bias:
            side = rng.integers(0, 4)
            band = rng.uniform(3.0, 8.0)
            if side == 0:
                p = np.array([band, rng.uniform(3, h - 3)])
            elif side == 1:
                p = np.array([w - band, rng.uniform(3, h - 3)])
            elif side == 2:
                p = np.array([rng.uniform(3, w - 3), band])
            else:
                p = np.array([rng.uniform(3, w - 3), h - band])
        else:
            p = np.array([rng.uniform(6, w - 6), rng.uniform(6, h - 6)])
        if all(np.hypot(*(p - o)) > 8.0 for o in objs):
            return p
    return np.array([w / 2, h / 2])


def _schedule_events(rng, n_frames, fps, obj_a, obj_b, w, h):
    """Build a time-ordered list of investigation bouts for the two objects."""
    n_novel = int(rng.integers(6, 11))
    n_fam = int(rng.integers(3, 7))
    labels = ["novel"] * n_novel + ["familiar"] * n_fam
    rng.shuffle(labels)

    pad = 15.0
    starts = np.sort(rng.uniform(pad, n_frames / fps - pad, size=len(labels)))
    min_gap = int(1.0 * fps)

    events = []
    prev_end = -min_gap
    for t, lab in zip(starts, labels):
        if lab == "novel":
            dur = int(rng.integers(int(0.5 * fps), int(1.3 * fps) + 1))
            center = obj_b
        else:
            dur = int(rng.integers(int(0.4 * fps), int(1.0 * fps) + 1))
            center = obj_a
        start = int(t * fps)
        if start < prev_end + min_gap:
            start = prev_end + min_gap
        end = start + dur
        if end >= n_frames - int(pad * fps):
            continue
        # Approach direction: from the object toward a random interior point
        u = _approach_dir(rng, center, w, h)
        events.append({"start": start, "end": end, "center": center, "u": u})
        prev_end = end
    return events


def _approach_dir(rng, center, w, h):
    for _ in range(50):
        target = np.array([rng.uniform(5, w - 5), rng.uniform(5, h - 5)])
        if np.hypot(*(target - center)) > 12.0:
            u = unit(target - center)
            if _inside(center + 9.0 * u, w, h):
                return u
    return unit(np.array([w / 2, h / 2]) - center)


def _inside(p, w, h, margin=2.0):
    return margin <= p[0] <= w - margin and margin <= p[1] <= h - margin


def build_animal(rng, n_frames, fps, obj_a, obj_b, w, h,
                 r_nose=4.0, track_sigma=0.15):
    """Return per-frame world (cm) coordinates for every body part."""
    events = _schedule_events(rng, n_frames, fps, obj_a, obj_b, w, h)
    in_event = [None] * n_frames
    for e in events:
        for f in range(e["start"], min(e["end"], n_frames)):
            in_event[f] = e

    approach_frames = int(1.5 * fps)
    base_step = 9.0 / fps  # cm per frame at roughly 9 cm/s

    nose = np.zeros((n_frames, 2))
    head = np.zeros((n_frames, 2))
    pos = np.array([w / 2, h / 2])          # body centroid
    heading = unit(np.array([1.0, 0.2]))
    waypoint = _safe_waypoint(rng, w, h, (obj_a, obj_b))
    pause_ctr = 0

    for f in range(n_frames):
        e = in_event[f]
        if e is not None:
            u = e["u"]
            centre = e["center"]
            rn = r_nose + rng.normal(0, 0.1)
            nose[f] = centre + rn * u
            head[f] = -u
            pos = centre + (rn + 5.0) * u
            heading = -u
            continue

        # Steer toward the next object if a bout is imminent, else a waypoint
        target = waypoint
        for e2 in events:
            if 0 < e2["start"] - f <= approach_frames:
                target = e2["center"] + (r_nose + 5.0) * e2["u"]
                break

        d = target - pos
        dist = float(np.hypot(*d))
        direction = d / dist if dist > 1e-6 else heading

        if pause_ctr > 0:
            step = abs(rng.normal(0, 0.01))
            pause_ctr -= 1
        elif rng.random() < 0.008:
            pause_ctr = int(rng.integers(int(0.2 * fps), int(1.2 * fps)))
            step = abs(rng.normal(0, 0.01))
        else:
            step = max(0.0, base_step + rng.normal(0, 0.05))
        step = min(step, dist) if dist > 1e-6 else step

        pos = pos + direction * step
        pos[0] = np.clip(pos[0], 2.0, w - 2.0)
        pos[1] = np.clip(pos[1], 2.0, h - 2.0)
        heading = unit(0.7 * heading + 0.3 * direction)
        nose[f] = pos + 5.0 * heading
        head[f] = heading

        if dist < 1.0:
            waypoint = _safe_waypoint(rng, w, h, (obj_a, obj_b))

    # Build the body chain behind the nose along the heading
    parts = {"Nose": nose.copy()}
    for name, off in CHAIN_OFFSET_CM.items():
        parts[name] = nose - off * head
    # Add independent tracking jitter
    for name in parts:
        parts[name] = parts[name] + rng.normal(0, track_sigma, size=parts[name].shape)
    return parts


# ----------------------------------------------------------------------------
# CSV writing in DeepLabCut layout
# ----------------------------------------------------------------------------


def _likelihood(rng, n, p_drop, high=(0.90, 0.99), low=(0.0, 0.39)):
    lik = rng.uniform(*high, size=n)
    drop = rng.random(n) < p_drop
    lik[drop] = rng.uniform(*low, size=int(drop.sum()))
    return lik


def write_animal_csv(path, parts_world, obj_a, obj_b, corners_world,
                     H_world_to_pixel, rng, n_frames):
    columns = []
    data = {}

    def add_point(bp, world_xy, p_drop):
        px = apply_h(H_world_to_pixel, world_xy)
        lik = _likelihood(rng, len(px), p_drop)
        for coord, series in (("x", px[:, 0]), ("y", px[:, 1]), ("likelihood", lik)):
            columns.append((SCORER, bp, coord))
            data[(SCORER, bp, coord)] = series

    for bp in ["Nose", "Neck", "Back1", "Back2", "Back3", "Tailbase"]:
        add_point(bp, parts_world[bp], p_drop=0.02)

    # Objects and corners are static points across frames
    add_point("ObjA_S", np.tile(obj_a, (n_frames, 1)), p_drop=0.005)
    add_point("ObjB_Novel", np.tile(obj_b, (n_frames, 1)), p_drop=0.005)
    for name, cw in zip(["A", "B", "C", "D"], corners_world):
        add_point(name, np.tile(cw, (n_frames, 1)), p_drop=0.002)

    cols = pd.MultiIndex.from_tuples(columns, names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame({c: data[c] for c in columns}, columns=cols)
    df.index = np.arange(n_frames)
    df.to_csv(path)


def generate(output_dir: Path, n_animals: int, duration_sec: float, fps: float,
             seed: int, maze_w: float, maze_h: float, pixel_to_cm: float) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_frames = int(round(duration_sec * fps))

    # Object locations (cm). A is the familiar location, B is the novel one.
    obj_a = np.array([0.30 * maze_w, 0.35 * maze_h])
    obj_b = np.array([0.70 * maze_w, 0.68 * maze_h])

    # World corners and a mildly skewed pixel image of them (adds perspective)
    world_corners = np.array([[0, 0], [maze_w, 0], [maze_w, maze_h], [0, maze_h]], dtype=float)
    s = 1.0 / pixel_to_cm
    W, Hpx = maze_w * s, maze_h * s
    ox, oy, inset = 50.0, 30.0, 0.04 * W
    pixel_corners = np.array([
        [ox + inset, oy],
        [ox + W - inset, oy],
        [ox + W, oy + Hpx],
        [ox, oy + Hpx],
    ], dtype=float)
    H_world_to_pixel = dlt_homography(world_corners, pixel_corners)

    paths: list[Path] = []
    for i in range(n_animals):
        rng = np.random.default_rng(seed + i)
        parts = build_animal(rng, n_frames, fps, obj_a, obj_b, maze_w, maze_h)
        out = output_dir / f"animal_{i + 1:02d}DLC_synthetic.csv"
        write_animal_csv(out, parts, obj_a, obj_b, world_corners,
                         H_world_to_pixel, rng, n_frames)
        paths.append(out)
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate synthetic DeepLabCut-style NOL tracking data.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-animals", type=int, default=8)
    p.add_argument("--duration-sec", type=float, default=300.0)
    p.add_argument("--frame-rate", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--maze-w", type=float, default=52.07)
    p.add_argument("--maze-h", type=float, default=68.58)
    p.add_argument("--pixel-to-cm", type=float, default=0.07895)
    ns = p.parse_args(argv if argv is not None else sys.argv[1:])

    out_dir = Path(ns.output_dir).resolve()
    paths = generate(
        out_dir, ns.n_animals, ns.duration_sec, ns.frame_rate, ns.seed,
        ns.maze_w, ns.maze_h, ns.pixel_to_cm,
    )
    print(f"Wrote {len(paths)} synthetic tracking files to {out_dir}")
    for pth in paths:
        print(f"  {pth.name}")
    print()
    print("Next, run the analysis with matching frame rate and a full time window.")
    print(f"  python3 nol_analysis.py --input-dir {out_dir} --output-dir outputs "
          f"--frame-rate {ns.frame_rate:g} --time-window-sec 0,{ns.duration_sec:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

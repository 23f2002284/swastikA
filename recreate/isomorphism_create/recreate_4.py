from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import svgwrite
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe
from skimage.morphology import skeletonize, remove_small_objects


# -----------------------------
# Utilities
# -----------------------------
def ensure_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def auto_invert(gray: np.ndarray) -> np.ndarray:
    """
    If background is brighter than foreground (strokes dark), invert.
    We want strokes as white-on-dark for consistent thresholding.
    """
    # Heuristic: check if median is above mean: often bright backgrounds
    mean_val = float(gray.mean())
    if mean_val < 128:
        # likely dark bg, bright strokes: OK
        return gray
    # else likely bright bg or light overall; invert
    return cv2.bitwise_not(gray)


def binarize(gray: np.ndarray, adaptive: bool = True) -> np.ndarray:
    """
    Return a binary mask (np.bool_) with True for stroke pixels.
    Uses adaptive threshold (robust to lighting) by default, else Otsu.
    """
    if adaptive:
        # Block size must be odd; C shifts threshold
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=35, C=-5
        )
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (bw > 0)


def clean_mask(mask: np.ndarray,
               open_iter: int = 1,
               close_iter: int = 1,
               min_object_area: int = 64) -> np.ndarray:
    """
    Clean small noise and fill tiny gaps.
    """
    mask_u8 = (mask.astype(np.uint8) * 255)
    # Morph open/close
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if open_iter > 0:
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=open_iter)
    if close_iter > 0:
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=close_iter)

    # Remove small connected components via skimage
    mask_bool = mask_u8 > 0
    mask_bool = remove_small_objects(mask_bool, min_size=max(1, min_object_area))
    return mask_bool


def thin_to_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Skeletonize a binary mask to a 1-pixel wide centerline.
    """
    # skimage.skeletonize expects boolean foreground == True
    skel = skeletonize(mask)
    return skel


# -----------------------------
# Skeleton to Paths (graph trace)
# -----------------------------
NB_OFFSETS_8: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

def neighbors8(y: int, x: int, h: int, w: int) -> Iterable[Tuple[int, int]]:
    for dy, dx in NB_OFFSETS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            yield ny, nx


def skeleton_degrees(skel: np.ndarray) -> Dict[Tuple[int, int], int]:
    """
    Map each skeleton pixel to its degree (number of 8-neighbors).
    """
    h, w = skel.shape
    deg: Dict[Tuple[int, int], int] = {}
    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        count = 0
        for ny, nx in neighbors8(y, x, h, w):
            if skel[ny, nx]:
                count += 1
        deg[(y, x)] = count
    return deg


def trace_from(seed: Tuple[int, int],
               skel: np.ndarray,
               deg: Dict[Tuple[int, int], int],
               visited_edges: set[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[Tuple[int, int]]:
    """
    Trace a path from a starting pixel, following unvisited edges until termination.
    """
    h, w = skel.shape
    path = [seed]

    # Choose first step
    curr = seed
    prev = None
    # At seed, pick any neighbor to start with
    start_neighbors = [p for p in neighbors8(*curr, h, w) if skel[p]]
    if not start_neighbors:
        return path

    # If degree > 1, we will choose a neighbor whose edge is unvisited
    nxt = None
    for cand in start_neighbors:
        edge = tuple(sorted([curr, cand]))
        if edge not in visited_edges:
            nxt = cand
            break
    if nxt is None:
        # dead start
        return path

    # Walk
    while True:
        # mark edge visited
        visited_edges.add(tuple(sorted([curr, nxt])))
        path.append(nxt)
        prev = curr
        curr = nxt

        # Find next step at curr (avoid going back to prev)
        nbrs = [p for p in neighbors8(*curr, h, w) if skel[p]]
        # Remove previous
        if prev in nbrs:
            nbrs.remove(prev)
        # Prefer an unvisited edge
        nxt = None
        for cand in nbrs:
            edge = tuple(sorted([curr, cand]))
            if edge not in visited_edges:
                nxt = cand
                break
        if nxt is None:
            # ended at endpoint or all edges from here are visited
            break

    return path


def skeleton_to_paths(skel: np.ndarray) -> List[np.ndarray]:
    """
    Convert skeleton into a list of polylines (pixel coordinates).
    We traverse from nodes with deg != 2 (endpoints or junctions).
    Then trace remaining cycles comprised only of deg==2.
    """
    h, w = skel.shape
    deg = skeleton_degrees(skel)
    visited_edges: set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    paths: List[np.ndarray] = []

    # 1) Trace from endpoints and junctions
    special_nodes = [p for p, d in deg.items() if d != 2]
    for seed in special_nodes:
        # From each special node, start along all unvisited edges
        for nbr in neighbors8(*seed, h, w):
            if not skel[nbr]:
                continue
            edge = tuple(sorted([seed, nbr]))
            if edge in visited_edges:
                continue
            # Trace a path starting via this neighbor
            path = trace_from(seed, skel, deg, visited_edges)
            if len(path) > 1:
                paths.append(np.array(path, dtype=float))

    # 2) Trace remaining cycles where all nodes deg==2
    # Find any pixel that still has an unvisited neighbor edge
    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        p = (y, x)
        for q in neighbors8(y, x, h, w):
            if not skel[q]:
                continue
            edge = tuple(sorted([p, q]))
            if edge in visited_edges:
                continue
            # Start a cycle trace from p
            path = trace_from(p, skel, deg, visited_edges)
            if len(path) > 1:
                paths.append(np.array(path, dtype=float))

    return paths


# -----------------------------
# Path processing: simplify, smooth
# -----------------------------
def rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """
    Ramer–Douglas–Peucker simplification. points shape (N,2).
    """
    if len(points) < 3:
        return points

    # perpendicular distance
    def perp_dist(pt, a, b):
        if np.allclose(a, b):
            return np.linalg.norm(pt - a)
        return np.abs(np.cross(b - a, a - pt)) / np.linalg.norm(b - a)

    a, b = points[0], points[-1]
    dmax = -1.0
    idx = -1
    for i in range(1, len(points) - 1):
        d = perp_dist(points[i], a, b)
        if d > dmax:
            dmax, idx = d, i

    if dmax > eps:
        left = rdp(points[: idx + 1], eps)
        right = rdp(points[idx:], eps)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([a, b])


def chaikin(points: np.ndarray, iterations: int = 2, weight: float = 0.25) -> np.ndarray:
    """
    Chaikin corner cutting for smoothing. Keeps endpoints fixed for open polylines.
    """
    if iterations <= 0 or len(points) < 3:
        return points.copy()
    P = points.copy()
    for _ in range(iterations):
        Q = [P[0]]
        for i in range(len(P) - 1):
            p, r = P[i], P[i + 1]
            q = (1 - weight) * p + weight * r
            s = weight * p + (1 - weight) * r
            Q.extend([q, s])
        Q.append(P[-1])
        P = np.array(Q)
    return P


def normalize_and_flip(paths: List[np.ndarray]) -> List[np.ndarray]:
    """
    Convert pixel coords (row, col) to Cartesian (x, y) with y up.
    Also center and scale to a consistent coordinate system for SVG/animation.
    """
    # Convert (row, col) -> (x, y) with y-up
    converted = []
    all_pts = []
    for p in paths:
        xy = np.column_stack([p[:, 1], -p[:, 0]])
        converted.append(xy)
        all_pts.append(xy)
    all_pts = np.vstack(all_pts) if all_pts else np.zeros((0, 2))

    if len(all_pts) == 0:
        return converted

    # Center
    center = all_pts.mean(axis=0, keepdims=True)
    converted = [p - center for p in converted]

    # Scale to approx unit frame (longest side to ~10 units)
    all_pts2 = np.vstack(converted)
    min_xy = all_pts2.min(axis=0)
    max_xy = all_pts2.max(axis=0)
    size = (max_xy - min_xy).max()
    scale = 10.0 / (size if size > 1e-6 else 10.0)
    converted = [p * scale for p in converted]
    return converted


# -----------------------------
# SVG export
# -----------------------------
def paths_to_svg(paths: List[np.ndarray],
                 out_svg: str,
                 stroke_color: str = "#FFFFFF",
                 stroke_width: float = 3.0,
                 canvas_pad: float = 1.0) -> None:
    """
    Save paths as a transparent SVG with stroke only.
    """
    if not paths:
        raise ValueError("No paths to export")

    all_pts = np.vstack(paths)
    minx, miny = all_pts.min(axis=0) - canvas_pad
    maxx, maxy = all_pts.max(axis=0) + canvas_pad
    width = maxx - minx
    height = maxy - miny

    dwg = svgwrite.Drawing(out_svg,
                           size=(f"{width}mm", f"{height}mm"),
                           viewBox=f"{minx} {miny} {width} {height}")
    # Transparent background: do not add any rect
    for poly in paths:
        if len(poly) < 2:
            continue
        d = "M " + " L ".join(f"{x:.4f},{y:.4f}" for x, y in poly)
        dwg.add(dwg.path(d=d,
                         fill="none",
                         stroke=stroke_color,
                         stroke_width=stroke_width,
                         stroke_linecap="round",
                         stroke_linejoin="round"))
    dwg.save()


# -----------------------------
# Animation (Matplotlib)
# -----------------------------
def animate_paths(paths: List[np.ndarray],
                  save_path: Optional[str] = None,
                  seconds: float = 6.0,
                  fps: int = 30,
                  dpi: int = 150,
                  bg: str = "#161616",
                  stroke: str = "white",
                  stroke_width: float = 3.5) -> None:
    """
    Animate drawing of the kolam paths. If save_path ends with .gif or .mp4, saves file.
    """
    if not paths:
        print("Nothing to animate.")
        return

    # merge with NaN separators
    parts = []
    for p in paths:
        parts.append(p)
        parts.append(np.array([[np.nan, np.nan]]))
    P = np.vstack(parts)

    diffs = np.nan_to_num(P[1:] - P[:-1], nan=0.0)
    seglen = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(seglen)])
    Ltot = float(cumlen[-1])

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi, facecolor=bg)
    ax.set_facecolor(bg)
    line, = ax.plot([], [], color=stroke, lw=stroke_width, solid_capstyle="round")
    line.set_path_effects([
        pe.Stroke(linewidth=stroke_width * 2.2, foreground=(1, 1, 1, 0.08)),
        pe.Stroke(linewidth=stroke_width * 1.6, foreground=(1, 1, 1, 0.18)),
        pe.Normal()
    ])

    # Dots estimation is hard generically; omit by default. You can overlay detected circles if needed.

    all_pts = np.vstack(paths)
    minx, miny = all_pts.min(axis=0)
    maxx, maxy = all_pts.max(axis=0)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    span = max(maxx - minx, maxy - miny)
    pad = span * 0.12
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(cx - span / 2 - pad, cx + span / 2 + pad)
    ax.set_ylim(cy - span / 2 - pad, cy + span / 2 + pad)
    ax.axis("off")

    frames = max(2, int(seconds * fps))

    def init():
        line.set_data([], [])
        return (line,)

    def update(f):
        t = f / (frames - 1)
        L = t * Ltot
        idx = np.searchsorted(cumlen, L, side="right")
        vis = P[:idx]
        line.set_data(vis[:, 0], vis[:, 1])
        return (line,)

    anim = FuncAnimation(fig, update, init_func=init, frames=frames, interval=1000 / fps, blit=True)

    if save_path:
        sp = save_path.lower()
        if sp.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        elif sp.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=fps, dpi=dpi)
        else:
            print(f"Unknown save extension for {save_path}. Previewing instead.")
            plt.show()
    else:
        plt.show()


# -----------------------------
# Main pipeline
# -----------------------------
@dataclass
class KolamOptions:
    adaptive_threshold: bool = True
    open_iter: int = 1
    close_iter: int = 1
    min_object_area: int = 64
    rdp_epsilon: float = 0.8
    chaikin_iters: int = 2
    chaikin_weight: float = 0.25


def kolam_from_image(
    image_path: str,
    *,
    options: KolamOptions = KolamOptions(),
    svg_out: Optional[str] = None,
    animate_out: Optional[str] = None,
    show_preview: bool = False
) -> List[np.ndarray]:
    """
    Full pipeline: load, binarize, skeletonize, trace, smooth/simplify, normalize.
    Returns list of polylines in normalized coordinates (x right, y up).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    gray = ensure_gray(bgr)
    gray = auto_invert(gray)

    # mild denoise to help threshold
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    mask = binarize(gray_blur, adaptive=options.adaptive_threshold)
    mask = clean_mask(mask,
                      open_iter=options.open_iter,
                      close_iter=options.close_iter,
                      min_object_area=options.min_object_area)

    skel = thin_to_skeleton(mask)

    # Trace to paths in pixel coords (row,col)
    raw_paths = skeleton_to_paths(skel)

    if show_preview:
        # Quick debug visualization
        plt.figure("Binary/Skel preview", figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Binary mask")
        plt.imshow(mask, cmap="gray")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Skeleton")
        plt.imshow(skel, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=False)

    # Smooth & simplify each path, then normalize
    processed: List[np.ndarray] = []
    for P in raw_paths:
        if len(P) < 2:
            continue
        # Convert to (x,y) in image axis for smoothing
        XY = np.column_stack([P[:, 1], -P[:, 0]])
        # Chaikin smoothing
        XY = chaikin(XY, iterations=max(0, options.chaikin_iters), weight=options.chaikin_weight)
        # RDP simplify
        XY = rdp(XY, eps=max(0.0, options.rdp_epsilon))
        processed.append(XY)

    # Normalize to centered coordinates
    norm_paths = normalize_and_flip(processed)  # already in (x,y up); normalize centers/scales

    # Export SVG (transparent, stroke only)
    if svg_out:
        paths_to_svg(norm_paths, svg_out, stroke_color="#FFFFFF", stroke_width=3.0)

    # Animate
    if animate_out is not None or show_preview:
        animate_paths(norm_paths,
                      save_path=animate_out,
                      seconds=6.0,
                      fps=30,
                      dpi=160,
                      bg="#161616",
                      stroke="white",
                      stroke_width=3.5)
    return norm_paths


if __name__ == "__main__":
    # Example usage:
    # python kolam_from_image.py
    # Edit image_path and outputs below before running.
    image_path = "enhanced_kolam.png"  # <-- replace with your input image
    svg_out = "reconstructed_kolam.svg"
    animate_out = None  # e.g., "reconstructed_kolam.mp4" or "reconstructed_kolam.gif"

    opts = KolamOptions(
        adaptive_threshold=True,
        open_iter=1,
        close_iter=1,
        min_object_area=80,   # increase if many small dust/dots
        rdp_epsilon=0.9,
        chaikin_iters=2,
        chaikin_weight=0.25
    )
    kolam_from_image(
        image_path,
        options=opts,
        svg_out=svg_out,
        animate_out=animate_out,
        show_preview=True
    )
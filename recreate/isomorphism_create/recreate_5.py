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
import networkx as nx  # NEW


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
    mean_val = float(gray.mean())
    if mean_val < 128:
        return gray
    return cv2.bitwise_not(gray)


def binarize(gray: np.ndarray, adaptive: bool = True) -> np.ndarray:
    """
    Return a binary mask (np.bool_) with True for stroke pixels.
    Uses adaptive threshold (robust to lighting) by default, else Otsu.
    """
    if adaptive:
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
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if open_iter > 0:
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=open_iter)
    if close_iter > 0:
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=close_iter)
    mask_bool = mask_u8 > 0
    mask_bool = remove_small_objects(mask_bool, min_size=max(1, min_object_area))
    return mask_bool


def thin_to_skeleton(mask: np.ndarray) -> np.ndarray:
    """Skeletonize a binary mask to a 1-pixel wide centerline."""
    return skeletonize(mask)


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
    """Map each skeleton pixel to its degree (number of 8-neighbors)."""
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
    """Trace a path from a starting pixel, following unvisited edges until termination."""
    h, w = skel.shape
    path = [seed]
    curr = seed
    prev = None
    start_neighbors = [p for p in neighbors8(*curr, h, w) if skel[p]]
    if not start_neighbors:
        return path
    nxt = None
    for cand in start_neighbors:
        edge = tuple(sorted([curr, cand]))
        if edge not in visited_edges:
            nxt = cand
            break
    if nxt is None:
        return path

    while True:
        visited_edges.add(tuple(sorted([curr, nxt])))
        path.append(nxt)
        prev = curr
        curr = nxt
        nbrs = [p for p in neighbors8(*curr, h, w) if skel[p]]
        if prev in nbrs:
            nbrs.remove(prev)
        nxt = None
        for cand in nbrs:
            edge = tuple(sorted([curr, cand]))
            if edge not in visited_edges:
                nxt = cand
                break
        if nxt is None:
            break
    return path


def skeleton_to_paths(skel: np.ndarray) -> List[np.ndarray]:
    """
    Convert skeleton into a list of polylines (pixel coordinates).
    Traverse endpoints/junctions first, then residual cycles.
    """
    h, w = skel.shape
    deg = skeleton_degrees(skel)
    visited_edges: set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    paths: List[np.ndarray] = []
    special_nodes = [p for p, d in deg.items() if d != 2]
    for seed in special_nodes:
        for nbr in neighbors8(*seed, h, w):
            if not skel[nbr]:
                continue
            edge = tuple(sorted([seed, nbr]))
            if edge in visited_edges:
                continue
            path = trace_from(seed, skel, deg, visited_edges)
            if len(path) > 1:
                paths.append(np.array(path, dtype=float))
    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        p = (y, x)
        for q in neighbors8(y, x, h, w):
            if not skel[q]:
                continue
            edge = tuple(sorted([p, q]))
            if edge in visited_edges:
                continue
            path = trace_from(p, skel, deg, visited_edges)
            if len(path) > 1:
                paths.append(np.array(path, dtype=float))
    return paths


# -----------------------------
# Path processing: simplify, smooth
# -----------------------------
def rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """Ramer–Douglas–Peucker simplification. points shape (N,2)."""
    if len(points) < 3:
        return points
    def perp_dist(pt, a, b):
        if np.allclose(a, b):
            return np.linalg.norm(pt - a)
        return np.abs(np.cross(b - a, a - pt)) / np.linalg.norm(b - a)
    a, b = points[0], points[-1]
    dmax, idx = -1.0, -1
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
    """Chaikin corner cutting for smoothing. Keeps endpoints fixed for open polylines."""
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
    Center and scale (longest side ≈ 10 units).
    """
    converted = []
    all_pts = []
    for p in paths:
        xy = np.column_stack([p[:, 1], -p[:, 0]])
        converted.append(xy)
        all_pts.append(xy)
    all_pts = np.vstack(all_pts) if all_pts else np.zeros((0, 2))
    if len(all_pts) == 0:
        return converted
    center = all_pts.mean(axis=0, keepdims=True)
    converted = [p - center for p in converted]
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
    """Save paths as a transparent SVG with stroke only."""
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
    """Animate drawing of the kolam paths."""
    if not paths:
        print("Nothing to animate.")
        return
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
# NEW: Graph compression and isomorphism counting
# -----------------------------
def compress_skeleton_to_graph(skel: np.ndarray) -> nx.Graph:
    """
    Compress the pixel skeleton into a graph whose nodes are junctions/endpoints (deg != 2),
    and edges are maximal degree-2 chains between them (with a 'length' attribute).
    """
    h, w = skel.shape
    deg = skeleton_degrees(skel)
    G = nx.Graph()
    nodes = [p for p, d in deg.items() if d != 2]
    # Edge-tracing from each node along each neighbor
    visited_edges: set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

    for u in nodes:
        uy, ux = u
        for v in neighbors8(uy, ux, h, w):
            if not skel[v]:
                continue
            e = tuple(sorted([u, v]))
            if e in visited_edges:
                continue
            # Walk until next junction/end or dead end
            path = [u, v]
            prev = u
            curr = v
            while True:
                visited_edges.add(tuple(sorted([prev, curr])))
                nbrs = [p for p in neighbors8(*curr, h, w) if skel[p]]
                # remove the node we came from
                if prev in nbrs:
                    nbrs.remove(prev)
                dcurr = deg.get(curr, 0)
                if dcurr != 2 or len(nbrs) == 0:
                    # Stop at junction/end or dead
                    break
                # continue along the unique next
                nxt = nbrs[0]
                path.append(nxt)
                prev, curr = curr, nxt

            # Add edge between endpoints of path
            a = path[0]
            b = path[-1]
            if a == b:
                continue
            # ensure nodes exist with degree attribute for pruning
            G.add_node(a, deg=deg.get(a, 0))
            G.add_node(b, deg=deg.get(b, 0))
            # length as number of steps (or Euclidean length)
            length = float(len(path) - 1)
            G.add_edge(a, b, length=length)
    return G


def count_graph_automorphisms(G: nx.Graph, max_enum: int = 10000) -> Tuple[int, bool, List[Dict]]:
    """
    Count (up to max_enum) automorphisms of G. Returns (count, truncated, samples).
    - truncated=True if enumeration hit the cap.
    """
    # Use degree as a node attribute to prune mappings
    nm = nx.isomorphism.categorical_node_match(['deg'], [None])
    GM = nx.isomorphism.GraphMatcher(G, G, node_match=nm)
    count = 0
    truncated = False
    samples: List[Dict] = []
    for mapping in GM.isomorphisms_iter():
        count += 1
        if len(samples) < 5:
            samples.append(mapping)
        if count >= max_enum:
            truncated = True
            break
    return count, truncated, samples


# -----------------------------
# NEW: Dihedral (D4) symmetry variants of polylines
# -----------------------------
def dihedral_transforms() -> List[np.ndarray]:
    """
    Return 2x2 matrices for the 8 elements of D4: rotations (0,90,180,270) and reflections.
    """
    R0  = np.array([[1, 0],[0, 1]], float)
    R90 = np.array([[0,-1],[1, 0]], float)
    R180= np.array([[-1,0],[0,-1]], float)
    R270= np.array([[0, 1],[-1,0]], float)
    # Reflections: about x, y, y=x, y=-x (compose with rotations as needed)
    Fx  = np.array([[1, 0],[0,-1]], float)   # reflect across x-axis
    Fy  = np.array([[-1,0],[0, 1]], float)   # reflect across y-axis
    Fxy = np.array([[0,1],[1,0]], float)     # reflect across y=x
    Fxny= np.array([[0,-1],[-1,0]], float)   # reflect across y=-x
    return [R0, R90, R180, R270, Fx, Fy, Fxy, Fxny]


def apply_transform_to_paths(paths: List[np.ndarray], M: np.ndarray) -> List[np.ndarray]:
    """Apply 2x2 linear transform to all points and recenter to original centroid."""
    if not paths:
        return paths
    all_pts = np.vstack(paths)
    c = all_pts.mean(axis=0, keepdims=True)
    out = []
    for p in paths:
        q = (p - c) @ M.T + c
        out.append(q)
    return out


def signature_of_paths(paths: List[np.ndarray], samples: int = 512) -> Tuple:
    """
    Robust shape signature: sample approximately 'samples' points along all polylines
    proportionally to length, center/scale, then quantize.
    """
    if not paths:
        return tuple()
    parts = []
    lengths = []
    for p in paths:
        if len(p) < 2: 
            continue
        seg = np.hypot(np.diff(p[:,0]), np.diff(p[:,1]))
        L = float(seg.sum())
        lengths.append(L)
        parts.append((p, L, seg))
    if not parts:
        return tuple()
    total = sum(L for _, L, _ in parts)
    pts = []
    for p, L, seg in parts:
        k = max(2, int(round(samples * (L / total))))
        # uniform parameter t in [0,1] along this polyline
        t = np.linspace(0, 1, k)
        # arc-length parameterization
        cl = np.concatenate([[0.0], np.cumsum(seg)])
        cl /= (cl[-1] if cl[-1] > 0 else 1.0)
        xi = np.interp(t, cl, p[:,0])
        yi = np.interp(t, cl, p[:,1])
        pts.append(np.column_stack([xi, yi]))
    P = np.vstack(pts)
    # center/scale
    P -= P.mean(axis=0, keepdims=True)
    rng = max(np.ptp(P[:,0]), np.ptp(P[:,1]), 1e-6)
    P /= rng
    # quantize
    Q = np.round(P, 3)
    return tuple(Q.flatten().tolist())


def unique_d4_variants(paths: List[np.ndarray]) -> List[List[np.ndarray]]:
    """Generate up to 8 dihedral variants and deduplicate by signature."""
    sigs = set()
    uniq: List[List[np.ndarray]] = []
    for M in dihedral_transforms():
        var = apply_transform_to_paths(paths, M)
        s = signature_of_paths(var)
        if s not in sigs:
            sigs.add(s)
            uniq.append(var)
    return uniq


def animate_variants_sequential(variants: List[List[np.ndarray]],
                                seconds_each: float = 3.0,
                                fps: int = 30,
                                dpi: int = 150,
                                bg: str = "#161616",
                                stroke: str = "white",
                                stroke_width: float = 3.5,
                                save_path: Optional[str] = None) -> None:
    """
    Animate a sequence of variants: draw each variant fully, fade to next.
    """
    if not variants:
        print("No variants to animate.")
        return
    # Precompute plot bounds from first
    base = variants[0]
    all_pts = np.vstack(base)
    minx, miny = all_pts.min(axis=0)
    maxx, maxy = all_pts.max(axis=0)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    span = max(maxx - minx, maxy - miny)
    pad = span * 0.12

    # Prepare path param arrays per variant for distance-based reveal
    reveals = []
    for paths in variants:
        parts = []
        for p in paths:
            parts.append(p)
            parts.append(np.array([[np.nan, np.nan]]))
        P = np.vstack(parts)
        diffs = np.nan_to_num(P[1:] - P[:-1], nan=0.0)
        seglen = np.hypot(diffs[:, 0], diffs[:, 1])
        cumlen = np.concatenate([[0.0], np.cumsum(seglen)])
        Ltot = float(cumlen[-1])
        reveals.append((P, cumlen, Ltot))

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi, facecolor=bg)
    ax.set_facecolor(bg)
    line, = ax.plot([], [], color=stroke, lw=stroke_width, solid_capstyle="round")
    line.set_path_effects([
        pe.Stroke(linewidth=stroke_width * 2.2, foreground=(1, 1, 1, 0.08)),
        pe.Stroke(linewidth=stroke_width * 1.6, foreground=(1, 1, 1, 0.18)),
        pe.Normal()
    ])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(cx - span / 2 - pad, cx + span / 2 + pad)
    ax.set_ylim(cy - span / 2 - pad, cy + span / 2 + pad)
    ax.axis("off")

    frames_each = max(2, int(seconds_each * fps))
    total_frames = frames_each * len(variants)

    def init():
        line.set_data([], [])
        return (line,)

    def update(f):
        vidx = min(len(variants) - 1, f // frames_each)
        local_f = f % frames_each
        P, cumlen, Ltot = reveals[vidx]
        t = local_f / (frames_each - 1)
        L = t * Ltot
        idx = np.searchsorted(cumlen, L, side="right")
        vis = P[:idx]
        line.set_data(vis[:, 0], vis[:, 1])
        return (line,)

    anim = FuncAnimation(fig, update, init_func=init, frames=total_frames, interval=1000 / fps, blit=True)
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
    show_preview: bool = False,
    return_skeleton: bool = False,        # NEW: optionally return the skeleton used
) -> List[np.ndarray] | Tuple[List[np.ndarray], np.ndarray]:
    """
    Full pipeline: load, binarize, skeletonize, trace, smooth/simplify, normalize.
    Returns list of polylines in normalized coordinates (x right, y up).
    If return_skeleton=True, also returns the skeleton image (boolean).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    gray = ensure_gray(bgr)
    gray = auto_invert(gray)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    mask = binarize(gray_blur, adaptive=options.adaptive_threshold)
    mask = clean_mask(mask,
                      open_iter=options.open_iter,
                      close_iter=options.close_iter,
                      min_object_area=options.min_object_area)

    skel = thin_to_skeleton(mask)

    raw_paths = skeleton_to_paths(skel)

    if show_preview:
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

    processed: List[np.ndarray] = []
    for P in raw_paths:
        if len(P) < 2:
            continue
        XY = np.column_stack([P[:, 1], -P[:, 0]])
        XY = chaikin(XY, iterations=max(0, options.chaikin_iters), weight=options.chaikin_weight)
        XY = rdp(XY, eps=max(0.0, options.rdp_epsilon))
        processed.append(XY)

    norm_paths = normalize_and_flip(processed)

    if svg_out:
        paths_to_svg(norm_paths, svg_out, stroke_color="#FFFFFF", stroke_width=3.0)

    if animate_out is not None or show_preview:
        animate_paths(norm_paths,
                      save_path=animate_out,
                      seconds=6.0,
                      fps=30,
                      dpi=160,
                      bg="#161616",
                      stroke="white",
                      stroke_width=3.5)
    if return_skeleton:
        return norm_paths, skel
    return norm_paths


# -----------------------------
# NEW: Report isomorphisms and animate samples
# -----------------------------
def kolam_isomorphism_report(
    image_path: str,
    *,
    options: KolamOptions = KolamOptions(),
    max_automorphisms: int = 20000,
    sample_variants: int = 4,
    save_variants_animation: Optional[str] = None,
) -> Dict[str, object]:
    """
    - Reconstruct kolam.
    - Build compressed skeleton graph and count automorphisms (with cap).
    - Generate unique D4 variants of the geometry.
    - Optionally animate up to 'sample_variants' of them.
    """
    paths, skel = kolam_from_image(image_path, options=options, return_skeleton=True, show_preview=False)

    # Graph automorphisms (topology)
    G = compress_skeleton_to_graph(skel)
    # add degree attribute (already added), ensure correctness
    for n in G.nodes:
        if 'deg' not in G.nodes[n]:
            G.nodes[n]['deg'] = G.degree(n)
    auto_count, truncated, samples = count_graph_automorphisms(G, max_enum=max_automorphisms)

    # D4 geometric variants
    variants = unique_d4_variants(paths)
    n_variants = len(variants)

    # Animate a few variants (geometrically distinct views)
    if sample_variants > 0 and n_variants > 0:
        to_show = variants[:min(sample_variants, n_variants)]
        animate_variants_sequential(
            to_show,
            seconds_each=3.0,
            fps=30,
            dpi=160,
            bg="#161616",
            stroke="white",
            stroke_width=3.5,
            save_path=save_variants_animation
        )

    return {
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "automorphism_count": auto_count,
        "automorphism_truncated": truncated,
        "automorphism_samples": samples,  # first few mappings (node->node)
        "distinct_D4_variants": n_variants,
    }


if __name__ == "__main__":
    # Example usage:
    image_path = "image.png"  # <- your input
    opts = KolamOptions(
        adaptive_threshold=True,
        open_iter=1,
        close_iter=1,
        min_object_area=80,
        rdp_epsilon=0.9,
        chaikin_iters=2,
        chaikin_weight=0.25,
    )

    report = kolam_isomorphism_report(
        image_path,
        options=opts,
        max_automorphisms=20000,          # cap enumeration to avoid explosion
        sample_variants=4,                 # show at most 4 D4 variants animated
        save_variants_animation=None       # e.g., "kolam_variants.mp4" or ".gif"
    )
    print("Isomorphism report:")
    for k, v in report.items():
        print(f"- {k}: {v if k != 'automorphism_samples' else f'{len(v)} mappings saved'}")
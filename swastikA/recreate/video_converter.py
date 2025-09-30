from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, TypedDict
import xml.etree.ElementTree as ET


from manim import (
    Scene,
    SVGMobject,
    Create,
    DrawBorderThenFill,
    FadeIn,
    ORIGIN,
    config,
    tempconfig,
    WHITE,
    BLACK,
)
from langgraph.graph import StateGraph, END


def strip_full_canvas_background_rects(
    svg_path: Path,
    *,
    write_dir: Optional[Path] = None,
) -> Path:
    """
    Remove full-canvas background <rect> elements so Manim doesn't import a solid 'box'.
    Returns a new .svg path in write_dir (or the same directory) if anything is removed,
    otherwise returns the original svg_path.

    Heuristics:
    - rect with width="100%" and height="100%"
    - rect matching the <svg> viewBox or width/height attributes exactly
    """
    try:
        svg_path = Path(svg_path)
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # SVG namespace handling
        ns = {"svg": "http://www.w3.org/2000/svg"}
        # Attempt to read canvas size hints
        viewBox = root.attrib.get("viewBox", "")
        vb_vals = [float(x) for x in viewBox.split() if x.replace(".", "", 1).isdigit()]
        vb_w = vb_vals[2] if len(vb_vals) == 4 else None
        vb_h = vb_vals[3] if len(vb_vals) == 4 else None

        svg_w_attr = root.attrib.get("width")
        svg_h_attr = root.attrib.get("height")

        def matches_canvas(w_attr: Optional[str], h_attr: Optional[str]) -> bool:
            if not w_attr or not h_attr:
                return False
            if w_attr == "100%" and h_attr == "100%":
                return True
            try:
                w = float(w_attr)
                h = float(h_attr)
                if vb_w is not None and vb_h is not None:
                    return abs(w - vb_w) < 1e-6 and abs(h - vb_h) < 1e-6
                if svg_w_attr and svg_h_attr:
                    # If <svg width/height> are numeric and match
                    sw = float(svg_w_attr) if svg_w_attr.replace(".", "", 1).isdigit() else None
                    sh = float(svg_h_attr) if svg_h_attr.replace(".", "", 1).isdigit() else None
                    if sw is not None and sh is not None:
                        return abs(w - sw) < 1e-6 and abs(h - sh) < 1e-6
            except Exception:
                pass
            return False

        removed = False
        # Search rects with or without namespace
        rects = list(root.findall(".//svg:rect", ns)) + list(root.findall(".//rect"))
        for rect in rects:
            w_attr = rect.attrib.get("width")
            h_attr = rect.attrib.get("height")
            if matches_canvas(w_attr, h_attr):
                # Also check if it's a pure background fill (no stroke, at 0,0)
                x = rect.attrib.get("x", "0")
                y = rect.attrib.get("y", "0")
                stroke = rect.attrib.get("stroke", "none")
                if x in ("0", "0.0") and y in ("0", "0.0") and (stroke.lower() == "none"):
                    parent = rect.getparent() if hasattr(rect, "getparent") else None
                    # xml.etree doesn't expose getparent; remove via parent traversal:
                    # Fallback: rebuild without this rect by iterating groups
                    # Easier approach: mark and remove later by searching parent candidates
                    # Instead, brute-force remove by scanning children:
                    for candidate_parent in [root, *root.findall(".//*")]:
                        for child in list(candidate_parent):
                            if child is rect:
                                candidate_parent.remove(child)
                                removed = True
                                break

        if removed:
            out_dir = write_dir if write_dir else svg_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (svg_path.stem + ".stripped.svg")
            tree.write(out_path, encoding="utf-8", xml_declaration=True)
            return out_path

    except Exception:
        # If anything goes wrong, fall back to original file
        pass

    return svg_path


class KolamStroke(Scene):
    def __init__(
        self,
        svg_path: Path,
        stroke_color=WHITE,
        stroke_width: float = 4.0,
        duration: Optional[float] = 5.0,
        background_color=BLACK,
        **kwargs,
    ):
        self.svg_path = Path(svg_path)
        self.stroke_color = stroke_color
        self.stroke_width = float(stroke_width)
        # Allow None for "use default Manim duration"
        self.duration = float(duration) if duration is not None else None
        self._background_color = background_color
        super().__init__(**kwargs)

    def construct(self):
        self.camera.background_color = self._background_color

        svg = SVGMobject(str(self.svg_path), should_center=True)

        # Size/position first so stroke width doesn’t get scaled later
        svg.height = config.frame_height * 0.9
        svg.move_to(ORIGIN)

        # Show the filled shape (white fill on black background)
        svg.set_fill(self.stroke_color, opacity=1.0)  # e.g., WHITE
        svg.set_stroke(self.stroke_color, width=0)     # hide border, or set a small width if you want an outline

        play_kwargs = {}
        if self.duration is not None:
            play_kwargs["run_time"] = float(self.duration)

        # Option 1: draw border then fill
        self.play(DrawBorderThenFill(svg), **play_kwargs)
        # Option 2: just fade in the filled shape (uncomment to use instead)
        # self.play(FadeIn(svg), **play_kwargs)
        
        self.wait(0.5)


def render_kolam(
    svg_path: Path,
    output_dir: Path,
    output_name: str = "kolam",
    *,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    background_color=BLACK,
    stroke_color=WHITE,
    stroke_width: float = 4.0,
    duration: Optional[float] = 5.0,
    renderer: str = "cairo",          # "cairo" or "opengl"
    transparent: bool = False,        # like CLI -t
    save_last_frame: bool = False,    # save still instead of full movie
    strip_background_rects: bool = True,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Render the KolamStroke scene programmatically. Returns (movie_path, image_path).
    Only one will be non-None depending on save_last_frame flag.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optionally sanitize the SVG to ensure no full-canvas background rects.
    svg_to_load = strip_full_canvas_background_rects(svg_path, write_dir=output_dir) if strip_background_rects else Path(svg_path)

    cfg = {
        "renderer": renderer,
        "pixel_width": int(width),
        "pixel_height": int(height),
        "frame_rate": int(fps),
        "media_dir": str(output_dir),
        "output_file": output_name,
        "transparent": bool(transparent),
        "write_to_movie": True,
        "save_last_frame": bool(save_last_frame),
        "movie_file_extension": ".mp4",
        "quality": "medium_quality",
    }

    with tempconfig(cfg):
        scene = KolamStroke(
            svg_path=svg_to_load,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            duration=duration,  # can be None
            background_color=background_color,
        )
        scene.render()

        writer = scene.renderer.file_writer
        movie_path = getattr(writer, "movie_file_path", None)
        image_path = getattr(writer, "image_file_path", None)

        return (Path(movie_path) if movie_path else None,
                Path(image_path) if image_path else None)


class RenderState(TypedDict, total=False):
    # Inputs
    svg_path: str
    output_dir: str
    output_name: str
    width: int
    height: int
    fps: int
    duration: Optional[float]           # allow None
    stroke_width: float
    transparent: bool
    save_last_frame: bool
    strip_background_rects: bool

    # Optional style
    stroke_color: str
    background_color: str

    # Outputs
    video_path: Optional[str]
    image_path: Optional[str]


async def render_kolam_node(state: RenderState) -> RenderState:
    """LangGraph node for rendering kolam animations."""
    # Debug: Print current working directory
    import os
    print(f"\n=== DEBUG: Current working directory: {os.getcwd()}")
    
    svg_path = Path(state["svg_path"])
    print(f"=== DEBUG: Input SVG path: {svg_path.absolute()}")
    print(f"=== DEBUG: SVG exists: {svg_path.exists()}")
    
    if not svg_path.exists():
        raise FileNotFoundError(f"SVG file not found: {svg_path.absolute()}")

    output_dir = Path(state.get("output_dir", "media"))
    output_dir = output_dir.absolute()  # Convert to absolute path
    print(f"=== DEBUG: Output directory: {output_dir}")
    print(f"=== DEBUG: Output directory exists: {output_dir.exists()}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = state.get("output_name", "kolam")
    print(f"=== DEBUG: Output base name: {output_name}")
    print(f"=== DEBUG: Full output paths will be in: {output_dir}")
    print(f"=== DEBUG: Expected video path: {output_dir / (output_name + '.mp4')}")
    print(f"=== DEBUG: Expected image path: {output_dir / (output_name + '.png')}\n")

    width = int(state.get("width", 1920))
    height = int(state.get("height", 1080))
    fps = int(state.get("fps", 30))
    duration = state.get("duration", 5.0)  # can be None
    # Normalize possible string inputs to float or None
    if isinstance(duration, str):
        duration = float(duration) if duration.strip() else None

    stroke_width = float(state.get("stroke_width", 4.0))
    transparent = bool(state.get("transparent", False))
    save_last_frame = bool(state.get("save_last_frame", False))
    strip_rects = bool(state.get("strip_background_rects", True))

    stroke_color = state.get("stroke_color", WHITE)
    background_color = state.get("background_color", BLACK)

    # Debug: Print render configuration
    print(f"\n=== DEBUG: Render Configuration ===")
    print(f"SVG Path: {svg_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Output Name: {output_name}")
    print(f"Dimensions: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Duration: {duration}")
    print(f"Save Last Frame: {save_last_frame}")
    print(f"Transparent: {transparent}\n")
    
    # Run the CPU-bound render_kolam in a thread pool
    import asyncio
    from functools import partial
    
    # Create a partial function with all the arguments
    render_func = partial(
        render_kolam,
        svg_path=svg_path,
        output_dir=output_dir,
        output_name=output_name,
        width=width,
        height=height,
        fps=fps,
        background_color=background_color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        duration=duration,
        transparent=transparent,
        save_last_frame=save_last_frame,
        strip_background_rects=strip_rects,
        renderer="cairo",
    )
    
    try:
        # Run the blocking function in a thread pool
        print("=== DEBUG: Starting render process...")
        movie_path, image_path = await asyncio.to_thread(render_func)
        
        print("\n=== DEBUG: Render completed!")
        print(f"Movie path: {movie_path}")
        print(f"Image path: {image_path}")
        
        if movie_path and isinstance(movie_path, (str, Path)):
            print(f"Movie exists: {Path(movie_path).exists()}")
        if image_path and isinstance(image_path, (str, Path)):
            print(f"Image exists: {Path(image_path).exists()}")
        
        return {
            **state,
            "video_path": str(movie_path) if movie_path else None,
            "image_path": str(image_path) if image_path else None,
        }
    except Exception as e:
        print(f"\n=== DEBUG: Error during render: {str(e)}")
        print("=== DEBUG: Traceback:", exc_info=True)
        raise

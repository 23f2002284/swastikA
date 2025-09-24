import argparse
import base64
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np
from PIL import Image, ImageOps

from skimage.metrics import structural_similarity as ssim
import cv2

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# -----------------------------
# State definition
# -----------------------------
class KolamState(TypedDict, total=False):
    # IO
    target_path: str
    target_image_b64: str
    width: int
    height: int

    # Planning
    plan_json: Dict[str, Any]
    layer_index: int

    # Code gen & execution
    code: str
    workdir: str
    output_path: str
    stdout: str
    stderr: str
    error: str

    # Images
    run_image_b64: str

    # Scores
    ssim: float
    edge_iou: float
    score: float

    # Meta / loop control
    iteration: int
    iter_in_layer: int
    max_iters: int
    max_iters_per_layer: int
    score_threshold: float
    done: bool
    critique: str

    # Logging
    run_dir: str


# -----------------------------
# Utility functions
# -----------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_and_resize_grayscale(path: str, width: int, height: int) -> Image.Image:
    img = Image.open(path).convert("L")
    img = ImageOps.contain(img, (width, height), method=Image.Resampling.LANCZOS)
    if img.size != (width, height):
        # Letterbox to exact size (centered)
        canvas = Image.new("L", (width, height), 255)
        x = (width - img.width) // 2
        y = (height - img.height) // 2
        canvas.paste(img, (x, y))
        img = canvas
    return img


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def b64_to_pil(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s)))


def image_edges_binary(img: Image.Image) -> np.ndarray:
    arr = np.array(img)  # grayscale [0..255]
    edges = cv2.Canny(arr, 100, 200)
    edges = (edges > 0).astype(np.uint8)
    return edges


def edge_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a == 1, b == 1).sum()
    union = np.logical_or(a == 1, b == 1).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def compare_images(target_b64: str, run_b64: str) -> Tuple[float, float, float]:
    target = b64_to_pil(target_b64).convert("L")
    run = b64_to_pil(run_b64).convert("L")

    target_np = np.array(target)
    run_np = np.array(run)

    ssim_score = ssim(target_np, run_np, data_range=255)
    e1 = image_edges_binary(target)
    e2 = image_edges_binary(run)
    e_iou = edge_iou(e1, e2)

    # Weighted score: tweak as needed
    final = 0.7 * ssim_score + 0.3 * e_iou
    return ssim_score, e_iou, final


def now_str():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_jsonl_line(path: Path, obj: Dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def to_data_url(b64: str) -> str:
    return f"data:image/png;base64,{b64}"


# -----------------------------
# Model setup (swap as needed)
# -----------------------------
def get_text_model(model_name: str = "gpt-4o-mini", temperature: float = 0.2):
    # Replace with your preferred LangChain chat model if desired
    import os
    from dotenv import load_dotenv
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=google_api_key)


# -----------------------------
# Prompt templates
# -----------------------------
SYSTEM_PLANNER = """You are an expert kolam analyst. Given a kolam image, produce a JSON plan to reconstruct it layer-by-layer.
- You MUST output valid JSON only.
- Identify a small number of layers (e.g., 2-6), such as: guide-dots grid, primary continuous stroke paths, secondary embellishments (loops, curls), and decorative elements.
- For each layer, describe it briefly and list drawing constraints (stroke widths, symmetry, curve types around dot grid, etc.).
- Include a 'global' section to capture shared parameters (canvas size, pulli grid guess, symmetry axes).
- Keep plan concise but precise, machine-usable, and consistent across iterations."""

SYSTEM_CODER = """You produce Python code to draw a single kolam layer adhering to constraints.
Rules:
- Save final raster output to 'out.png' in the current working directory.
- You may only import from: PIL, numpy, svgwrite, cairosvg, math.
- Prefer vector construction with svgwrite and then rasterize with cairosvg to improve curve smoothness.
- Use a white background and draw strokes in black unless specified.
- Ensure the output image is exactly the requested width and height.
- Do not read any external files. Do not use network.
- Your code must be self-contained and executable with 'python script.py'.
- If there is a pulli grid, align strokes precisely with calculated grid coordinates.
- Keep code readable; comment where key geometry is defined."""

SYSTEM_CRITIC = """You are a meticulous kolam critic and instructor.
You will receive:
- The target image and the current generated image
- Numeric metrics (SSIM, Edge IoU, combined score)
- The current plan layer description and constraints
- The last code or error
Output:
- A concise critique explaining key mismatches (geometry, continuity, stroke width, alignment with dots).
- A prioritized list of revisions for the next attempt.
- If code errored, include a minimal fix strategy.
Be specific, and keep it under ~200 words."""

PROMPT_PLANNER = """Target image (base64 data URL): {data_url}

Canvas: {width}x{height}.

Produce JSON with fields:
- global: {{"canvas": {{"width": {width}, "height": {height}}}, "grid": <guess or null>, "symmetry": <axes or rotational>, "notes": <short> }}
- layers: [{{"name": "...","description": "...","constraints": {{ "stroke_width": <px>, "curve_style": "...", "grid_alignment": "...", "continuity": "..."}}}}]

JSON ONLY:"""

PROMPT_CODER = """Target image (base64 data URL): {target_data_url}
Current layer index: {layer_index}
Layer spec: {layer_spec}

Canvas: width={width}, height={height}

If helpful, you may render helper dots faintly for alignment. Produce code to draw ONLY this layer and the necessary background. Do not draw future layers.

Past critique (if any):
{critique}

Past execution error (if any):
{error}

Generate Python code now (remember to write 'out.png')."""

PROMPT_CRITIC = """Target image: {target_data_url}
Generated image: {run_data_url}

Scores:
- SSIM: {ssim_score:.4f}
- Edge IoU: {edge_iou:.4f}
- Combined: {final_score:.4f}
Threshold: {threshold:.4f}

Current layer index: {layer_index}
Layer spec: {layer_spec}

Last code or error (truncated to 2000 chars):
{code_or_error}

Provide critique and prioritized revision instructions."""
# -----------------------------
# Execution sandbox
# -----------------------------

ALLOWED_IMPORT_HINT = """# Allowed imports:
# from PIL import Image, ImageDraw
# import numpy as np
# import svgwrite
# import cairosvg
# import math
"""

def run_code_in_subprocess(code: str, workdir: Path, timeout_sec: int = 30) -> Tuple[int, str, str, Optional[Path]]:
    """
    Writes code to script.py in workdir and runs it with a timeout.
    Returns (returncode, stdout, stderr, out_path_if_exists)
    """
    ensure_dir(workdir)
    script_path = workdir / "script.py"
    with script_path.open("w", encoding="utf-8") as f:
        f.write(ALLOWED_IMPORT_HINT)
        f.write("\n")
        f.write(code)
        f.write("\n")

    # Ensure clean output
    out_path = workdir / "out.png"
    if out_path.exists():
        out_path.unlink()

    env = os.environ.copy()
    # Prevent interactive backends
    env["MPLBACKEND"] = "Agg"

    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            return 124, "", "TimeoutExpired: script exceeded execution time limit", None
    except Exception as e:
        return 1, "", f"Failed to start subprocess: {e}", None

    out_file = out_path if out_path.exists() else None
    return proc.returncode, stdout or "", stderr or "", out_file


# -----------------------------
# Nodes
# -----------------------------
def node_plan_layers(state: KolamState) -> KolamState:
    model = get_text_model()
    msg = [
        SystemMessage(content=SYSTEM_PLANNER),
        HumanMessage(content=PROMPT_PLANNER.format(
            data_url=to_data_url(state["target_image_b64"]),
            width=state["width"],
            height=state["height"],
        )),
    ]
    resp = model.invoke(msg)
    # Model must return JSON only
    try:
        plan = json.loads(resp.content)
    except Exception as e:
        # Fallback minimal plan if parsing fails
        plan = {
            "global": {
                "canvas": {"width": state["width"], "height": state["height"]},
                "grid": None,
                "symmetry": "unknown",
                "notes": f"Planner JSON parse error: {e}",
            },
            "layers": [
                {
                    "name": "kolam",
                    "description": "Complete kolam strokes",
                    "constraints": {
                        "stroke_width": 4,
                        "curve_style": "smooth",
                        "grid_alignment": "approximate",
                        "continuity": "continuous where possible",
                    },
                }
            ],
        }

    # Save plan
    run_dir = Path(state["run_dir"])
    with (run_dir / "plan.json").open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    state["plan_json"] = plan
    # If layer_index not present, initialize
    state["layer_index"] = state.get("layer_index", 0)
    return state


def node_generate_code(state: KolamState) -> KolamState:
    model = get_text_model(temperature=0.2)
    layers = state["plan_json"].get("layers", [])
    idx = state["layer_index"]
    layer_spec = layers[idx] if 0 <= idx < len(layers) else {"name": f"layer_{idx}", "description": "", "constraints": {}}

    critique = state.get("critique", "").strip() or "None."
    error = state.get("error", "").strip() or "None."

    prompt = PROMPT_CODER.format(
        target_data_url=to_data_url(state["target_image_b64"]),
        layer_index=idx,
        layer_spec=json.dumps(layer_spec, ensure_ascii=False),
        width=state["width"],
        height=state["height"],
        critique=critique,
        error=error,
    )
    msgs = [
        SystemMessage(content=SYSTEM_CODER),
        HumanMessage(content=prompt),
    ]
    resp = model.invoke(msgs)
    code = resp.content

    # Persist code
    run_dir = Path(state["run_dir"])
    code_dir = run_dir / "code"
    ensure_dir(code_dir)
    code_path = code_dir / f"layer_{idx}_iter_{state['iter_in_layer']:04d}.py"
    with code_path.open("w", encoding="utf-8") as f:
        f.write(code)

    state["code"] = code
    return state


def node_execute_code(state: KolamState) -> KolamState:
    # Fresh temp dir for each attempt
    workdir = Path(tempfile.mkdtemp(prefix="kolam_exec_"))
    rc, out, err, out_file = run_code_in_subprocess(state["code"], workdir, timeout_sec=40)

    state["workdir"] = str(workdir)
    state["stdout"] = out
    state["stderr"] = err
    state["error"] = ""

    if rc != 0:
        state["error"] = f"ReturnCode={rc}\nSTDERR:\n{err}\nSTDOUT:\n{out}"
        state["run_image_b64"] = ""
        state["output_path"] = ""
        return state

    if out_file is None or not out_file.exists():
        state["error"] = "Execution succeeded but 'out.png' not found."
        state["run_image_b64"] = ""
        state["output_path"] = ""
        return state

    # Load output and possibly resize to canonical size for comparison
    out_img = Image.open(out_file).convert("L")
    out_img = ImageOps.contain(out_img, (state["width"], state["height"]), method=Image.Resampling.LANCZOS)
    if out_img.size != (state["width"], state["height"]):
        canvas = Image.new("L", (state["width"], state["height"]), 255)
        x = (state["width"] - out_img.width) // 2
        y = (state["height"] - out_img.height) // 2
        canvas.paste(out_img, (x, y))
        out_img = canvas

    out_b64 = pil_to_b64(out_img)
    state["run_image_b64"] = out_b64
    state["output_path"] = str(out_file)

    # Save artifact
    run_dir = Path(state["run_dir"])
    img_path = run_dir / f"layer_{state['layer_index']}_iter_{state['iter_in_layer']:04d}.png"
    out_img.save(img_path)

    return state


def node_compare(state: KolamState) -> KolamState:
    if not state.get("run_image_b64"):
        state["ssim"] = 0.0
        state["edge_iou"] = 0.0
        state["score"] = 0.0
        return state

    s, e, f = compare_images(state["target_image_b64"], state["run_image_b64"])
    state["ssim"] = s
    state["edge_iou"] = e
    state["score"] = f
    return state


def node_critic(state: KolamState) -> KolamState:
    model = get_text_model(temperature=0.1)
    layers = state["plan_json"].get("layers", [])
    idx = state["layer_index"]
    layer_spec = layers[idx] if 0 <= idx < len(layers) else {"name": f"layer_{idx}", "description": "", "constraints": {}}

    code_or_error = state.get("error") or state.get("code") or ""
    code_or_error = code_or_error[:2000]

    prompt = PROMPT_CRITIC.format(
        target_data_url=to_data_url(state["target_image_b64"]),
        run_data_url=to_data_url(state["run_image_b64"]) if state.get("run_image_b64") else "N/A",
        ssim_score=state.get("ssim", 0.0),
        edge_iou=state.get("edge_iou", 0.0),
        final_score=state.get("score", 0.0),
        threshold=state.get("score_threshold", 0.9),
        layer_index=idx,
        layer_spec=json.dumps(layer_spec, ensure_ascii=False),
        code_or_error=code_or_error,
    )
    msgs = [
        SystemMessage(content=SYSTEM_CRITIC),
        HumanMessage(content=prompt),
    ]
    resp = model.invoke(msgs)
    critique = resp.content.strip()
    state["critique"] = critique

    # Log line
    run_dir = Path(state["run_dir"])
    log_path = run_dir / "log.jsonl"
    save_jsonl_line(log_path, {
        "ts": datetime.utcnow().isoformat(),
        "layer_index": idx,
        "iter_in_layer": state["iter_in_layer"],
        "ssim": state.get("ssim", 0.0),
        "edge_iou": state.get("edge_iou", 0.0),
        "score": state.get("score", 0.0),
        "critique": critique,
        "error": state.get("error", ""),
    })
    return state


def decide_next(state: KolamState) -> Literal["retry", "next_layer", "done"]:
    # Stop if max iters
    if state["iteration"] >= state["max_iters"]:
        return "done"

    # If we don't have an image, retry (likely code error)
    if not state.get("run_image_b64"):
        # Per-layer iteration bump will happen through loop
        if state["iter_in_layer"] >= state["max_iters_per_layer"]:
            # Give up on this layer, move on
            return "next_layer"
        return "retry"

    # If score good enough, advance layer
    if state.get("score", 0.0) >= state.get("score_threshold", 0.9):
        # If this was the last layer, done
        layers = state["plan_json"].get("layers", [])
        if state["layer_index"] >= len(layers) - 1:
            return "done"
        return "next_layer"

    # Not good enough, try again unless we've exhausted attempts for this layer
    if state["iter_in_layer"] >= state["max_iters_per_layer"]:
        layers = state["plan_json"].get("layers", [])
        if state["layer_index"] >= len(layers) - 1:
            return "done"
        return "next_layer"

    return "retry"


def node_advance_layer(state: KolamState) -> KolamState:
    state["layer_index"] = state.get("layer_index", 0) + 1
    state["iter_in_layer"] = 0
    # Reset critiques/errors for clarity between layers
    state["critique"] = ""
    state["error"] = ""
    return state


def node_bookkeeping_retry(state: KolamState) -> KolamState:
    state["iteration"] += 1
    state["iter_in_layer"] += 1
    return state


def node_bookkeeping_next(state: KolamState) -> KolamState:
    state["iteration"] += 1
    return state


# -----------------------------
# Build graph
# -----------------------------
def build_graph() -> Any:
    graph = StateGraph(KolamState)

    graph.add_node("plan", node_plan_layers)
    graph.add_node("gen", node_generate_code)
    graph.add_node("exec", node_execute_code)
    graph.add_node("cmp", node_compare)
    graph.add_node("crit", node_critic)
    graph.add_node("retry_tick", node_bookkeeping_retry)
    graph.add_node("next_tick", node_bookkeeping_next)
    graph.add_node("advance", node_advance_layer)

    # Entry: plan once, then first generation
    graph.set_entry_point("plan")
    graph.add_edge("plan", "gen")
    graph.add_edge("gen", "exec")
    graph.add_edge("exec", "cmp")
    graph.add_edge("cmp", "crit")

    # Conditional from crit
    def router(state: KolamState) -> str:
        choice = decide_next(state)
        if choice == "retry":
            return "retry"
        if choice == "next_layer":
            return "next_layer"
        return "done"

    graph.add_conditional_edges(
        "crit",
        router,
        {
            "retry": "retry_tick",
            "next_layer": "next_tick",
            "done": END,
        },
    )

    # After retry bookkeeping, loop back to gen
    graph.add_edge("retry_tick", "gen")
    # After next layer bookkeeping, advance layer then gen
    graph.add_edge("next_tick", "advance")
    graph.add_edge("advance", "gen")

    return graph.compile()


# -----------------------------
# Orchestration
# -----------------------------
def main():
    image_path = "input.jpg"
    width = 1024
    height = 1024
    max_iters = 50
    per_layer_iters = 12
    score_threshold = 0.93
    model = "gemini-2.5-pro"

    # Prepare run dir
    run_dir = Path("runs") / now_str()
    ensure_dir(run_dir)

    # Load and normalize target
    target_img = load_and_resize_grayscale(image_path, width, height)
    target_b64 = pil_to_b64(target_img)
    target_img.save(run_dir / "target_resized.png")

    # Initial state
    state: KolamState = {
        "target_path": os.path.abspath(image_path),
        "target_image_b64": target_b64,
        "width": width,
        "height": height,
        "plan_json": {},
        "layer_index": 0,
        "code": "",
        "workdir": "",
        "output_path": "",
        "stdout": "",
        "stderr": "",
        "error": "",
        "run_image_b64": "",
        "ssim": 0.0,
        "edge_iou": 0.0,
        "score": 0.0,
        "iteration": 0,
        "iter_in_layer": 0,
        "max_iters": max_iters,
        "max_iters_per_layer": per_layer_iters,
        "score_threshold": score_threshold,
        "done": False,
        "critique": "",
        "run_dir": str(run_dir),
    }

    app = build_graph()

    # Drive until END
    final_state = app.invoke(state)  # synchronous run

    # Choose final output
    # If last successful output exists, copy as final
    last_png = None
    for p in sorted(Path(run_dir).glob("layer_*_iter_*.png")):
        last_png = p
    if last_png:
        shutil.copy(last_png, run_dir / "final.png")

    # Dump final state for inspection
    with (run_dir / "final_state.json").open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in final_state.items() if k != "target_image_b64" and k != "run_image_b64"}, f, indent=2)

    print(f"Run complete. Artifacts in: {run_dir}")
    if last_png:
        print(f"Final image: {run_dir / 'final.png'}")
    else:
        print("No successful render produced; see log.jsonl and code snapshots.")


if __name__ == "__main__":
    main()
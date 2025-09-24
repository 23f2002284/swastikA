import os
from typing import Dict, Any, Optional, TypedDict, cast
import numpy as np
from PIL import Image
import torch
from preprocessing.schemas import State
from svg_converter import TS_ImageToSVGStringBW_Potracer


class SVGConverterNode:
    """A LangGraph node for converting images to SVG format."""
    
    def __init__(self, output_dir: str = "svg_output"):
        self.output_dir = output_dir
        self.converter = TS_ImageToSVGStringBW_Potracer()
        
    def _load_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess an image into a PyTorch tensor."""
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(img_array).unsqueeze(0)
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")

    def _convert_to_svg(self, img_tensor: torch.Tensor) -> str:
        """Convert image tensor to SVG string."""
        try:
            svg_string, = self.converter.vectorize(
                image=img_tensor,
                threshold=128,
                turnpolicy="minority",
                turdsize=2,
                corner_threshold=1.0,
                opttolerance=0.2,
                input_foreground="Black on White",
                optimize_curve=True,
                zero_sharp_corners=False,
                foreground_color="#000000",
                background_color="#FFFFFF",
                stroke_color="#000000",
                stroke_width=0.0
            )
            return svg_string
        except Exception as e:
            raise RuntimeError(f"SVG conversion failed: {e}")

    def _save_svg(self, svg_string: str, output_path: str) -> None:
        """Save SVG string to file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(svg_string)
        except Exception as e:
            raise IOError(f"Failed to save SVG: {e}")

    def __call__(self, state: State) -> State:
        """Execute the SVG conversion as a LangGraph node.
        
        Args:
            state: The current state containing the SVG conversion parameters and results.
                Expected to have:
                - input_image_path: Path to the input image
                - output_svg_path: Path to save the output SVG (optional, defaults to 'output.svg')
                - svg_converter: Optional existing SVG converter state
                
        Returns:
            Updated State object with SVG conversion results
        """
        try:
            # Initialize SVG converter state if not present
            if state.svg_converter is None:
                state.svg_converter = {
                    "input_image_path": "",
                    "output_svg_path": "",
                    "svg_string": None,
                    "status": "initialized",
                    "error": None
                }
            
            # Only use recreated_kolam_image_path as the input source
            if not state.recreated_kolam_image_path:
                raise ValueError("recreated_kolam_image_path is required but was not provided in the state")
                
            if not os.path.exists(state.recreated_kolam_image_path):
                raise FileNotFoundError(f"Recreated kolam image not found: {state.recreated_kolam_image_path}")
                
            input_path = state.recreated_kolam_image_path
            output_path = state.svg_converter.get("output_svg_path") or getattr(state, "output_svg_path", "output.svg")
            
            # Update state with paths and mark as processing
            state.svg_converter.update({
                "input_image_path": input_path,
                "output_svg_path": output_path,
                "status": "processing"
            })
            
            # Process the image
            img_tensor = self._load_image(input_path)
            svg_string = self._convert_to_svg(img_tensor)
            self._save_svg(svg_string, output_path)
            
            # Update state with results
            state.svg_converter.update({
                "svg_string": svg_string,
                "status": "completed",
                "error": None
            })
            
        except Exception as e:
            error_msg = str(e)
            if state.svg_converter is not None:
                state.svg_converter.update({
                    "status": "failed",
                    "error": error_msg
                })
            # Re-raise to allow LangGraph to handle the error
            raise RuntimeError(f"SVG conversion failed: {error_msg}") from e
        
        return state


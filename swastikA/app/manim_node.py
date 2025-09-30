# swastikA/app/nodes/manim_node.py
from pathlib import Path
from typing import Any, Dict, Optional

from swastikA.app.schemas import WorkflowState
from swastikA.recreate.video_converter import render_kolam_node, RenderState

class ManimNode:
    """Node for handling Manim rendering in the workflow."""
    
    def __init__(self, output_name: str = "kolam_animation", **kwargs):
        """Initialize the Manim node.
        
        Args:
            output_name: Base name for output files
            **kwargs: Additional arguments to pass to RenderState
        """
        self.output_name = output_name
        self.render_kwargs = {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'duration': 5.0,
            'stroke_width': 4.0,
            'transparent': False,
            'save_last_frame': False,
            **kwargs
        }
    
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """Process the state through the Manim renderer.
        
        Args:
            state: Current workflow state containing preprocessing results
            
        Returns:
            Updated workflow state with manim rendering results
        """
        if not state.preprocessing_state or not state.preprocessing_state.svg_converter:
            raise ValueError("SVG path not available for Manim rendering")
            
        svg_path = state.preprocessing_state.svg_converter.get("output_svg_path")
        if not svg_path:
            raise ValueError("No SVG path or content found in preprocessing state")
            
        # Ensure output directory exists
        output_dir = Path(state.output_folder_path or "output").absolute()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create render state
            render_state = RenderState(
                svg_path=str(Path(svg_path).absolute()),
                output_dir=str(output_dir),
                output_name=self.output_name,
                **self.render_kwargs
            )
            
            # Process with Manim
            result = await render_kolam_node(render_state)
            print(f"Manim rendering completed successfully. Output in: {output_dir}")
            return state.model_copy(update={"manim_state": result})
            
        except Exception as e:
            print(f"Error during Manim rendering: {str(e)}")
            raise
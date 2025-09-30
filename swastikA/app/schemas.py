# workflow_agent/schemas.py
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path


class WorkflowState(BaseModel):
    """Main state for the Kolam processing workflow."""
    # Input/Output paths
    original_image_path: str = Field(
        default="",
        description="Path to the input image in the media directory"
    )
    is_completed: bool = Field(
        default=False,
        description="Whether the kolam is completed or not"
    )
    is_in_frame: bool = Field(
        default=False,
        description="Whether the kolam is in frame or not"
    )
    file_prefix: str = Field(
        default="kolam",
        description="Prefix for generated variation files"
    )
    output_folder_path: str = Field(
        default=str(Path(__file__).parent.parent / "media"),
        description="Output directory for processed files (defaults to swastikA/media)"
    )
    
    # Processing states
    preprocessing_state: Optional['PreprocessingState'] = Field(default=None)
    manim_state: Optional['RenderState'] = Field(default=None)
    analysis_state: Optional['FinalAnalysisSchema'] = Field(default=None)
    variation_state: Optional['VariationRecreateSchema'] = Field(default=None)


from swastikA.preprocessing import State as PreprocessingState
from swastikA.recreate import RenderState, VariationRecreateSchema
from swastikA.analysis import FinalAnalysisSchema
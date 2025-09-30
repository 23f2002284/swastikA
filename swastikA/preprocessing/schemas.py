from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from typing import TypedDict
import torch
from enum import Enum

class TensorType:
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        from pydantic_core import core_schema as cs
        return cs.no_info_after_validator_function(
            lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x),
            cs.any_schema(),
            serialization=cs.plain_serializer_function_ser_schema(lambda x: x.tolist() if x is not None else None),
        )

class EnhancementStatus(str, Enum):
    """Status of the enhancement process."""
    NOT_STARTED = "not_started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class EnhancementState(TypedDict):
    """State for image enhancement process."""
    input_image_path: str
    enhanced_image_path: Optional[str]
    incomplete_image_path: Optional[str]
    in_frame_image_path: Optional[str]
    status: EnhancementStatus
    error: Optional[str]
    metadata: Dict[str, Any]

class SVGConverterState(TypedDict):
    """State for SVG conversion process."""
    input_image_path: str
    output_svg_path: str
    svg_string: Optional[str]
    status: str
    error: Optional[str]

class State(BaseModel):
    """Main application state."""
    # Core state
    is_completed: bool = Field(False, description="Whether the kolam is completed or not.")
    is_in_frame: bool = Field(False, description="Whether the kolam is in frame or not.")
    
    # Image paths
    original_image_path: Optional[str] = Field(None, description="Path to the original input image.")
    completed_image_path: Optional[str] = Field(None, description="Path to the completed image.")
    in_frame_image_path: Optional[str] = Field(None, description="Path to the in frame image.")
    recreated_kolam_image_path: Optional[str] = Field(None, description="Path to the recreated kolam image.")
    
    # Component states
    enhancement: EnhancementState = Field(
        default_factory=lambda: {
            "input_image_path": "",
            "enhanced_image_path": None,
            "incomplete_image_path": None,
            "in_frame_image_path": None,
            "status": EnhancementStatus.NOT_STARTED,
            "error": None,
            "metadata": {}
        },
        description="Enhancement process state."
    )
    
    svg_converter: SVGConverterState = Field(
        default_factory=lambda: {
            "input_image_path": "",
            "output_svg_path": "",
            "svg_string": None,
            "status": "not_started",
            "error": None
        },
        description="SVG conversion state."
    )
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            torch.Tensor: lambda x: x.tolist() if x is not None else None
        }

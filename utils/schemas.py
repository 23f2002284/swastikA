from pydantic import BaseModel, Field 
from typing import Dict, Optional, Any, List, Literal
from datetime import datetime
        
class BaseRecreation(BaseModel):
    """Base schema for kolam recreation outputs."""
    instructions: str = Field(
        ...,
        min_length=20,
        description="Detailed instructions for recreation"
    )
    path: str = Field(
        ...,
        description="File system path to the generated output file"
    )
    chain_of_thought: str = Field(
        ...,
        description="Detailed reasoning process with tool verifications"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used for generation"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the recreation was generated"
    )

class RecreationVariation(BaseRecreation):
    """Schema for kolam variation recreation."""
    variation_type: str = Field(
        ...,
        description="Type of variation applied to the original kolam"
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score to original kolam (0-1)"
    )

class RecreationManim(BaseRecreation):
    """Schema for kolam recreation using Manim."""
    manim_version: str = Field(
        ...,
        description="Version of Manim used for generation"
    )
    animation_duration: float = Field(
        ...,
        gt=0,
        description="Duration of the animation in seconds"
    )
    resolution: tuple[int, int] = Field(
        (1920, 1080),
        description="Resolution of the output video (width, height)"
    )

class RecreationIsomorphism(BaseRecreation):
    """Schema for isomorphic kolam recreation."""
    transformation_matrix: List[List[float]] = Field(
        ...,
        description="Transformation matrix used for isomorphism"
    )
    is_exact: bool = Field(
        ...,
        description="Whether the isomorphism is exact or approximate"
    )

class RecreationUsingDesignPrinciples(BaseRecreation):
    """Schema for kolam recreation using design principles."""
    principles_used: List[str] = Field(
        ...,
        description="List of design principles applied"
    )
    complexity: Literal["low", "medium", "high"] = Field(
        ...,
        description="Complexity level of the design"
    )
    symmetry_type: Optional[str] = Field(
        None,
        description="Type of symmetry used in the design"
    )
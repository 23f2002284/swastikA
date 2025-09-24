from pydantic import BaseModel, Field, field_validator 
from typing import Dict, Optional, Any, List, Literal
from enum import Enum
from datetime import datetime
from pathlib import Path

class AnalysisStatus(str, Enum):
    """Status of the analysis process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class ImageInput(BaseModel):
    """Schema for image input data.
    
    Attributes:
        path: File system path to the image file
        source_url: Optional URL where the image can be downloaded from
        metadata: Additional metadata about the image
    """
    path: str = Field(..., 
                     description="File system path to the image",
                     example="/path/to/kolam.png")
    
    @field_validator('path')
    def validate_path(cls, v):
        if not Path(v).suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
            raise ValueError("Image must be a valid image file (png, jpg, jpeg, bmp)")
        return v

class AnalysisOutput(BaseModel):
    """Schema for analysis results.
    
    Attributes:
        summary: Concise summary of the analysis
        analysis_proof: Detailed proof/explanation of the analysis
        chain_of_thought: Step-by-step reasoning process
        confidence: Confidence score (0-1) of the analysis
        status: Current status of the analysis
        timestamp: When the analysis was performed
    """
    summary: str = Field(
        ...,
        max_length=30,
        description="Concise summary of the analysis"
    )
    analysis_proof: str = Field(
        ...,
        description="Detailed proof/explanation of the analysis"
    )
    chain_of_thought: str = Field(
        ...,
        description="Step-by-step reasoning process with tool verifications"
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the analysis"
    )
    status: AnalysisStatus = Field(
        AnalysisStatus.COMPLETED,
        description="Current status of the analysis"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the analysis was performed"
    )
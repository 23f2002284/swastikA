from typing import List
from pydantic import BaseModel, Field, field_validator

#---------------pydantic Schemas---------------

class CreateQueriesOutput(BaseModel):
    queries: List[str]

class RecreateKolamOutput(BaseModel):
    query: str = Field(..., description="Detailed description of the kolam to generate.")
    image_path: str = Field("recreated_kolam.png", description="Output image path (.png/.jpg/.jpeg).")

    @field_validator("image_path")
    def validate_ext(cls, v):
        if not v.lower().endswith((".png", ".jpg", ".jpeg")):
            raise ValueError("image_path must end with .png, .jpg, or .jpeg")
        return v


class VariationRecreateSchema(BaseModel):
    variation_results: List[RecreateKolamOutput] = Field(..., description="The list of variation results")
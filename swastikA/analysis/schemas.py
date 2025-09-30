from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional

class HypothesisStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"

class EvidenceType(str, Enum):
    OBSERVATION = "observation"
    LOGICAL_INFERENCE = "logical_inference"
    EXTERNAL_KNOWLEDGE = "external_knowledge"

class Evidence(BaseModel):
    """Evidence supporting or refuting a hypothesis"""
    content: str = Field(..., description="Description of the evidence")
    type: EvidenceType = Field(..., description="Type of evidence")
    supports_hypothesis: bool = Field(..., description="Whether this evidence supports the hypothesis")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this piece of evidence (0-1)")

class Hypothesis(BaseModel):
    """A testable hypothesis about the problem"""
    id: str = Field(..., description="Unique identifier for the hypothesis")
    statement: str = Field(..., description="Clear statement of the hypothesis")
    status: HypothesisStatus = Field(default=HypothesisStatus.PENDING, description="Current status of the hypothesis")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in the hypothesis (0-1)")
    

class VerificationResult(BaseModel):
    hypothesis: Hypothesis = Field(..., description="Hypothesis being verified")
    reasoning: str = Field(..., description="Reasoning for the verification result")
    evidence: Optional[List[Evidence]] = Field(default_factory=list, description="Supporting or refuting evidence")
    conclusion: str = Field(..., description="Conclusion of the verification(short summary of conclusion of reasoning,evidence and statement)")


class FinalAnalysisSchema(BaseModel):
    image_path: str = Field(..., description="Path to the image")
    extracted_features : List[Hypothesis] = Field(default_factory=list, description="List of hypotheses")
    verification_results : List[VerificationResult] = Field(default_factory=list, description="List of verification results")


class SearchKolamQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries related to the kolam to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )
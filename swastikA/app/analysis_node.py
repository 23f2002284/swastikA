"""
Analysis node for the Kolam workflow.

This node handles both hypothesis generation and verification in a single step.
"""
from typing import Optional
import logging
from pathlib import Path

from swastikA.app.schemas import WorkflowState
from swastikA.analysis.create_hypothesis_node import KolamFeatureExtractor
from swastikA.analysis.hypothesis_verifier_node import KolamHypothesisVerifier
from swastikA.analysis.schemas import FinalAnalysisSchema, HypothesisStatus

# Set up logging
logger = logging.getLogger(__name__)

class AnalysisNode:
    """Node for analyzing Kolam images by generating and verifying hypotheses.
    
    This node combines both feature extraction (hypothesis generation) and
    hypothesis verification into a single workflow step.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-pro", include_thoughts: bool = False):
        """Initialize the analysis node.
        
        Args:
            model_name: Name of the Gemini model to use
            include_thoughts: Whether to include reasoning in the output
        """
        self.feature_extractor = KolamFeatureExtractor(model_name=model_name)
        self.verifier = KolamHypothesisVerifier(
            model_name=model_name,
            include_thoughts=include_thoughts
        )
        logger.info(f"Initialized AnalysisNode with model: {model_name}")
    
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """Process the state through the analysis workflow.
        
        Args:
            state: The current workflow state containing the image and preprocessing results
            
        Returns:
            Updated workflow state with analysis results
        """
        try:
            if not state.original_image_path:
                raise ValueError("No image path provided in workflow state")
                
            logger.info(f"Starting analysis for: {state.original_image_path}")
            image_path = state.preprocessing_state.recreated_kolam_image_path
            # Create initial analysis state
            analysis = FinalAnalysisSchema(
                image_path=image_path,
                extracted_features=[],
                verification_results=[]
            )
            
            # Step 1: Extract features and generate hypotheses
            logger.info("Extracting features and generating hypotheses...")
            feature_results = await self.feature_extractor.extract_features(
                image_path=image_path
            )
            
            # Update analysis with extracted features
            if feature_results and hasattr(feature_results, 'extracted_features'):
                analysis.extracted_features = feature_results.extracted_features
            
            # Step 2: Verify hypotheses
            if analysis.extracted_features:
                logger.info(f"Verifying {len(analysis.extracted_features)} hypotheses...")
                verified_results = await self.verifier.verify_hypotheses(
                    analysis=analysis,
                    max_parallel_verifications=3
                )
                
                if verified_results and hasattr(verified_results, 'verification_results'):
                    analysis.verification_results = verified_results.verification_results
            
            # Update the workflow state with analysis results
            state.analysis_state = analysis
            logger.info("Successfully completed analysis")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}", exc_info=True)
            # Create a minimal error state
            if not hasattr(state, 'analysis_state') or not state.analysis_state:
                state.analysis_state = FinalAnalysisSchema(
                    image_path=state.preprocessing_state.recreated_kolam_image_path,
                    extracted_features=[],
                    verification_results=[]
                )
            raise

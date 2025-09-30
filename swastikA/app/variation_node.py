from swastikA.app.schemas import WorkflowState
from swastikA.recreate.schemas import VariationRecreateSchema
from swastikA.recreate.variation_recreate_node import variation_recreate_node
import logging
logger = logging.getLogger(__name__)

class VariationRecreateNode:
    """Node for generating kolam variations in a workflow.
    
    This node takes a WorkflowState, generates kolam variations based on the analysis state,
    and updates the workflow state with the results.
    """
    
    def __init__(self, output_dir: str = None, file_prefix: str = None):
        """Initialize the variation recreate node.
        
        Args:
            output_dir: Optional base directory for saving variations.
                      If not provided, it will be set from the WorkflowState.
            file_prefix: Prefix for generated variation files.
        """
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        logger.info(f"Initialized VariationRecreateNode with prefix: {file_prefix}")
    
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """Process the state through the variation recreation workflow.
        
        Args:
            state: The current workflow state containing analysis results
            
        Returns:
            Updated workflow state with variation results
        """
        try:
            if not state.analysis_state:
                raise ValueError("No analysis state available in workflow state")
                
            if not state.original_image_path:
                raise ValueError("No original image path in workflow state")
            image_path = (
                state.preprocessing_state.in_frame_image_path or 
                state.preprocessing_state.completed_image_path or 
                state.preprocessing_state.recreated_kolam_image_path
            )
            # Use provided output_dir or fall back to state's output_folder_path
            original_image_path = image_path
            output_dir = state.output_folder_path
            file_prefix = state.file_prefix
            logger.info(f"Starting variation generation for: {original_image_path}")
            
            # Call the existing variation_recreate_node function
            result = await variation_recreate_node(
                analysis=state.analysis_state,
                original_image_path=original_image_path,
                output_dir=output_dir,
                file_prefix=file_prefix
            )
            
            # Create a new state with only the updated fields
            updated_state = state.model_copy(update={
                "variation_state": VariationRecreateSchema(**result)
            })
            
            logger.info("Successfully completed variation generation")
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in variation generation: {str(e)}", exc_info=True)
            raise
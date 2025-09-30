from typing import Dict, Any, Optional
from langgraph.graph import StateGraph
from swastikA.preprocessing.schemas import State as PreprocessingState, EnhancementStatus
from swastikA.preprocessing.enhance_node import EnhanceKolamNode
from swastikA.preprocessing.svg_converter_node import SVGConverterNode
from swastikA.app.schemas import WorkflowState  # Import the WorkflowState
import logging
import os
import time
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = "kolam_processor.log"):
    """Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(process)d] - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / log_file)
        ]
    )
    
    # Set log level for external libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# Initialize logging
logger = setup_logging()

class PreprocessingNode:
    """Main agent for processing Kolam images through enhancement and SVG conversion.
    
    This processor is designed to work as a node in a LangGraph workflow and uses the
    WorkflowState for input/output.
    """
    
    def __init__(self, output_dir: str = None):
        """Initialize the Kolam processor.
        
        Args:
            output_dir: Optional base directory for saving processed files.
                      If provided, will be used as fallback when not specified in WorkflowState.
        """
        try:
            self.output_dir = os.path.abspath(output_dir) if output_dir else None
            logger.info("Initializing KolamProcessor")
            
            # We'll initialize the workflow when __call__ is invoked with the actual paths
            self.workflow = None
            logger.info("KolamProcessor initialization completed successfully")
            
        except Exception as e:
            logger.critical(f"Failed to initialize KolamProcessor: {str(e)}", exc_info=True)
            raise
    
    def _create_workflow(self, enhanced_dir: str, svg_dir: str) -> StateGraph:
        """Create and configure the processing workflow.
        
        Args:
            enhanced_dir: Directory to store enhanced images
            svg_dir: Directory to store SVG outputs
            
        Returns:
            StateGraph: Configured workflow for Kolam processing
        """
        try:
            logger.debug("Creating new processing workflow")
            
            # Initialize the graph with our custom State
            workflow = StateGraph(PreprocessingState)
            logger.debug("Initialized StateGraph with State model")
            
            # Initialize nodes with output directories
            enhancer = EnhanceKolamNode(output_dir=enhanced_dir)
            svg_converter = SVGConverterNode(output_dir=svg_dir)
            
            # Add nodes to the workflow
            workflow.add_node("enhance_kolam", enhancer)
            workflow.add_node("convert_to_svg", svg_converter)
            logger.debug("Added nodes to workflow: enhance_kolam, convert_to_svg")
            
            # Define the workflow edges
            workflow.add_edge("enhance_kolam", "convert_to_svg")
            logger.debug("Added workflow edge: enhance_kolam -> convert_to_svg")
            
            # Set entry and finish points
            workflow.set_entry_point("enhance_kolam")
            workflow.set_finish_point("convert_to_svg")
            logger.debug("Set entry and finish points for the workflow")
            
            compiled_workflow = workflow.compile()
            logger.info("Successfully compiled processing workflow")
            
            return compiled_workflow
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {str(e)}", exc_info=True)
            raise
    
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """Process the state through the Kolam processing workflow.
        
        Args:
            state: The current workflow state containing input parameters
            
        Returns:
            Updated workflow state with processing results
        """
        try:
            if not state.original_image_path:
                raise ValueError("No image path provided in workflow state")
                
            # Create output directories based on the workflow state
            output_dir = Path(state.output_folder_path)
            enhanced_dir = output_dir / "enhanced"
            svg_dir = output_dir / "svg"
            
            # Ensure directories exist
            enhanced_dir.mkdir(parents=True, exist_ok=True)
            svg_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing image: {state.original_image_path}")
            
            # Initialize the workflow with dynamic output directories
            self.workflow = self._create_workflow(str(enhanced_dir), str(svg_dir))
            
            # Create the initial processing state
            process_state = PreprocessingState(
                original_image_path=state.original_image_path,
                is_completed=state.is_completed,
                is_in_frame=state.is_in_frame,
            )
            
            # Execute the workflow
            result = await self.workflow.ainvoke(process_state)
            
            # Update the workflow state with results
            state.preprocessing_state = result
            
            logger.info("Successfully completed Kolam processing")
            return state
            
        except Exception as e:
            logger.error(f"Error in Kolam processing: {str(e)}", exc_info=True)
            raise



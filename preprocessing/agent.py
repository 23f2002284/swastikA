from typing import Dict, Any, Optional
from langgraph.graph import StateGraph
from preprocessing.schemas import State, EnhancementStatus
from preprocessing.nodes import EnhanceKolamNode, SVGConverterNode
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

class KolamProcessor:
    """Main agent for processing Kolam images through enhancement and SVG conversion."""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize the Kolam processor.
        
        Args:
            output_dir: Base directory for saving processed files. Will be created if it doesn't exist.
        """
        try:
            self.output_dir = os.path.abspath(output_dir)
            logger.info(f"Initializing KolamProcessor with output directory: {self.output_dir}")
            
            # Create output directories if they don't exist
            enhanced_dir = os.path.join(self.output_dir, "enhanced")
            svg_dir = os.path.join(self.output_dir, "svg")
            
            os.makedirs(enhanced_dir, exist_ok=True)
            logger.debug(f"Created/Verified enhanced output directory: {enhanced_dir}")
            
            os.makedirs(svg_dir, exist_ok=True)
            logger.debug(f"Created/Verified SVG output directory: {svg_dir}")
            
            logger.info("Initializing processing workflow...")
            self.workflow = self._create_workflow()
            logger.info("KolamProcessor initialization completed successfully")
            
        except Exception as e:
            logger.critical(f"Failed to initialize KolamProcessor: {str(e)}", exc_info=True)
            raise
    
    def _create_workflow(self) -> StateGraph:
        """Create and configure the processing workflow.
        
        Returns:
            StateGraph: Configured workflow for Kolam processing
        """
        try:
            logger.debug("Creating new processing workflow")
            
            # Initialize the graph
            workflow = StateGraph(State)
            logger.debug("Initialized StateGraph with State model")
            
            # Initialize nodes
            enhanced_dir = os.path.join(self.output_dir, "enhanced")
            svg_dir = os.path.join(self.output_dir, "svg")
            
            logger.info(f"Initializing EnhanceKolamNode with output directory: {enhanced_dir}")
            enhancer = EnhanceKolamNode(output_dir=enhanced_dir)
            
            logger.info(f"Initializing SVGConverterNode with output directory: {svg_dir}")
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
    
    def process_kolam(
        self,
        image_path: str,
        is_completed: bool = False,
        is_in_frame: bool = False
    ) -> Dict[str, Any]:
        """Process a Kolam image through the workflow.
        
        Args:
            image_path: Path to the input Kolam image
            is_completed: Whether the Kolam is already completed
            is_in_frame: Whether the Kolam is properly framed
            
        Returns:
            Dict containing the processing results and status
        """
        process_id = f"{os.getpid()}_{os.path.basename(image_path)}_{int(time.time())}"
        logger.info(f"[{process_id}] Starting Kolam processing for: {image_path}")
        logger.info(f"[{process_id}] Processing parameters - is_completed: {is_completed}, is_in_frame: {is_in_frame}")
        
        try:
            # Verify input file exists
            if not os.path.exists(image_path):
                error_msg = f"Input image not found: {image_path}"
                logger.error(f"[{process_id}] {error_msg}")
                return {
                    "success": False,
                    "status": "failed",
                    "error": error_msg,
                    "metadata": {"process_id": process_id}
                }
            
            # Prepare output paths
            svg_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}.svg"
            output_svg_path = os.path.join(self.output_dir, "svg", svg_filename)
            os.makedirs(os.path.dirname(output_svg_path), exist_ok=True)
            
            # Initialize the state with proper svg_converter settings
            logger.debug(f"[{process_id}] Initializing processing state with SVG output: {output_svg_path}")
            state = State(
                original_image_path=image_path,
                is_completed=is_completed,
                is_in_frame=is_in_frame,
                svg_converter={
                    "input_image_path": image_path,
                    "output_svg_path": output_svg_path,
                    "svg_string": None,
                    "status": "pending",
                    "error": None
                },
                enhancement={
                    "input_image_path": image_path,
                    "enhanced_image_path": None,
                    "incomplete_image_path": None,
                    "in_frame_image_path": None,
                    "status": EnhancementStatus.NOT_STARTED,
                    "error": None,
                    "metadata": {
                        "source": image_path,
                        "process_id": process_id,
                        "start_time": time.time()
                    }
                }
            )
            
            # Run the workflow
            logger.info(f"[{process_id}] Starting workflow execution")
            start_time = time.time()
            result = self.workflow.invoke(state)
            breakpoint()
            processing_time = time.time() - start_time
            
            # Prepare the response with proper dictionary access
            enhancement = result["enhancement"] or {}
            svg_converter = result["svg_converter"] or {}
            
            response = {
                "success": True,
                "status": "completed",
                "process_id": process_id,
                "processing_time_seconds": round(processing_time, 2),
                "enhanced_image": enhancement.get("enhanced_image_path"),
                "svg_output": svg_converter.get("output_svg_path"),
                "metadata": {
                    "process_id": process_id,
                    "processing_steps": enhancement.get("metadata", {}),
                    "processing_time_seconds": round(processing_time, 2),
                    "input_image_size_mb": round(os.path.getsize(image_path) / (1024 * 1024), 2) if os.path.exists(image_path) else 0
                }
            }
            
            logger.info(f"[{process_id}] Successfully completed processing in {processing_time:.2f} seconds")
            logger.debug(f"[{process_id}] Processing results: {response}")
            
            logger.info(f"Successfully processed Kolam: {image_path}")
            return response
            
        except Exception as e:
            error_msg = f"Failed to process Kolam: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "status": "failed",
                "error": error_msg,
                "metadata": {
                    "input_image": image_path,
                    "error_type": type(e).__name__
                }
            }



from preprocessing.prompts import ENHANCE_SYSTEM_PROMPT, ENHANCE_USER_PROMPT, INCOMPLETE_SYSTEM_PROMPT, INCOMPLETE_USER_PROMPT, NOT_IN_FRAME_SYSTEM_PROMPT, NOT_IN_FRAME_USER_PROMPT
from langchain_core.messages import HumanMessage, SystemMessage
from utils.utils import encode_image, save_image, get_image_base64
from utils.llm import get_llm
from preprocessing.schemas import State, EnhancementStatus
from typing import Optional

import logging
import os
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhance_kolam.log')
    ]
)
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhance_kolam.log')
    ]
)
logger = logging.getLogger(__name__)

def _generate_output_path(base_dir: str, prefix: str = "enhanced") -> str:
    """Generate a unique output path for processed images."""
    os.makedirs(base_dir, exist_ok=True)
    return str(Path(base_dir) / f"{prefix}_{uuid.uuid4().hex[:8]}.png")

def vision_llm_call(
    image_path: str,
    system_prompt: str,
    user_prompt: str,
    model_name: str = "gemini-2.5-flash-image-preview",
    temperature: float = 0.2,
    output_dir: str = "output"
) -> Optional[str]:
    """
    Call vision LLM to process an image.
    
    Args:
        image_path: Path to the input image
        system_prompt: System prompt for the LLM
        user_prompt: User prompt for the LLM
        model_name: Name of the model to use
        temperature: Temperature for the model
        output_dir: Directory to save the output image
        
    Returns:
        Path to the saved output image or None if failed
    """
    try:
        output_path = _generate_output_path(output_dir)
        llm = get_llm(model_name, temperature)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64," + encode_image(image_path)
                }
            ])
        ]
        
        response = llm.invoke(messages)
        if not response:
            logger.error("No response received from image processing")
            return None
            
        # Extract and save the image
        image_base64 = get_image_base64(response)
        if not image_base64:
            logger.error("No image data found in the response")
            return None
            
        saved_path = save_image(image_base64, output_path)
        if not saved_path:
            logger.error(f"Failed to save enhanced image to: {output_path}")
            return None
            
        logger.info(f"Successfully saved enhanced image to: {saved_path}")
        return saved_path
    
    except Exception as e:
        logger.error(f"Error in vision_llm_call: {str(e)}", exc_info=True)
        return None



class EnhanceKolamNode:
    """A node for enhancing kolam images using vision LLMs."""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize the EnhanceKolamNode.
        
        Args:
            output_dir: Directory to save processed images
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def __call__(self, state: State) -> State:
        """
        Enhance a kolam image based on the current state.
        
        Args:
            state: Current state containing the image to process
            
        Returns:
            Updated state with enhancement results
        """
        try:
            # Update state to processing
            state.enhancement.update({
                "status": EnhancementStatus.PROCESSING,
                "error": None
            })
            
            # Get input image path from state
            input_path = state.original_image_path or state.enhancement.get("input_image_path", "")
            if not input_path or not os.path.exists(input_path):
                raise FileNotFoundError(f"Input image not found: {input_path}")
            
            # Update state with input path
            state.enhancement["input_image_path"] = input_path
            
            # Track processing steps
            processing_steps = []
            
            # Handle incomplete kolam if needed
            if not state.is_completed:
                enhanced_path = self._process_incomplete_kolam(input_path)
                if enhanced_path:
                    state.enhancement["incomplete_image_path"] = enhanced_path
                    state.completed_image_path = enhanced_path
                    state.is_completed = True
                    input_path = enhanced_path
                    processing_steps.append("incomplete_handling")
            
            # Handle out-of-frame kolam if needed
            if not state.is_in_frame:
                in_frame_path = self._process_out_of_frame_kolam(input_path)
                if in_frame_path:
                    state.enhancement["in_frame_image_path"] = in_frame_path
                    state.in_frame_image_path = in_frame_path
                    state.is_in_frame = True
                    input_path = in_frame_path
                    processing_steps.append("out_of_frame_handling")
            
            # Always perform final enhancement
            enhanced_path = self._enhance_kolam(input_path)
            if enhanced_path:
                # Update enhancement state
                state.enhancement.update({
                    "enhanced_image_path": enhanced_path,
                    "status": EnhancementStatus.COMPLETED,
                    "metadata": {
                        "processing_steps": processing_steps + ["final_enhancement"]
                    }
                })
                
                # Set the final recreated kolam image path
                state.recreated_kolam_image_path = enhanced_path
                
            return state
            
        except Exception as e:
            error_msg = f"Enhancement failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            state.enhancement.update({
                "status": EnhancementStatus.FAILED,
                "error": error_msg
            })
            raise
    
    def _process_incomplete_kolam(self, image_path: str) -> Optional[str]:
        """Process an incomplete kolam image."""
        return vision_llm_call(
            image_path=image_path,
            system_prompt=INCOMPLETE_SYSTEM_PROMPT,
            user_prompt=INCOMPLETE_USER_PROMPT,
            output_dir=os.path.join(self.output_dir, "incomplete")
        )
    
    def _process_out_of_frame_kolam(self, image_path: str) -> Optional[str]:
        """Process an out-of-frame kolam image."""
        return vision_llm_call(
            image_path=image_path,
            system_prompt=NOT_IN_FRAME_SYSTEM_PROMPT,
            user_prompt=NOT_IN_FRAME_USER_PROMPT,
            output_dir=os.path.join(self.output_dir, "in_frame")
        )
    
    def _enhance_kolam(self, image_path: str) -> Optional[str]:
        """Apply final enhancement to the kolam image."""
        return vision_llm_call(
            image_path=image_path,
            system_prompt=ENHANCE_SYSTEM_PROMPT,
            user_prompt=ENHANCE_USER_PROMPT,
            output_dir=self.output_dir  # Use the output_dir directly as it already includes 'enhanced'
        )
        
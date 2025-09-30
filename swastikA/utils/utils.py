import os
import base64
import logging
from pathlib import Path
from typing import Optional, Union
from langchain_core.messages import AIMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('kolam_utils.log')
    ]
)
logger = logging.getLogger(__name__)

def get_image_base64(response: AIMessage) -> Optional[str]:
    """
    Extract base64 encoded image data from AIMessage response.
    
    Args:
        response: AIMessage containing image data
        
    Returns:
        Base64 encoded image string or None if not found
    """
    try:
        if isinstance(response.content, list):
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    image_url_data = block["image_url"]["url"]
                    if "data:image" in image_url_data:
                        logger.debug("Successfully extracted image data from response")
                        return image_url_data.split(",")[-1]
        logger.warning("No valid image data found in the response")
        return None
    except Exception as e:
        logger.error(f"Error extracting image data: {str(e)}", exc_info=True)
        return None

def save_image(image_base64: Optional[str], image_path: Union[str, Path]) -> Optional[str]:
    """
    Save a base64 encoded image to the specified file path.
    
    Args:
        image_base64: Base64 encoded image string
        image_path: Path where the image should be saved
        
    Returns:
        str: Absolute path to the saved image if successful, None otherwise
    """
    if not image_base64 or not isinstance(image_base64, str):
        logger.error("Invalid base64 image data provided")
        return None
        
    try:
        # Convert to Path object if it's a string
        image_path = Path(image_path) if isinstance(image_path, str) else image_path
        image_path = image_path.resolve()  # Get absolute path
        
        logger.info(f"Attempting to save image to: {image_path}")
        
        # Create parent directories if they don't exist
        image_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure the directory is writable
        if not os.access(image_path.parent, os.W_OK):
            error_msg = f"No write permission for directory: {image_path.parent}"
            logger.error(error_msg)
            raise PermissionError(error_msg)
            
        # Handle potential padding issues in base64 string
        padding = len(image_base64) % 4
        if padding:
            image_base64 += '=' * (4 - padding)
            
        # Decode and write the image
        with open(image_path, 'wb') as f:
            f.write(base64.b64decode(image_base64))
            
        logger.info(f"Successfully saved image to: {image_path}")
        return str(image_path.absolute())
        
    except Exception as e:
        logger.error(f"Failed to save image to {image_path}: {str(e)}", exc_info=True)
        return None

def encode_image(image_path: str) -> bytes:
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image

def decode_image(image_base64: str) -> bytes:
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image


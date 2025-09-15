import os
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from utils import get_image_base64, save_image, make_system_prompt
from enhance_kolam.prompts import SUFFIX_PROMPT

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

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-image-preview",
    google_api_key=google_api_key
)

def process_rangoli_image(image_path: str, output_path: str = "output.png") -> Dict[str, Any]:
    """
    Process a rangoli image using the Gemini model.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the enhanced image
        
    Returns:
        Response from the model
    """
    try:
        logger.info(f"Starting image processing for: {image_path}")
        
        # Read and encode the image
        with open(image_path, "rb") as f:
            image_data = f.read()
            logger.debug(f"Successfully read image: {image_path} ({len(image_data)} bytes)")
            
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        system_prompt = make_system_prompt(SUFFIX_PROMPT)
        
        logger.debug("Creating prompt for image enhancement")
        user_prompt = HumanMessage(
            content=[
                {"type": "text", "text": system_prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
            ]
        )
        
        logger.info("Sending request to Gemini model")
        response = llm.invoke([user_prompt])
        logger.info("Successfully received response from Gemini model")
        
        return response
        
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {image_path}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise

def enhance_kolam(image_path: str, output_path: str = "output.png") -> Optional[str]:
    """
    Enhance a kolam/rangoli image.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the enhanced image
        
    Returns:
        Path to the saved enhanced image or None if failed
    """
    try:
        logger.info(f"Starting kolam enhancement for: {image_path}")
        
        # Process the image
        response = process_rangoli_image(image_path, output_path)
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
        logger.error(f"Error in enhance_kolam: {str(e)}", exc_info=True)
        return None
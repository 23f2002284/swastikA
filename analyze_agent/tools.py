from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
from dotenv import load_dotenv
import os
import logging
from pathlib import Path
from datetime import datetime
from utils import get_image_base64, save_image, make_system_prompt
from enhance_kolam import enhance_kolam
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import asyncio

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"kolam_tools_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
#---------------Environment Variables---------------
google_api_key = os.getenv("GOOGLE_API_KEY")

# Log environment setup
logger.info("Initializing Kolam Tools...")
logger.info(f"Log file: {LOG_FILE.absolute()}")

#---------------Primary llm---------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Use the vision model
    google_api_key=google_api_key
)


#---------------pydantic Schemas---------------
class CreateQueriesInput(BaseModel):
    analysis: str = Field(..., description="Natural language analysis text of a kolam design.")

class CreateQueriesOutput(BaseModel):
    queries: List[str]

class RecreateKolamInput(BaseModel):
    query: str = Field(..., description="Detailed description of the kolam to generate.")
    image_path: str = Field("recreated_kolam.png", description="Output image path (.png/.jpg/.jpeg).")

    @field_validator("image_path")
    def validate_ext(cls, v):
        if not v.lower().endswith((".png", ".jpg", ".jpeg")):
            raise ValueError("image_path must end with .png, .jpg, or .jpeg")
        return v

class RecreateKolamOutput(BaseModel):
    image_path: str
    error: Optional[str] = None

class AnalyzeKolamInput(BaseModel):
    image_path: str = Field(..., description="Path to an existing kolam image (png/jpg/jpeg).")

class AnalyzeKolamOutput(BaseModel):
    analysis: str
    error: Optional[str] = None

class EnhanceKolamInput(BaseModel):
    image_path: str
    output_path: str = Field("enhanced_kolam.png")

class EnhanceKolamOutput(BaseModel):
    output_path: Optional[str] = None
    error: Optional[str] = None

class RecreateMultipleKolamsInput(BaseModel):
    analysis: str = Field(..., description="Analysis text of the kolam design to generate variations from.")
    output_dir: str = Field("output", description="Directory to save the generated images")
    file_prefix: str = Field("kolam", description="Prefix for output filenames")

#---------------Helper Functions---------------
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    return image_base64

def decode_image(image_base64: str) -> bytes:
    image_bytes = base64.b64decode(image_base64)
    return image_bytes

def create_kolam_queries(analysis: str) -> Dict[str, Any]:
    """
    Analyzes the design principles from a kolam design and generates queries to recreate similar kolam designs.
    
    Args:
        analysis (str): The design principle analysis response containing kolam design details
        
    Returns:
        str: A set of formatted queries that can be used to generate similar kolam designs
        
    Example:
        analysis = "The kolam features a circular pattern with 8-fold symmetry..."
        queries = create_kolam_queries(analysis)
    """
    logger.info("Generating kolam queries from analysis")
    try:
        from langchain_core.messages import HumanMessage
        from analyze_agent.prompts import RECREATE_KOLAM_QUERIES_SYSTEM_PROMPT, RECREATE_KOLAM_QUERIES_USER_PROMPT
        
        prompt = RECREATE_KOLAM_QUERIES_USER_PROMPT.format(analysis=analysis)
        system_prompt = make_system_prompt(RECREATE_KOLAM_QUERIES_SYSTEM_PROMPT)
        # Format the prompt with the provided response
        
        # Get the response from the language model
        logger.debug(f"Sending prompt to LLM for query generation")
        result = llm.invoke([system_prompt, HumanMessage(content=prompt)])
        lines = [l.strip() for l in result.content.splitlines() if l.strip()]
        # If the model included numbering, strip it
        cleaned = []
        for line in lines:
            cleaned.append(line.lstrip("0123456789).,- ").strip())
        # Enforce exactly 5
        cleaned = cleaned[:5]
        logger.info(f"Generated {len(cleaned)} kolam queries")
        logger.debug(f"Generated queries: {cleaned}")
        return CreateQueriesOutput(queries=cleaned).model_dump()
    except Exception as e:
        logger.error(f"Error generating kolam queries: {str(e)}", exc_info=True)
        return CreateQueriesOutput(queries=[f"ERROR: {e}"]).model_dump()

async def recreate_kolam_save_image(
    query: str, 
    image_path: str = "recreated_kolam.png",
) -> Dict[str, Any]:
    """
    Generate a kolam image based on a descriptive query while maintaining design consistency.
    
    Args:
        query (str): A detailed description of the kolam design to be generated.
        image_path (str, optional): Path where the generated image will be saved.
        
    Returns:
        str: Path to the generated image or error message.
    """
    logger.info(f"Generating kolam image from query. Output: {image_path}")
    try:
        from pathlib import Path
        from analyze_agent.prompts import KOLAM_RECREATION_SYSTEM_PROMPT, KOLAM_RECREATION_PROMPT
        
        # Input validation
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            error_msg = "Image path must end with .png, .jpg, or .jpeg"
            logger.error(error_msg)
            return RecreateKolamOutput(image_path=image_path, error=error_msg).model_dump()
        
        # Create directory if it doesn't exist
        output_dir = Path(image_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Format the prompt with provided parameters
        formatted_prompt = KOLAM_RECREATION_PROMPT.format(
            query=query.strip()
        )
        
        # Get the system prompt
        system_prompt = make_system_prompt(KOLAM_RECREATION_SYSTEM_PROMPT)
        
        # Prepare messages for the model
        messages = [
            system_prompt,
            HumanMessage(content=formatted_prompt)
        ]
        llm_image = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-image-preview",  # Use the vision model
            google_api_key=google_api_key
        )
        
        # Generate the kolam design
        logger.info("Sending request to generate kolam image")
        response = await llm_image.ainvoke(messages)
        
        # Process and save the image
        image_base64 = get_image_base64(response)
        if not image_base64:
            error_msg = "Model did not return an image payload"
            logger.error(error_msg)
            return RecreateKolamOutput(image_path=image_path, error=error_msg).model_dump()

        logger.debug("Saving generated image")
        saved = save_image(image_base64, image_path)
        if not saved:
            error_msg = "Failed to persist the generated image"
            logger.error(error_msg)
            return RecreateKolamOutput(image_path=image_path, error=error_msg).model_dump()

        logger.info(f"Successfully saved kolam to {saved}")
        return RecreateKolamOutput(image_path=str(Path(saved).absolute())).model_dump()
        
    except Exception as e:
        logger.error(f"Error in recreate_kolam_save_image: {str(e)}", exc_info=True)
        return RecreateKolamOutput(image_path=image_path, error=str(e)).model_dump()

@tool("recreate_multiple_kolams", args_schema=RecreateMultipleKolamsInput)
async def recreate_multiple_kolams(
    analysis: str,
    output_dir: str = "output",
    file_prefix: str = "kolam",
) -> Dict[str, Any]:
    """
    Generate multiple kolam images in parallel based on queries from the analysis.
    
    Args:
        analysis (str): The kolam analysis to generate queries from
        output_dir (str): Directory to save the generated images
        file_prefix (str): Prefix for output filenames
        
    Returns:
        Dict[str, Any]: Dictionary containing paths to generated images and any errors
    """
    logger.info(f"Starting multiple kolam generation. Output dir: {output_dir}, Prefix: {file_prefix}")
    try:
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating kolam queries from analysis")
        queries_result = create_kolam_queries(analysis)
        queries = queries_result.get('queries', [])
        
        if not queries:
            error_msg = "No queries generated from analysis"
            logger.warning(error_msg)
            return {"error": error_msg, "generated": []}
        
        logger.info(f"Generating {len(queries)} kolam variations")
        tasks = []
        for i, query in enumerate(queries, 1):
            output_file = output_path / f"{file_prefix}_{i}.png"
            logger.debug(f"Queueing kolam generation {i}: {query[:50]}...")
            tasks.append(recreate_kolam_save_image(query, str(output_file)))
        
        # Run all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        generated = []
        errors = []
        
        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                error_msg = f"Error generating kolam {i}: {str(result)}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
            elif isinstance(result, dict) and 'image_path' in result:
                if result.get('error'):
                    logger.warning(f"Partial error in kolam {i}: {result.get('error')}")
                else:
                    logger.info(f"Successfully generated kolam {i}: {result['image_path']}")
                generated.append({
                    "query": queries[i-1],
                    "image_path": result['image_path'],
                    "error": result.get('error')
                })
            else:
                error_msg = f"Unexpected result for kolam {i}: {result}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        logger.info(f"Completed multiple kolam generation. Success: {len(generated)}/{len(queries)}")
        return {
            "generated": generated,
            "errors": errors if errors else None
        }
        
    except Exception as e:
        error_msg = f"Failed to generate kolams: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "error": error_msg,
            "generated": []
        }

@tool("analyze_kolam", args_schema=AnalyzeKolamInput)
def analyze_kolam(image_path: str) -> Dict[str, Any]:
    """
    Analyze a kolam image and provide a detailed breakdown of its properties.
    
    Args:
        image_path (str): Path to the kolam image file
        
    Returns:
        str: Detailed analysis of the kolam design
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image format is not supported
    """
    logger.info(f"Analyzing kolam image: {image_path}")
    try:
        # Verify image exists and is accessible
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Validate image format
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported image format. Supported formats: {', '.join(valid_extensions)}")
        
        # Encode image to base64
        image_base64 = encode_image(image_path)
        
        # Prepare the analysis request
        user_prompt = HumanMessage(
            content=[
                {"type": "text", "text": "Please analyze this kolam design in detail."},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
            ]
        )
        
        # Get the system prompt from prompts module
        from analyze_agent.prompts import ANALYZE_KOLAM_PROMPT
        system_prompt = make_system_prompt(ANALYZE_KOLAM_PROMPT)
        
        # Get the analysis from the model
        logger.info("Sending request to analyze kolam image")
        response = llm.invoke([system_prompt, user_prompt])
        
        logger.info("Received analysis response")
        return AnalyzeKolamOutput(analysis=response.content).model_dump()
    except Exception as e:
        logger.error(f"Error analyzing kolam image: {str(e)}", exc_info=True)
        return AnalyzeKolamOutput(analysis="", error=str(e)).model_dump()

@tool("enhance_kolam_tool", args_schema=EnhanceKolamInput)
def enhance_kolam_tool(image_path: str='image.jpg', output_path: str = "enhanced_kolam.png") -> Dict[str, Any]:
    """Enhance an existing kolam image.
    
    Args:
        image_path: Path to the input kolam image
        output_path: Where to save the enhanced image (default: "enhanced_kolam.png")
        
    Returns:
        str: Path to the enhanced image or error message
    """
    logger.info(f"Enhancing kolam image: {image_path}")
    try:
        result_path = enhance_kolam(image_path, output_path)
        logger.info(f"Successfully enhanced kolam image: {result_path}")
        return EnhanceKolamOutput(output_path=result_path).model_dump()
    except Exception as e:
        logger.error(f"Error enhancing kolam image: {str(e)}", exc_info=True)
        return EnhanceKolamOutput(error=str(e)).model_dump()

# Export list of tools for convenience
ALL_TOOLS = [
    recreate_multiple_kolams,
    analyze_kolam,
    enhance_kolam_tool,
]
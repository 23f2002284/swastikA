from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from swastikA.utils.utils import get_image_base64, save_image, encode_image
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from swastikA.analysis import FinalAnalysisSchema, HypothesisStatus
# from swastikA.app.schemas import WorkflowState
from swastikA.recreate.schemas import (
    CreateQueriesOutput,
    RecreateKolamOutput,
    VariationRecreateSchema,
)

# --------------------------------- Logging Setup ---------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"variation_recreate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info("Initializing Variation Recreate Node...")
logger.info(f"Log file: {LOG_FILE.absolute()}")

# --------------------------------- Environment -----------------------------------
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    logger.warning("GOOGLE_API_KEY not found in environment.")

# --------------------------------- Base LLM (Text) --------------------------------
text_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
)

# --------------------------------- Helpers ---------------------------------------


def _safe_split_lines(content: str) -> List[str]:
    return [ln.strip() for ln in content.splitlines() if ln.strip()]


def _clean_query_lines(lines: List[str], expected: int = 3) -> List[str]:
    """Clean and validate query lines.
    
    Args:
        lines: List of strings to clean and validate
        expected: Number of queries expected (default: 3)
        
    Returns:
        List of cleaned queries, guaranteed to have 'expected' number of items
    """
    if not isinstance(lines, list):
        lines = [str(lines)] if lines else []
    
    # Clean lines
    cleaned = [line.strip() for line in lines if line.strip()]
    
    # If we don't have enough lines, fill with fallbacks
    if len(cleaned) < expected:
        fallbacks = [
            "introduce moderate radial 6-fold symmetry with simple loop lattice",
            "create a balanced composition with rhythmic dot patterns",
            "design with flowing curves and symmetrical arrangements"
        ]
        cleaned.extend(fallbacks)
    
    # Return exactly the expected number of items
    return cleaned[:expected]


# --------------------------------- Query Generation --------------------------------


def create_kolam_queries(
    analysis: FinalAnalysisSchema,
    image_path: str
) -> Dict[str, Any]:
    """
    Generate 3 kolam variation queries from analysis (and optional variation directives).

    Returns:
        dict conforming to CreateQueriesOutput schema: { 'queries': [str, str, str] }
    """
    logger.info("Generating kolam variation queries")
    try:
        from swastikA.recreate.prompts import (
            RECREATE_KOLAM_VARIATION_QUERIES_SYSTEM_PROMPT as SYSTEM_PROMPT,
            RECREATE_KOLAM_VARIATION_QUERIES_USER_PROMPT as USER_PROMPT,
        )
        analysis = [
            result.hypothesis.statement 
            for result in analysis.verification_results 
            if result.hypothesis.status == HypothesisStatus.VERIFIED
        ]
        analysis = "\n".join(analysis)
        # Verify image exists and is accessible
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Validate image format
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported image format. Supported formats: {', '.join(valid_extensions)}")
        
        # Encode image to base64
        image_base64 = encode_image(image_path)

        user_prompt = USER_PROMPT.format(
            analysis=analysis
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_base64}"
                    }
                ]
            )
        ]

        result = text_llm.invoke(messages)

        raw_output = result.content if isinstance(result.content, str) else ""
        lines = _safe_split_lines(raw_output)
        queries = _clean_query_lines(lines, expected=3)

        logger.debug(f"Raw LLM output:\n{raw_output}")
        logger.info(f"Final queries count: {len(queries)}")

        return CreateQueriesOutput(queries=queries).model_dump()

    except Exception as e:
        logger.error(f"Error generating queries: {e}", exc_info=True)
        fallback = [
            "introduce moderate radial 6-fold symmetry with simple loop lattice emphasizing rhythmic repetition and subtle motif substitution",
        ]
        fallback = _clean_query_lines(fallback, expected=3)
        return CreateQueriesOutput(queries=fallback).model_dump()


# --------------------------------- Image Generation --------------------------------


async def recreate_kolam_save_image(
    query: str,
    original_image_path: str,
    output_image_path: str,
) -> Dict[str, Any]:
    """
    Generate a kolam image from a single query.

    Returns:
        dict conforming to RecreateKolamOutput schema.
    """
    logger.info(f"Generating image for query -> {query[:70]}...")
    from swastikA.recreate.prompts import (
        KOLAM_RECREATION_SYSTEM_PROMPT,
        KOLAM_RECREATION_PROMPT,
    )
    try:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not output_image_path.lower().endswith((".png", ".jpg", ".jpeg")):
            raise ValueError("Image path must end with .png, .jpg, or .jpeg")

        # Create directory if it doesn't exist
        output_dir = Path(output_image_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare messages
        formatted_prompt = KOLAM_RECREATION_PROMPT.format(query=query.strip())
        system_msg = SystemMessage(content=KOLAM_RECREATION_SYSTEM_PROMPT)
        image_base64 = encode_image(original_image_path)
        user_msg = HumanMessage(content=[
            {"type": "text", "text": formatted_prompt},
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_base64}"
            }
        ])
        messages = [system_msg, user_msg]

        image_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-image-preview",
            google_api_key=google_api_key,
        )

        response = await image_llm.ainvoke(messages)
        image_base64 = get_image_base64(response)
        if not image_base64:
            raise RuntimeError("Model did not return an image payload")

        saved_path = save_image(image_base64, Path(output_image_path))
        if not saved_path:
            raise RuntimeError("Failed to save generated image")

        return RecreateKolamOutput(query=query, image_path=saved_path).model_dump()

    except Exception as e:
        logger.error(f"Image generation failed for query: {e}", exc_info=True)
        # Even on failure, return intended path (not guaranteed to exist) to satisfy schema.
        return RecreateKolamOutput(query=query, image_path=output_image_path).model_dump()


# --------------------------------- Batch Generation --------------------------------


async def variation_recreate_node(
    analysis: FinalAnalysisSchema,
    original_image_path: str,
    output_dir: str = "output",
    file_prefix: str = "kolam",
) -> Dict[str, Any]:
    """
    Generate 3 kolam variations (queries + images) in parallel.

    Returns:
        dict conforming to VariationRecreateSchema:
        {
          "variation_results": [
              {"query": "...", "image_path": "..."},
              ...
          ]
        }
    """
    # Debug: Print current working directory and input paths
    import os
    print(f"\n=== VARIATION DEBUG: Current working directory: {os.getcwd()}")
    print(f"=== VARIATION DEBUG: Original image path: {os.path.abspath(original_image_path)}")
    print(f"=== VARIATION DEBUG: Output directory: {os.path.abspath(output_dir)}")
    print(f"=== VARIATION DEBUG: File prefix: {file_prefix}")
    
    output_path = Path(output_dir).absolute()  # Ensure absolute path
    print(f"=== VARIATION DEBUG: Full output path: {output_path}")
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"=== VARIATION DEBUG: Created output directory: {output_path}")
    except Exception as e:
        print(f"=== VARIATION DEBUG: Failed to create output directory: {e}")
        raise
        
    logger.info("Starting batch kolam variation generation")

    # Generate queries
    print("\n=== VARIATION DEBUG: Generating kolam variation queries...")
    queries_obj = create_kolam_queries(
        analysis=analysis, image_path=original_image_path
    )
    queries: List[str] = queries_obj.get("queries", [])
    print(f"=== VARIATION DEBUG: Generated {len(queries)} queries")

    if not queries:
        logger.warning("No queries generated; returning empty list.")
        return VariationRecreateSchema(variation_results=[]).model_dump()

    # Create tasks for parallel processing
    tasks = []
    for idx, q in enumerate(queries, start=1):
        img_path = output_path / f"{file_prefix}_{idx}.png"
        print(f"=== VARIATION DEBUG: Will save variation {idx} to: {img_path}")
        tasks.append(recreate_kolam_save_image(q, original_image_path, str(img_path)))

    # Execute all image generation tasks in parallel
    print("\n=== VARIATION DEBUG: Starting parallel image generation...")
    try:
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)
        print(f"=== VARIATION DEBUG: Completed {len(results_raw)} image generations")
    except Exception as e:
        print(f"=== VARIATION DEBUG: Error in asyncio.gather: {str(e)}")
        raise

    # Process results
    variation_results: List[Dict[str, str]] = []
    for i, res in enumerate(results_raw):
        if isinstance(res, Exception):
            error_msg = f"Unhandled exception for variation {i+1}: {res}"
            logger.error(error_msg, exc_info=True)
            print(f"=== VARIATION DEBUG: {error_msg}")
            
            # Create a failed output
            failed_path = output_path / f"{file_prefix}_{i+1}_FAILED.png"
            print(f"=== VARIATION DEBUG: Creating failed placeholder at: {failed_path}")
            try:
                # Create an empty file to indicate failure
                failed_path.touch(exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create failure marker at {failed_path}: {e}")
            
            variation_results.append(
                RecreateKolamOutput(
                    query=queries[i],
                    image_path=str(failed_path),
                ).model_dump()
            )
        else:
            print(f"=== VARIATION DEBUG: Successfully generated variation {i+1}")
            if isinstance(res, dict) and 'image_path' in res:
                image_path = Path(res['image_path'])
                file_exists = image_path.exists()
                print(f"=== VARIATION DEBUG:   Saved to: {image_path}")
                print(f"=== VARIATION DEBUG:   File exists: {file_exists}")
                
                if not file_exists:
                    logger.warning(f"Expected image file not found: {image_path}")
                    # Optionally create a failed marker or handle missing file
                    res['error'] = "Generated file not found on disk"
            
            variation_results.append(res)

    logger.info(f"Completed batch generation. Variations: {len(variation_results)}")
    return VariationRecreateSchema(variation_results=variation_results).model_dump()



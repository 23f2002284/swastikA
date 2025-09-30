from swastikA.analysis.prompts import (
    INITIAL_KOLAM_ANALYSIS_SYSTEM_PROMPT,
    INITIAL_KOLAM_ANALYSIS_USER_PROMPT,
)
from swastikA.analysis.schemas import (
    FinalAnalysisSchema,
    Hypothesis,
    HypothesisStatus,
    )
from dotenv import load_dotenv
from google.genai import Client, types
import os
from dotenv import load_dotenv
import json


# Load environment variables
load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT")
LOCATION = os.getenv("GCP_LOCATION")




class KolamFeatureExtractor:
    """
    Extracts design principles and features from Kolam images using Gemini 2.0 Flash.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """Initialize the extractor with the Gemini model."""
        self.model_name = model_name
        self.system_prompt = INITIAL_KOLAM_ANALYSIS_SYSTEM_PROMPT
        self.user_prompt = INITIAL_KOLAM_ANALYSIS_USER_PROMPT
        self._client = Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    async def extract_features(
        self,
        image_path: str
    ) -> FinalAnalysisSchema:
        """
        Extract features and generate hypotheses from a Kolam image.
        
        Args:
            image_path: Path to the Kolam image file
            raw_description: Optional text description of the Kolam
            observed_features: List of observed features
            prior_notes: Any prior knowledge or notes about the Kolam
            
        Returns:
            Dictionary containing extracted features and hypotheses
        """
        # Load and process the image
        try:
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
            
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")

        try:
            # Initialize client if not already done
            if self._client is None:
                self._client = Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
                
            # Generate the response
            response = await self._client.aio.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    self.system_prompt,
                    self.user_prompt,
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/jpeg',
                    )
                ]
            )
             
            # Parse the response
            result = self._parse_response(response.text)
            result.image_path = image_path
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error generating features: {str(e)}")

    def _parse_response(self, response_text: str) -> FinalAnalysisSchema:
        """
        Parse the model's response into a structured format.
        
        Args:
            response_text: Raw text response from the model
            
        Returns:
            Dictionary containing parsed features and hypotheses
        """
        try:
            # Clean the response text
            text = response_text.strip()
            
            # Handle markdown code blocks
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].strip()
            
            # Parse JSON
            result = json.loads(text)
            features = []
            for feature in result["extracted_features"]:
                features.append(Hypothesis(**feature))
            
            # Validate the structure
            if not isinstance(result, dict):
                raise ValueError("Response is not a JSON object")
                
            if "extracted_features" not in result:
                raise ValueError("Response missing 'extracted_features' field")
                
            # Ensure verification_results is an empty list
            result["verification_results"] = []
            
            return FinalAnalysisSchema(image_path="", extracted_features=features, verification_results=result["verification_results"])
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing response: {str(e)}")



import anthropic
import base64
import json
import os
from PIL import Image
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import io
import ipdb


# Load environment variables from .env file
load_dotenv()

def extract_wine_info_from_image(image, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract wine information from a wine label image using Anthropic's Claude API.

    Args:
        image_path (str): Path to the wine label image
        api_key (str, optional): Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var

    Returns:
        Dict containing extracted wine information
    """

    # Initialize Anthropic client
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")

    client = anthropic.Anthropic(api_key=api_key)

    # Resize and prepare image
    processed_image = resize_image_for_api(image)

    # Convert image to base64
    image_base64 = encode_image_to_base64(processed_image)


    # Create the prompt for wine information extraction
    system_prompt = """You are an expert wine sommelier and label reader. Your task is to carefully analyze wine label images and extract specific information.

Return the information in JSON format with exactly these fields:
- wine_type: The type of wine (Red, White, Rosé, Sparkling, Dessert, etc.)
- grape_varieties: List of grape varieties mentioned on the label and included in the next list (list: ['Cabernet Sauvignon', 'Chardonnay', 'Merlot', 'Pinot Noir', 'Syrah/Shiraz', 'Cabernet Franc', 'Grenache', 'Sauvignon Blanc', 'Riesling', 'Malbec', 'Sangiovese', 'Tempranillo', 'Touriga Nacional', 'Mourvedre', 'Petit Verdot', 'Nebbiolo', 'Corvina', 'Viognier', 'Zinfandel', 'Tinta Roriz', 'Glera/Prosecco', 'Touriga Franca', 'Rondinella', 'Carmenère', 'Sémillon', 'Chenin Blanc', 'Barbera', 'Garnacha', 'Carignan/Cariñena', 'Cinsault', 'Pinot Meunier', 'Gewürztraminer', 'Pinot Blanc', 'Pinot Grigio', 'Primitivo', 'Spätburgunder', 'Montepulciano', 'Gamay Noir', 'Molinara', 'Pinot Gris', 'Petite Sirah', 'Roussanne', 'Tinta Barroca', 'Alicante Bouschet', 'Muscat/Moscato', 'Grüner Veltliner', 'Aragonez', 'Tannat', "Nero d'Avola", 'Pinotage', 'Macabeo', 'Trebbiano', 'Zweigelt', 'Malvasia', 'Negroamaro', 'Grenache Blanc', 'Garganega', 'Muscat Blanc', 'Corvinone', 'Grauburgunder'])
- body: Wine body classification (Very light-bodied, Light-bodied, Medium-bodied, Full-bodied, Very full-bodied)
- acidity: Acidity level if mentioned (High, Medium, Low)
- country: Country of origin
- region: Region of origin (if specified). Omit AOC or DOC designations.
- ABV: Alcohol by volume percentage if mentioned (e.g., 12.5%)

If information is not visible or unclear on the label, use null for that field. Be accurate and only extract information that is clearly visible on the label."""

    user_prompt = """Please analyze this wine label image and extract the following information:

1. Wine type (Red, White, Rosé, Sparkling, etc.)
2. Grape varieties (specific grape names if listed)
3. Body classification (if mentioned or can be inferred from wine type/style)
4. Acidity level (if mentioned)
5. Country of origin
6. Region of Origin
7. ABV (Alcohol by volume percentage if mentioned)

Return the results in JSON format as specified."""

    try:
        # Send request to Claude
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Using latest Claude model
            max_tokens=1000,
            temperature=0,  # Low temperature for consistent extraction
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]
        )

        # Parse the response
        response_text = message.content[0].text

        # Try to extract JSON from the response
        try:
            # Look for JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                wine_info = json.loads(json_str)
            else:
                # Fallback: try to parse the entire response as JSON
                wine_info = json.loads(response_text)

        except json.JSONDecodeError:
            # If JSON parsing fails, return structured default with raw response
            wine_info = {
                "wine_type": None,
                "grape_varieties": None,
                "body": None,
                "acidity": None,
                "country": None,
                "region": None,
                "raw_response": response_text,
                "error": "Failed to parse JSON response"
            }

        # Ensure all expected fields are present
        expected_fields = ["wine_type", "grape_varieties", "body", "acidity", "country", "region", "ABV"]
        for field in expected_fields:
            if field not in wine_info:
                wine_info[field] = None

        # Add metadata
        #wine_info["image_path"] = image_path
        wine_info["extraction_successful"] = True

        return wine_info

    except Exception as e:
        # Return error information
        return {
            "wine_type": None,
            "grape_varieties": None,
            "body": None,
            "acidity": None,
            "country": None,
            "region": None,
            "ABV": None,
            #"image_path": image_path,
            "extraction_successful": False,
            "error": str(e)
        }

def resize_image_for_api(image_file, max_size: int = 400) -> str:
    """
    Resize image to maximum 400x400 pixels while maintaining aspect ratio.

    Args:
        image_path (str): Path to the original image
        max_size (int): Maximum width or height in pixels

    Returns:
        str: Path to the resized image (may be the same as input if no resize needed)
    """
    try:
        image = image_file
        #image = Image.open(image_file)
        # Check if resize is needed
        if max(image.width, image.height) <= max_size:
            return image  # No resize needed

        # Calculate new dimensions maintaining aspect ratio
        if image.width > image.height:
            new_width = max_size
            new_height = int((max_size * image.height) / image.width)
        else:
            new_height = max_size
            new_width = int((max_size * image.width) / image.height)

        # Resize the image using high-quality resampling
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image

    except Exception as e:
        print(f"Error resizing image: {e}")
        return Image.open(image_file)  # Return original image if resize fails

def encode_image_to_base64(image: Image.Image) -> str:
    """
    Convert image file to base64 string for API transmission.

    Args:
        image_path (str): Path to the image file

    Returns:
        str: Base64 encoded image data
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Example usage function
def main():
    """
    Example usage of the wine label extraction function
    """
    # Set your API key (or set ANTHROPIC_API_KEY environment variable)
    # api_key = "your-anthropic-api-key-here"

    # Example image path
    image_path = "/Users/mjh/code/Obispodino/cvino/raw_data/last/XWines_Slim_1K_labels/198885.jpeg"

    # Extract wine information
    result = extract_wine_info_from_image(image_path)

    # Print results
    print("Wine Information Extracted:")
    print(f"Wine Type: {result.get('wine_type')}")
    print(f"Grape Varieties: {result.get('grape_varieties')}")
    print(f"Body: {result.get('body')}")
    print(f"Acidity: {result.get('acidity')}")
    print(f"Country: {result.get('country')}")
    print(f"Region: {result.get('region')}")
    print(f"ABV: {result.get('ABV')}")
    print(f"Extraction Successful: {result.get('extraction_successful')}")

    if not result.get('extraction_successful'):
        print(f"Error: {result.get('error')}")

if __name__ == "__main__":
    main()

# # Simple usage
# image_path = "/Users/dino/code/Obispodino/cvino/raw_data/X-Wines_Official_Repository/last/XWines_Slim_1K_labels/193914.jpeg"
# wine_info = extract_wine_info_from_image(image_path)

# print(wine_info)

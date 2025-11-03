"""
Ollama Cloud Vision Client

Handles communication with Ollama Cloud API for vision model inference
using the OpenAI-compatible API format.
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI
from PIL import Image

logger = logging.getLogger("OllamaVisionClient")


class OllamaVisionClient:
    """Client for Ollama Cloud vision model API"""

    # Supported image formats
    SUPPORTED_FORMATS = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://ollama.com/v1",
        model: str = "qwen3-vl:235b-cloud",
        temperature: float = 0.2,
        max_tokens: int = 1000
    ):
        """
        Initialize Ollama Cloud vision client.

        Args:
            api_key: Ollama Cloud API key
            base_url: Ollama Cloud API base URL
            model: Vision model to use (must have -cloud suffix)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenAI client for Ollama Cloud
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        logger.info(f"Initialized Ollama Cloud client")
        logger.info(f"Base URL: {base_url}")
        logger.info(f"Model: {model}")

    def _get_mime_type(self, file_path: Path) -> str:
        """Determine MIME type from file extension"""
        ext = file_path.suffix.lower()
        return self.SUPPORTED_FORMATS.get(ext, 'image/jpeg')

    def _validate_image(self, file_path: Path) -> None:
        """Validate that the file is a supported image format"""
        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {ext}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS.keys())}"
            )

        # Try to open with PIL to verify it's a valid image
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image file: {e}")

    def _encode_image(self, file_path: Path) -> str:
        """Encode image to base64 string"""
        try:
            with open(file_path, 'rb') as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                logger.debug(f"Encoded image: {file_path.name} ({len(encoded)} bytes)")
                return encoded
        except Exception as e:
            raise IOError(f"Failed to read image file: {e}")

    def describe_image(
        self,
        image_path: str,
        prompt: Optional[str] = None
    ) -> str:
        """
        Describe an image using the vision model.

        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt. Defaults to "Describe this image in detail."

        Returns:
            Description of the image

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is unsupported or invalid
            RuntimeError: If API call fails
        """
        file_path = Path(image_path)

        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Validate image format
        self._validate_image(file_path)

        # Encode image
        base64_image = self._encode_image(file_path)
        mime_type = self._get_mime_type(file_path)

        # Default prompt
        user_prompt = prompt or "Describe this image in detail."

        logger.info(f"Analyzing image with Ollama Cloud")
        logger.info(f"Model: {self.model}")
        logger.info(f"Prompt: {user_prompt}")

        try:
            # Call Ollama Cloud API (OpenAI-compatible)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract description from response
            description = response.choices[0].message.content

            if not description:
                raise RuntimeError("Empty response from vision model")

            logger.info(f"Successfully received description ({len(description)} chars)")
            return description

        except Exception as e:
            error_msg = f"Ollama Cloud API error: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def check_connection(self) -> tuple[bool, str]:
        """
        Check if Ollama Cloud is accessible.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Try to list models as a connection test
            models = self.client.models.list()
            logger.info("Successfully connected to Ollama Cloud")
            return True, f"Connected to {self.base_url}"
        except Exception as e:
            error_msg = f"Cannot connect to Ollama Cloud: {e}"
            logger.error(error_msg)
            return False, error_msg

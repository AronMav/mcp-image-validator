#!/usr/bin/env python3
"""
MCP Image Validator Server

Provides image description capabilities using Qwen3-VL vision model
through Ollama Cloud integration.
"""

import logging
import os
from typing import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from ollama_vision_client import OllamaVisionClient

# Configure logging to stderr to avoid interfering with stdio transport
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MCP_Image_Validator")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:235b-cloud")
VISION_TEMPERATURE = float(os.getenv("VISION_TEMPERATURE", "0.2"))
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "1000"))

if not OLLAMA_API_KEY:
    logger.error("OLLAMA_API_KEY environment variable is required")
    raise ValueError("OLLAMA_API_KEY not set")

# Global Ollama client
ollama_client = OllamaVisionClient(
    api_key=OLLAMA_API_KEY,
    base_url=OLLAMA_BASE_URL,
    model=VISION_MODEL,
    temperature=VISION_TEMPERATURE,
    max_tokens=VISION_MAX_TOKENS
)

# Define application context
class AppContext:
    def __init__(self, ollama_client: OllamaVisionClient):
        self.ollama_client = ollama_client

# Lifespan management
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle"""
    logger.info("Starting MCP Image Validator server...")
    logger.info(f"Using Ollama Cloud at {OLLAMA_BASE_URL}")
    logger.info(f"Vision model: {VISION_MODEL}")
    try:
        # Startup
        yield AppContext(ollama_client=ollama_client)
    finally:
        # Shutdown
        logger.info("Shutting down MCP Image Validator server")

# Initialize FastMCP with lifespan
mcp = FastMCP("Image_Validator_MCP_Server", lifespan=app_lifespan)

# Define the image description tool
@mcp.tool()
def describe_image(
    image_path: str,
    prompt: str | None = None
) -> str:
    """
    Analyzes an image and provides a detailed description using Qwen3-VL vision model through Ollama Cloud.

    Supports common image formats (JPEG, PNG, GIF, WebP, BMP).

    Args:
        image_path: Absolute path to the image file to analyze
        prompt: Optional custom prompt for the vision model. If not provided, a default description prompt will be used.

    Returns:
        Detailed description of the image
    """
    logger.info(f"Analyzing image: {image_path}")
    if prompt:
        logger.info(f"Custom prompt: {prompt}")

    try:
        # Validate image path
        img_path = Path(image_path)
        if not img_path.exists():
            error_msg = f"Image file not found: {image_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not img_path.is_file():
            error_msg = f"Path is not a file: {image_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Generate description
        description = ollama_client.describe_image(
            image_path=str(img_path.absolute()),
            prompt=prompt
        )

        logger.info(f"Successfully described image: {img_path.name}")
        return description

    except FileNotFoundError as e:
        error_msg = f"File not found: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    except ValueError as e:
        error_msg = f"Invalid input: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Error describing image: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


# Entry point for running the server
if __name__ == "__main__":
    import asyncio

    # Run the FastMCP server
    mcp.run()

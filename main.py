import os
import uuid
import time
import json
import asyncio
import logging
import threading
import re
import base64
from typing import List, Dict, Any, Union, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import Part
from google.api_core.exceptions import GoogleAPIError
import httpx
import keyboard  # For keyboard shortcut handling

# Debug flag to toggle detailed logging
DEBUG_ENABLED = False

# Configure logging with minimal overhead
logging.basicConfig(
    level=logging.DEBUG if DEBUG_ENABLED else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="OpenAI-Compatible Gemini Server with Image Support",
    description="A FastAPI server that provides an OpenAI-compatible API interface for Google's Gemini model with API key, proxy rotation, model selection, and image processing support."
)

# File paths for persistent storage
LAST_KEY_INDEX_FILE = "last_key_index.json"
RATE_LIMITED_KEYS_FILE = "rate_limited_keys.json"

# Gemini API keys
API_KEYS = [
    "AIzaSyCT1PXjhup0VHx3Fz4AioHbVUHED0fVBP4",
    "AIzaSyArNqpA1EeeXBx-S3EVnP0tzao6r4BQnO0",
    "AIzaSyCXICPfRTnNAFwNQMmtBIb3Pi0pR4SydHg",
    "AIzaSyDiLvp7CU443luErAz3Ck0B8zFdm8UvNRs",
    "AIzaSyBzqJebfbVPcBXQy7r4Y5sVgC499uV85i0",
    "AIzaSyD6AFGKycSp1glkNEuARknMLvo93YbCqH8",
    "AIzaSyBTara5UhTbLR6qnaUI6nyV4wugycoABRM",
    "AIzaSyBI2Jc8mHJgjnXnx2udyibIZyNq8SGlLSY",
    "AIzaSyAcgdqbZsX9UOG4QieFSW7xCcwlHzDSURY",
    "AIzaSyAwOawlX-YI7_xvXY-A-3Ks3k9CxiTQfy4",
    "AIzaSyCJVUeJkqYeLNG6UsF06Gasn4mvMFfPhzw",
    "AIzaSyBFOK0YgaQOg5wilQul0P2LqHk1BgeYErw",
    "AIzaSyBQRsGHOhaiD2cNb5F68hI6BcZR7CXqmwc",
    "AIzaSyCIC16VVTlFGbiQtq7RlstTTqPYizTB7yQ",
    "AIzaSyCIlfHXQ9vannx6G9Pae0rKwWJpdstcZIM",
    "AIzaSyAUIR9gx08SNgeHq8zKAa9wyFtFu00reTM",
    "AIzaSyAST1jah1vAcnLfmofR4DDw0rjYkJXJoWg",
    "AIzaSyAV8OU1_ANXTIvkRooikeNrI1EMR3IbTyQ"
]

# Proxy configuration
PROXY_SOURCE_URL = "https://www.proxy-list.download/api/v1/get?type=https&anon=elite"
PROXY_REFRESH_INTERVAL = 3600  # Refresh proxies every hour
MAX_RETRIES = 3  # Max retries for rate limit errors
RATE_LIMIT_DURATION = 86400  # 1 day in seconds

# Supported image formats and MIME types
SUPPORTED_IMAGE_FORMATS = {
    'jpeg': 'image/jpeg',
    'jpg': 'image/jpeg',
    'png': 'image/png',
    'webp': 'image/webp',
    'gif': 'image/gif'
}

# Thread-safe API key, proxy rotation, and model management
class ProxyManager:
    def __init__(self):
        self.api_key_index = self._load_last_key_index()
        self.lock = threading.Lock()
        self.proxies = []
        self.proxy_mapping = {}
        self.last_refresh = 0
        self.rate_limited_keys = self._load_rate_limited_keys()
        self.allowed_models = set()  # Dynamically updated based on requests
        logger.warning("Using free proxies may pose security risks (e.g., data logging). Consider paid proxies for production.")

    def _load_last_key_index(self) -> int:
        """Load the last used API key index from file."""
        try:
            if os.path.exists(LAST_KEY_INDEX_FILE):
                with open(LAST_KEY_INDEX_FILE, "r") as f:
                    data = json.load(f)
                    index = data.get("index", 0)
                    logger.info("Loaded last API key index: %d", index)
                    return index
        except Exception as e:
            logger.error("Failed to load last key index: %s", e)
        return 0

    def _save_last_key_index(self):
        """Save the current API key index to file."""
        try:
            with open(LAST_KEY_INDEX_FILE, "w") as f:
                json.dump({"index": self.api_key_index}, f)
                logger.debug("Saved API key index: %d", self.api_key_index)
        except Exception as e:
            logger.error("Failed to save last key index: %s", e)

    def _load_rate_limited_keys(self) -> Dict[str, float]:
        """Load rate-limited keys and remove expired ones."""
        try:
            if os.path.exists(RATE_LIMITED_KEYS_FILE):
                with open(RATE_LIMITED_KEYS_FILE, "r") as f:
                    data = json.load(f)
                    current_time = time.time()
                    # Remove keys older than 1 day
                    filtered_data = {k: t for k, t in data.items() if current_time - t < RATE_LIMIT_DURATION}
                    if len(data) != len(filtered_data):
                        with open(RATE_LIMITED_KEYS_FILE, "w") as f:
                            json.dump(filtered_data, f)
                        logger.info("Cleaned up expired rate-limited keys")
                    return filtered_data
        except Exception as e:
            logger.error("Failed to load rate-limited keys: %s", e)
        return {}

    def _save_rate_limited_key(self, api_key: str):
        """Add an API key to the rate-limited list."""
        try:
            self.rate_limited_keys[api_key] = time.time()
            with open(RATE_LIMITED_KEYS_FILE, "w") as f:
                json.dump(self.rate_limited_keys, f)
                logger.info("Added API key %s... to rate-limited list", api_key[:10])
        except Exception as e:
            logger.error("Failed to save rate-limited key: %s", e)

    async def fetch_proxies(self) -> List[str]:
        """Fetch elite HTTPS proxies from Proxy-List.download API."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(PROXY_SOURCE_URL)
                response.raise_for_status()
                proxies = response.text.strip().split("\r\n")
                valid_proxies = []
                for proxy in proxies:
                    if self._is_valid_proxy(proxy):
                        valid_proxies.append(f"https://{proxy}")
                    else:
                        logger.debug("Skipped invalid proxy: %s", proxy)
                logger.info("Fetched %d valid proxies from Proxy-List.download", len(valid_proxies))
                return valid_proxies
        except httpx.HTTPError as e:
            logger.error("Failed to fetch proxies: %s", e)
            return []

    def _is_valid_proxy(self, proxy: str) -> bool:
        """Validate proxy format and content."""
        if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}$", proxy):
            return False
        if not all(32 <= ord(c) <= 126 for c in proxy):
            return False
        return True

    async def refresh_proxies(self):
        """Refresh proxy list if stale."""
        current_time = time.time()
        if current_time - self.last_refresh > PROXY_REFRESH_INTERVAL or not self.proxies:
            self.proxies = await self.fetch_proxies()
            self.last_refresh = current_time
            self._assign_proxies()
            logger.info("Refreshed proxies: %d available", len(self.proxies))

    def _assign_proxies(self):
        """Assign proxies to API keys."""
        self.proxy_mapping = {}
        for i, api_key in enumerate(API_KEYS):
            proxy = self.proxies[i % len(self.proxies)] if self.proxies else None
            self.proxy_mapping[api_key] = proxy
            logger.debug("Assigned proxy %s to API key %s...", proxy, api_key[:10])

    def get_next_api_key_and_proxy(self) -> Tuple[str, Optional[str]]:
        """Get the next API key and its assigned proxy thread-safely."""
        with self.lock:
            for _ in range(len(API_KEYS)):
                api_key = API_KEYS[self.api_key_index]
                self.api_key_index = (self.api_key_index + 1) % len(API_KEYS)
                if api_key not in self.rate_limited_keys:
                    proxy = self.proxy_mapping.get(api_key)
                    self._save_last_key_index()
                    return api_key, proxy
                logger.debug("Skipped rate-limited API key %s...", api_key[:10])
            logger.error("No available API keys (all rate-limited)")
            raise HTTPException(status_code=429, detail="All API keys are rate-limited")

# Initialize proxy manager
proxy_manager = ProxyManager()

# Enhanced Pydantic models for request validation with image support
class ImageUrl(BaseModel):
    """Model for image URL with optional detail parameter."""
    url: str
    detail: Optional[str] = "auto"

class ContentItem(BaseModel):
    """Model for content items that can be text or image."""
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    """Model for a single chat message with image support."""
    role: str
    content: Union[str, List[ContentItem]]

class ChatCompletionRequest(BaseModel):
    """Model for chat completion request payload with image support."""
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False

    class Config:
        validate_assignment = True
        use_enum_values = True

def extract_image_data(data_url: str) -> Tuple[bytes, str]:
    """Extract image data and format from data URL."""
    if not data_url.startswith("data:"):
        raise ValueError("Invalid data URL format")
    
    try:
        # Parse data URL: data:image/jpeg;base64,<base64_data>
        header, base64_data = data_url.split(",", 1)
        mime_type = header.split(";")[0].split(":")[1]
        
        # Extract format from MIME type
        image_format = mime_type.split("/")[1].lower()
        if image_format not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format: {image_format}")
        
        # Decode base64 data
        image_bytes = base64.b64decode(base64_data)
        logger.debug("Successfully extracted image data: format=%s, size=%d bytes", image_format, len(image_bytes))
        
        return image_bytes, mime_type
    except Exception as e:
        logger.error("Failed to extract image data from URL: %s", e)
        raise ValueError(f"Invalid image data URL: {str(e)}")

def process_content(content: Union[str, List[ContentItem]]) -> List[Dict[str, Any]]:
    """Process message content into Gemini API format with image support."""
    if isinstance(content, str):
        return [{"text": content}]
    
    parts = []
    for item in content:
        if item.type == "text" and item.text:
            parts.append({"text": item.text})
        elif item.type == "image_url" and item.image_url:
            try:
                image_bytes, mime_type = extract_image_data(item.image_url.url)
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(image_bytes).decode('utf-8')
                    }
                })
                logger.debug("Added image part with MIME type: %s", mime_type)
            except Exception as e:
                logger.error("Failed to process image: %s", e)
                # Add error message as text instead of failing completely
                parts.append({"text": f"[Image processing error: {str(e)}]"})
    
    return parts if parts else [{"text": ""}]

def openai_to_gemini_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style messages to Gemini API format with image support."""
    gemini_messages = []
    for msg in messages:
        role = "user" if msg.role == "user" else "model"
        parts = process_content(msg.content)
        gemini_messages.append({"role": role, "parts": parts})
        
        if DEBUG_ENABLED:
            content_preview = str(parts)[:100] if len(str(parts)) > 100 else str(parts)
            logger.debug("Processed message: role=%s, parts=%s...", role, content_preview)
    
    return gemini_messages

def gemini_to_openai_response(gemini_response: Any, model: str) -> Dict[str, Any]:
    """Convert Gemini response to OpenAI-compatible non-streaming format."""
    content = getattr(gemini_response, "text", "The task is unclear. Please provide more details.")
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

async def stream_openai_response(gemini_stream: Any, model: str):
    """Stream Gemini response in OpenAI-compatible chunked format."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    try:
        for response_chunk in gemini_stream:
            content = getattr(response_chunk, "text", "")
            if content:
                yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"
                if DEBUG_ENABLED:
                    logger.debug("Streaming chunk: content=%s", content[:100])
            await asyncio.sleep(0)
    except Exception as e:
        if DEBUG_ENABLED:
            logger.error("Streaming error: %s", str(e))
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': f'Streaming error: {str(e)}'}, 'finish_reason': 'error'}]})}\n\n"

    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"

async def make_gemini_request(api_key: str, proxy: Optional[str], messages: List[Dict[str, Any]], generation_config: Dict[str, Any], stream: bool, model: str) -> Any:
    """Make a request to the Gemini API with the specified API key, proxy, and model with image support."""
    genai.configure(api_key=api_key)
    
    gemini_model = genai.GenerativeModel(model)

    if proxy:
        logger.debug("Proxy support is limited with Google's genai library for API key %s...", api_key[:10])
    else:
        logger.debug("No proxy used for API key %s...", api_key[:10])

    try:
        if stream:
            return gemini_model.generate_content(contents=messages, generation_config=generation_config, stream=True)
        else:
            return gemini_model.generate_content(contents=messages, generation_config=generation_config)
    except Exception as e:
        logger.error("Error making Gemini request: %s", str(e))
        raise

def start_keyboard_listener():
    """Start listening for Alt+number shortcuts to switch models."""
    def on_key_event(e):
        if e.event_type == keyboard.KEY_DOWN and e.name in ["1", "2", "3"] and keyboard.is_pressed("alt"):
            index = int(e.name) - 1
            models = list(proxy_manager.allowed_models)
            if index < len(models):
                logger.info("Keyboard shortcut: Available model %d: %s", index + 1, models[index])

    try:
        keyboard.hook(on_key_event)
        logger.info("Started keyboard listener for model information (Alt+1, Alt+2, etc.)")
    except Exception as e:
        logger.warning("Could not start keyboard listener: %s", e)

@app.get("/v1/models")
async def list_models():
    """List available models in OpenAI format."""
    models = []
    for model in proxy_manager.allowed_models:
        models.append({
            "id": model,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google"
        })
    
    return {
        "object": "list",
        "data": models
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion requests with API key, proxy rotation, model selection, and image support."""
    try:
        raw_body = await request.json()
        if DEBUG_ENABLED:
            logger.debug("Received request: %s", json.dumps(raw_body, default=str)[:500])
    except ValueError:
        logger.error("Invalid JSON payload")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    try:
        validated_request = ChatCompletionRequest.model_validate(raw_body)
    except ValidationError as e:
        logger.error("Validation error: %s", e.errors())
        raise HTTPException(status_code=422, detail={"error": "Validation failed", "details": e.errors()})

    # Use the model from the request and add it to allowed_models
    requested_model = validated_request.model
    proxy_manager.allowed_models.add(requested_model)  # Dynamically add model to allowed_models
    logger.debug("Added model %s to allowed_models", requested_model)

    await proxy_manager.refresh_proxies()

    # Check if any message contains images
    has_images = False
    for message in validated_request.messages:
        if isinstance(message.content, list):
            for item in message.content:
                if hasattr(item, 'type') and item.type == "image_url":
                    has_images = True
                    break
    
    if has_images:
        logger.info("Processing request with images using model: %s", requested_model)

    gemini_messages = openai_to_gemini_messages(validated_request.messages)
    generation_config = {
        "temperature": validated_request.temperature,
        "max_output_tokens": validated_request.max_tokens,
    }

    for attempt in range(MAX_RETRIES):
        try:
            api_key, proxy = proxy_manager.get_next_api_key_and_proxy()
            response = await make_gemini_request(api_key, proxy, gemini_messages, generation_config, validated_request.stream, requested_model)
            
            if validated_request.stream:
                return StreamingResponse(
                    stream_openai_response(response, requested_model),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache"} # Removed "Connection": "keep-alive" to force immediate connection closure
                )
            return gemini_to_openai_response(response, requested_model)
            
        except GoogleAPIError as e:
            logger.error("Gemini API error with API key %s... and proxy %s: %s", api_key[:10], proxy, str(e))
            if "API key not valid" in str(e):
                raise HTTPException(status_code=401, detail="Invalid Gemini API key")
            if "Quota exceeded" in str(e) or "Resource has been exhausted" in str(e):
                logger.warning("Rate limit hit for API key %s..., adding to rate-limited list", api_key[:10])
                proxy_manager._save_rate_limited_key(api_key)
                if attempt < MAX_RETRIES - 1:
                    logger.warning("Retrying with next API key and proxy (%d/%d)", attempt + 1, MAX_RETRIES)
                    continue
                raise HTTPException(status_code=429, detail="All API keys hit rate limits")
            raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")
        except Exception as e:
            logger.error("Request error with API key %s... and proxy %s: %s", api_key[:10], proxy, str(e))
            if attempt < MAX_RETRIES - 1:
                logger.warning("Retrying with next API key and proxy (%d/%d)", attempt + 1, MAX_RETRIES)
                continue
            raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

    raise HTTPException(status_code=429, detail="All API keys hit rate limits")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "available_models": list(proxy_manager.allowed_models),
        "image_support": True,
        "supported_formats": list(SUPPORTED_IMAGE_FORMATS.keys())
    }

if __name__ == "__main__":
    # Start keyboard listener in a separate thread
    keyboard_thread = threading.Thread(target=start_keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import uuid
import time
import json
import asyncio
import base64
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
import httpx # Keep httpx for now, might be needed for other things or remove later

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Gemini Backend for Vercel")

# API Keys - production'da environment variable kullanın
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

# Simple state management (in-memory)
current_key_index = 0
key_usage = {key: 0 for key in API_KEYS}

# Pydantic Models
class ImageUrl(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False

def get_next_key():
    """Simple round-robin key selection"""
    global current_key_index
    key = API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    key_usage[key] = key_usage.get(key, 0) + 1
    return key

def process_content(content):
    """Convert OpenAI format to Gemini format"""
    if isinstance(content, str):
        return [{"text": content}]
    
    parts = []
    for item in content:
        if item.type == "text" and item.text:
            parts.append({"text": item.text})
        elif item.type == "image_url" and item.image_url:
            try:
                # Handle base64 images
                if item.image_url.url.startswith("data:"):
                    header, base64_data = item.image_url.url.split(",", 1)
                    mime_type = header.split(";")[0].split(":")[1]
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64_data
                        }
                    })
            except:
                parts.append({"text": "[Image processing error]"})
    
    return parts or [{"text": ""}]

def convert_messages(messages):
    """Convert OpenAI messages to Gemini format"""
    return [
        {
            "role": "user" if msg.role == "user" else "model",
            "parts": process_content(msg.content)
        }
        for msg in messages
    ]

async def stream_openai_response(gemini_stream: Any, model: str):
    """Stream Gemini response in OpenAI-compatible chunked format."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # Initial chunk - role başlangıcı
    initial_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    try:
        yield f"data: {json.dumps(initial_chunk)}\n\n"
    except BrokenPipeError:
        logger.warning("Client disconnected during initial chunk (BrokenPipeError).")
        return

    try:
        for response_chunk in gemini_stream:
            content = getattr(response_chunk, "text", "")
            if content:
                content_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(content_chunk)}\n\n"
            
            # Check for finish_reason from Gemini's response_chunk (if available)
            # The genai library handles finish reasons internally, and the stream
            # will naturally end. We just need to send the final stop chunk.


    except Exception as e:
        logger.error("Streaming error: %s", str(e))
        error_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": f"Streaming error: {str(e)}"},
                "finish_reason": "error"
            }]
        }
        try:
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except BrokenPipeError:
            logger.warning("Client disconnected during error handling (BrokenPipeError).")
        return

    # Final chunk - streaming tamamlandı sinyali (empty delta, finish_reason "stop")
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {}, # Empty delta
            "finish_reason": "stop"
        }]
    }
    try:
        yield f"data: {json.dumps(final_chunk)}\n\n"
    except BrokenPipeError:
        logger.warning("Client disconnected during final chunk (BrokenPipeError).")
        return


async def make_gemini_request(api_key: str, model: str, messages: list, generation_config: dict, stream: bool = False) -> Any:
    """Make a request to the Gemini API with the specified API key and model."""
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)

    try:
        return gemini_model.generate_content(contents=messages, generation_config=generation_config, stream=stream)
    except Exception as e:
        logger.error("Error making Gemini request: %s", str(e))
        raise

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest): # Keep ChatRequest for now, will refactor to use Request later if needed
    try:
        # Convert messages
        gemini_messages = convert_messages(request.messages)
        generation_config = {
            "temperature": request.temperature
        }
        
        # Only add maxOutputTokens if it's provided and not None
        if request.max_tokens is not None:
            generation_config["max_output_tokens"] = min(request.max_tokens, 8192)
        
        # API key seç
        api_key = get_next_key()
        
        # Gemini API çağrısı
        response = await make_gemini_request(
            api_key,
            request.model,
            gemini_messages,
            generation_config,
            stream=request.stream # Pass stream parameter
        )
        
        if request.stream:
            return StreamingResponse(
                stream_openai_response(response, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Content-Type": "text/event-stream",
                }
            )

        # Regular response (non-streaming)
        text = ""
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text = "".join(part.text for part in candidate.content.parts if part.text)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(str(request.messages)), # This will be inaccurate for genai, but keeping for now
                "completion_tokens": len(text),
                "total_tokens": len(str(request.messages)) + len(text)
            }
        }
        
    except HTTPException:
        raise
    except GoogleAPIError as e:
        logger.error("Gemini API error: %s", str(e))
        if "Quota exceeded" in str(e) or "Resource has been exhausted" in str(e):
            # For now, just raise 429. No retry logic from the example yet.
            raise HTTPException(status_code=429, detail="Rate limit exceeded for Gemini API")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")
    except Exception as e:
        logger.error("Internal error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "total_requests": sum(key_usage.values()),
        "active_keys": len([k for k, v in key_usage.items() if v > 0])
    }

@app.get("/")
async def root():
    return {
        "message": "Gemini Proxy API",
        "version": "1.0.0",
        "endpoints": ["/v1/chat/completions", "/v1/models", "/health"]
    }

# CORS Middleware
@app.middleware("http")
async def cors_handler(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    response.headers["Access-Control-Max-Age"] = "86400"
    
    # OPTIONS request handling
    if request.method == "OPTIONS":
        response.status_code = 200
    
    return response

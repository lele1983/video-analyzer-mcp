#!/usr/bin/env python3
"""
MCP Server for Social Video Analysis.

Downloads videos from Instagram and TikTok via Apify, 
analyzes them with Google Gemini Flash API, and returns structured insights.
"""

import os
import json
import base64
import tempfile
import asyncio
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, field_validator, ConfigDict
from mcp.server.fastmcp import FastMCP
import sys
from functools import wraps

# Initialize the MCP server
mcp = FastMCP("video_analyzer_mcp")

# Constants
APIFY_API_BASE = "https://api.apify.com/v2"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# Environment variables (loaded at runtime)
def get_apify_token() -> str:
    token = os.environ.get("APIFY_API_TOKEN")
    if not token:
        raise ValueError("APIFY_API_TOKEN environment variable is required")
    return token

def get_gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    return key

def get_openrouter_key() -> Optional[str]:
    return os.environ.get("OPENROUTER_API_KEY")


# ============================================================================
# Enums and Models
# ============================================================================

class Platform(str, Enum):
    """Supported social media platforms."""
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"

class AnalysisType(str, Enum):
    """Type of video analysis to perform."""
    FULL = "full"           # Complete analysis
    SUMMARY = "summary"     # Quick summary only
    TRANSCRIPT = "transcript"  # Speech-to-text focus
    CONTENT = "content"     # Visual content description
    SENTIMENT = "sentiment" # Sentiment and tone analysis

class ResponseFormat(str, Enum):
    """Output format for responses."""
    MARKDOWN = "markdown"
    JSON = "json"

class GeminiModel(str, Enum):
    """Available Gemini models for video analysis."""
    FLASH_2_0 = "google/gemini-2.0-flash-001"
    FLASH_2_5_LITE = "google/gemini-2.5-flash-lite"
    FLASH_2_5 = "google/gemini-2.5-flash"
    PRO_2_5 = "google/gemini-2.5-pro"
    FLASH_3_PREVIEW = "google/gemini-3-flash-preview"
    PRO_3_PREVIEW = "google/gemini-3-pro-preview"




# ============================================================================
# Input Models
# ============================================================================

class DownloadVideoInput(BaseModel):
    """Input for downloading a video from social media."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')
    
    url: str = Field(
        ..., 
        description="Full URL of the Instagram Reel/Post or TikTok video (e.g., 'https://www.instagram.com/reel/ABC123/' or 'https://www.tiktok.com/@user/video/123456')",
        min_length=10,
        max_length=500
    )
    platform: Optional[Platform] = Field(
        default=None,
        description="Platform type (auto-detected if not provided): 'instagram' or 'tiktok'"
    )
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return v


class AnalyzeVideoInput(BaseModel):
    """Input for analyzing a video with Gemini."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')
    
    video_url: str = Field(
        ...,
        description="Direct URL to the video file (mp4/webm) or local file path",
        min_length=5
    )
    analysis_type: AnalysisType = Field(
        default=AnalysisType.FULL,
        description="Type of analysis: 'full' (complete), 'summary' (quick), 'transcript' (speech), 'content' (visual), 'sentiment' (tone)"
    )
    model: GeminiModel = Field(
        default=GeminiModel.FLASH_2_0,
        description="Gemini model to use."
    )
    custom_prompt: Optional[str] = Field(
        default=None,
        description="Custom analysis prompt to append (optional)",
        max_length=1000
    )
    language: str = Field(
        default="italiano",
        description="Language for the analysis response (e.g., 'italiano', 'english', 'español')"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for structured"
    )


class DownloadAndAnalyzeInput(BaseModel):
    """Input for the combined download and analyze workflow."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')
    
    url: str = Field(
        ...,
        description="Full URL of the Instagram or TikTok video",
        min_length=10,
        max_length=500
    )
    analysis_type: AnalysisType = Field(
        default=AnalysisType.FULL,
        description="Type of analysis to perform"
    )
    model: GeminiModel = Field(
        default=GeminiModel.FLASH_2_0,
        description="Gemini model to use."
    )
    custom_prompt: Optional[str] = Field(
        default=None,
        description="Custom analysis prompt (optional)",
        max_length=1000
    )
    language: str = Field(
        default="italiano",
        description="Language for analysis response"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return v


class BatchAnalyzeInput(BaseModel):
    """Input for analyzing multiple videos in parallel."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')
    
    urls: List[str] = Field(
        ...,
        description="List of Instagram or TikTok video URLs to analyze",
        min_length=1,
        max_length=10
    )
    analysis_type: AnalysisType = Field(
        default=AnalysisType.FULL,
        description="Type of analysis to perform on all videos"
    )
    model: GeminiModel = Field(
        default=GeminiModel.FLASH_2_0,
        description="Gemini model to use"
    )
    custom_prompt: Optional[str] = Field(
        default=None,
        description="Custom analysis prompt (applied to all videos)",
        max_length=1000
    )
    language: str = Field(
        default="italiano",
        description="Language for analysis response"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )
    
    @field_validator('urls')
    @classmethod
    def validate_urls(cls, v: List[str]) -> List[str]:
        if len(v) > 20:
             raise ValueError("Max 20 videos per batch allowed")
        for url in v:
            if not url.strip().startswith(('http://', 'https://')):
                raise ValueError(f"Invalid URL: {url}")
        return v


# ============================================================================
# Utility Functions
# ============================================================================

def with_retry(retries: int = 3, delay: int = 10):
    """Decorator to retry async functions with delay."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < retries:
                        print(f"[RETRY] Error in {func.__name__}: {e}. Retrying in {delay}s... (Attempt {attempt + 1}/{retries})", file=sys.stderr)
                        await asyncio.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


def detect_platform(url: str) -> Platform:
    """Auto-detect platform from URL."""
    url_lower = url.lower()
    if 'instagram.com' in url_lower or 'instagr.am' in url_lower:
        return Platform.INSTAGRAM
    elif 'tiktok.com' in url_lower or 'vm.tiktok.com' in url_lower:
        return Platform.TIKTOK
    else:
        raise ValueError(f"Could not detect platform from URL. Supported: Instagram, TikTok")


def get_analysis_prompt(analysis_type: AnalysisType, language: str, custom_prompt: Optional[str] = None) -> str:
    """Generate the analysis prompt based on type."""
    
    base_prompts = {
        AnalysisType.FULL: f"""Analizza questo video in modo completo. Rispondi in {language}.

Fornisci:
1. **Riassunto**: Descrizione breve del contenuto (2-3 frasi)
2. **Contenuto Visivo**: Cosa si vede nel video (ambientazione, persone, azioni, testo sovrimpresso)
3. **Audio/Parlato**: Trascrizione o descrizione di cosa viene detto/sentito
4. **Tono e Sentiment**: Analisi del tono (serio, comico, informativo, emotivo, etc.) e sentiment generale
5. **Target Audience**: A chi sembra rivolto questo contenuto
6. **Elementi Chiave**: Hashtag visibili, menzioni, call-to-action, prodotti mostrati
7. **Potenziale Virale**: Valutazione del potenziale di engagement (1-10) con motivazione""",

        AnalysisType.SUMMARY: f"""Fornisci un riassunto conciso di questo video in {language}.
Descrivi in 3-5 frasi: cosa succede, chi è coinvolto, e qual è il messaggio principale.""",

        AnalysisType.TRANSCRIPT: f"""Trascrivi tutto il parlato e il testo visibile in questo video.
Rispondi in {language}.
Includi:
- Dialoghi e monologhi
- Testo sovrimpresso / sottotitoli
- Testo visibile su cartelli, prodotti, etc.""",

        AnalysisType.CONTENT: f"""Descrivi dettagliatamente il contenuto visivo di questo video in {language}.
Includi:
- Ambientazione e location
- Persone presenti (aspetto, abbigliamento, espressioni)
- Azioni e movimenti
- Oggetti e prodotti visibili
- Transizioni e effetti video
- Colori e stile visivo""",

        AnalysisType.SENTIMENT: f"""Analizza il tono, il sentiment e l'impatto emotivo di questo video.
Rispondi in {language}.
Valuta:
- Tono generale (serio, umoristico, inspirazionale, etc.)
- Emozioni evocate (gioia, tristezza, sorpresa, rabbia, etc.)
- Sentiment complessivo (positivo, negativo, neutro) con score 1-10
- Possibili reazioni del pubblico
- Elementi che generano engagement emotivo"""
    }
    
    prompt = base_prompts.get(analysis_type, base_prompts[AnalysisType.FULL])
    
    if custom_prompt:
        prompt += f"\n\nIstruzioni aggiuntive: {custom_prompt}"
    
    return prompt


@with_retry(retries=3, delay=10)
async def analyze_video_with_openrouter(
    video_data: bytes,
    prompt: str,
    model: str = "google/gemini-2.0-flash-001",  # OpenRouter model ID
    mime_type: str = "video/mp4"
) -> str:
    """Send video to OpenRouter for analysis."""
    api_key = get_openrouter_key()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found")

    # Encode video as base64
    video_b64 = base64.standard_b64encode(video_data).decode('utf-8')
    data_url = f"data:{mime_type};base64,{video_b64}"

    request_body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",  # OpenRouter uses image_url type for video data URLs often, or content parts
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://video-analyzer-mcp.local",
        "X-Title": "Video Analyzer MCP",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{OPENROUTER_API_BASE}/chat/completions",
            json=request_body,
            headers=headers
        )
        
        if response.status_code != 200:
            error_text = response.text
            try:
                error_json = response.json()
                if "error" in error_json:
                    error_text = json.dumps(error_json["error"])
            except:
                pass
            raise Exception(f"OpenRouter API Error ({response.status_code}): {error_text}")

        result = response.json()
        choices = result.get("choices", [])
        if not choices:
            raise Exception("No response choices from OpenRouter")
            
        return choices[0].get("message", {}).get("content", "")


@with_retry(retries=3, delay=10)
async def _apify_run_actor(actor_id: str, input_data: dict, timeout: int = 120) -> dict:
    """Run an Apify actor and wait for results."""
    token = get_apify_token()
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Start the actor run
        run_response = await client.post(
            f"{APIFY_API_BASE}/acts/{actor_id}/runs",
            params={"token": token},
            json=input_data,
            headers={"Content-Type": "application/json"}
        )
        run_response.raise_for_status()
        run_data = run_response.json()
        run_id = run_data["data"]["id"]
        
        # Poll for completion
        for _ in range(60):  # Max 60 attempts (2 minutes with 2s interval)
            await asyncio.sleep(2)
            
            status_response = await client.get(
                f"{APIFY_API_BASE}/actor-runs/{run_id}",
                params={"token": token}
            )
            status_response.raise_for_status()
            status_data = status_response.json()
            
            status = status_data["data"]["status"]
            if status == "SUCCEEDED":
                # Get dataset items
                dataset_id = status_data["data"]["defaultDatasetId"]
                items_response = await client.get(
                    f"{APIFY_API_BASE}/datasets/{dataset_id}/items",
                    params={"token": token, "format": "json"}
                )
                items_response.raise_for_status()
                return items_response.json()
            
            elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
                raise Exception(f"Apify actor run failed with status: {status}")
        
        raise Exception("Apify actor run timed out")


@with_retry(retries=3, delay=10)
async def download_video_from_url(video_url: str) -> bytes:
    """Download video content from a direct URL."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        response = await client.get(video_url)
        response.raise_for_status()
        return response.content


@with_retry(retries=3, delay=10)
async def analyze_video_with_gemini(
    video_data: bytes,
    prompt: str,
    model: str = "gemini-2.0-flash",
    mime_type: str = "video/mp4"
) -> str:
    """Send video to Gemini for analysis."""
    api_key = get_gemini_key()
    
    # Encode video as base64
    video_b64 = base64.standard_b64encode(video_data).decode('utf-8')
    
    # Prepare request for Gemini
    request_body = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": video_b64
                    }
                },
                {
                    "text": prompt
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.4,
            "topK": 32,
            "topP": 1,
            "maxOutputTokens": 4096
        }
    }
    
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{GEMINI_API_BASE}/models/{model}:generateContent",
            params={"key": api_key},
            json=request_body,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract text from response
        candidates = result.get("candidates", [])
        if not candidates:
            raise Exception("No response from Gemini")
        
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            raise Exception("Empty response from Gemini")
        
        return parts[0].get("text", "")


def _handle_error(e: Exception, context: str = "") -> str:
    """Format errors consistently."""
    prefix = f"[{context}] " if context else ""
    
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status == 401:
            return f"{prefix}Error: Authentication failed. Check your API keys."
        elif status == 403:
            return f"{prefix}Error: Access forbidden. Check API permissions."
        elif status == 404:
            return f"{prefix}Error: Resource not found. The video may have been deleted."
        elif status == 429:
            return f"{prefix}Error: Rate limit exceeded. Please wait before retrying."
        return f"{prefix}Error: API request failed (HTTP {status})"
    
    elif isinstance(e, httpx.TimeoutException):
        return f"{prefix}Error: Request timed out. The video may be too large or the service is slow."
    
    elif isinstance(e, ValueError):
        return f"{prefix}Error: {str(e)}"
    
    return f"{prefix}Error: {type(e).__name__} - {str(e)}"


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool(
    name="video_download_instagram",
    annotations={
        "title": "Download Instagram Video",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def video_download_instagram(params: DownloadVideoInput) -> str:
    """Download a video from Instagram (Reel, Post, or Story) using Apify.
    
    This tool extracts video metadata and download URLs from Instagram posts.
    It uses the Apify Instagram Scraper to fetch video information.
    
    Args:
        params (DownloadVideoInput): Input containing:
            - url (str): Full Instagram URL (e.g., 'https://www.instagram.com/reel/ABC123/')
            - platform (Optional[Platform]): Will be set to 'instagram'
    
    Returns:
        str: JSON with video metadata including:
            - video_url: Direct URL to download the video
            - thumbnail_url: Video thumbnail
            - caption: Post caption
            - author: Username of the creator
            - likes: Like count
            - comments: Comment count
            - duration: Video duration in seconds
    
    Example:
        Use when: "Scarica il video da questo reel Instagram"
        URL format: https://www.instagram.com/reel/ABC123/ or https://www.instagram.com/p/XYZ789/
    """
    try:
        # Validate it's an Instagram URL
        if 'instagram.com' not in params.url.lower() and 'instagr.am' not in params.url.lower():
            return "Error: This tool only works with Instagram URLs. Use video_download_tiktok for TikTok."
        
        # Apify Instagram Scraper actor
        actor_id = "apify~instagram-scraper"
        
        input_data = {
            "directUrls": [params.url],
            "resultsType": "posts",
            "resultsLimit": 1,
            "searchType": "hashtag",
            "searchLimit": 1
        }
        
        results = await _apify_run_actor(actor_id, input_data)
        
        if not results:
            return "Error: No video found at this URL. The post may be private or deleted."
        
        post = results[0]
        
        # Extract video URL
        video_url = post.get("videoUrl") or post.get("displayUrl")
        if not video_url:
            return "Error: Could not extract video URL. This might be an image post, not a video."
        
        response = {
            "status": "success",
            "platform": "instagram",
            "video_url": video_url,
            "thumbnail_url": post.get("displayUrl"),
            "caption": post.get("caption", "")[:500],  # Truncate long captions
            "author": post.get("ownerUsername"),
            "author_id": post.get("ownerId"),
            "likes": post.get("likesCount", 0),
            "comments": post.get("commentsCount", 0),
            "views": post.get("videoViewCount", 0),
            "duration": post.get("videoDuration"),
            "timestamp": post.get("timestamp"),
            "post_url": params.url
        }
        
        return json.dumps(response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        return _handle_error(e, "Instagram Download")


@mcp.tool(
    name="video_download_tiktok",
    annotations={
        "title": "Download TikTok Video",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def video_download_tiktok(params: DownloadVideoInput) -> str:
    """Download a video from TikTok using Apify.
    
    This tool extracts video metadata and download URLs from TikTok posts.
    It uses the Apify TikTok Scraper to fetch video information.
    
    Args:
        params (DownloadVideoInput): Input containing:
            - url (str): Full TikTok URL (e.g., 'https://www.tiktok.com/@user/video/123456')
            - platform (Optional[Platform]): Will be set to 'tiktok'
    
    Returns:
        str: JSON with video metadata including:
            - video_url: Direct URL to download the video (no watermark)
            - thumbnail_url: Video cover image
            - caption: Post description
            - author: Username of the creator
            - likes, comments, shares, plays: Engagement metrics
            - music: Audio track information
    
    Example:
        Use when: "Scarica questo video TikTok"
        URL formats: 
            - https://www.tiktok.com/@user/video/123456
            - https://vm.tiktok.com/ABC123/
    """
    try:
        # Validate it's a TikTok URL
        if 'tiktok.com' not in params.url.lower():
            return "Error: This tool only works with TikTok URLs. Use video_download_instagram for Instagram."
        
        # Apify TikTok Scraper actor
        actor_id = "clockworks~tiktok-scraper"
        
        input_data = {
            "postURLs": [params.url],
            "resultsPerPage": 1,
            "shouldDownloadVideos": True,
            "shouldDownloadCovers": False
        }
        
        results = await _apify_run_actor(actor_id, input_data)
        
        if not results:
            return "Error: No video found at this URL. The video may be private or deleted."
        
        post = results[0]
        
        # Extract video URL (try multiple fields)
        video_url = (
            (post.get("mediaUrls") and post.get("mediaUrls")[0]) or
            post.get("videoUrlNoWaterMark") or 
            post.get("videoUrl") or 
            post.get("video", {}).get("downloadAddr")
        )
        
        if not video_url:
            return "Error: Could not extract video URL from TikTok response."
        
        # Extract music info
        music_info = post.get("musicMeta") or post.get("music", {})
        
        response = {
            "status": "success",
            "platform": "tiktok",
            "video_url": video_url,
            "thumbnail_url": post.get("coverUrl") or post.get("video", {}).get("cover"),
            "caption": post.get("text", "")[:500],
            "author": post.get("authorMeta", {}).get("name") or post.get("author", {}).get("uniqueId"),
            "author_nickname": post.get("authorMeta", {}).get("nickName") or post.get("author", {}).get("nickname"),
            "author_id": post.get("authorMeta", {}).get("id") or post.get("author", {}).get("id"),
            "likes": post.get("diggCount") or post.get("stats", {}).get("diggCount", 0),
            "comments": post.get("commentCount") or post.get("stats", {}).get("commentCount", 0),
            "shares": post.get("shareCount") or post.get("stats", {}).get("shareCount", 0),
            "plays": post.get("playCount") or post.get("stats", {}).get("playCount", 0),
            "duration": post.get("videoMeta", {}).get("duration") or post.get("video", {}).get("duration"),
            "music": {
                "title": music_info.get("musicName") or music_info.get("title"),
                "author": music_info.get("musicAuthor") or music_info.get("authorName"),
                "original": music_info.get("musicOriginal", False)
            },
            "hashtags": post.get("hashtags", []),
            "post_url": params.url
        }
        
        return json.dumps(response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        return _handle_error(e, "TikTok Download")


@mcp.tool(
    name="video_analyze",
    annotations={
        "title": "Analyze Video with AI",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def video_analyze(params: AnalyzeVideoInput) -> str:
    """Analyze a video using Google Gemini Flash AI.
    
    This tool downloads a video from a URL and sends it to Gemini for analysis.
    Supports various analysis types: full analysis, summary, transcript, 
    content description, or sentiment analysis.
    
    Args:
        params (AnalyzeVideoInput): Input containing:
            - video_url (str): Direct URL to video file (mp4/webm)
            - analysis_type (AnalysisType): Type of analysis ('full', 'summary', 'transcript', 'content', 'sentiment')
            - custom_prompt (Optional[str]): Additional instructions for analysis
            - language (str): Response language (default: 'italiano')
            - response_format (ResponseFormat): Output format ('markdown' or 'json')
    
    Returns:
        str: AI-generated analysis of the video content
    
    Example:
        Use after downloading a video with video_download_instagram or video_download_tiktok.
        Pass the 'video_url' from the download response to this tool.
    """
    try:
        # Download video
        video_data = await download_video_from_url(params.video_url)
        
        if len(video_data) > 50 * 1024 * 1024:  # 50MB limit
            return "Error: Video file is too large (>50MB). Gemini has size limits."
        
        if len(video_data) < 1000:
            return "Error: Video file appears to be invalid or too small."
        
        # Generate prompt
        prompt = get_analysis_prompt(params.analysis_type, params.language, params.custom_prompt)
        
        # Add JSON format instruction if needed
        if params.response_format == ResponseFormat.JSON:
            prompt += "\n\nIMPORTANTE: Rispondi SOLO con un oggetto JSON valido, senza markdown o testo aggiuntivo."
        
        # Analyze with OpenRouter or Gemini
        openrouter_key = get_openrouter_key()
        
        if openrouter_key:
            # Map Gemini model enum to OpenRouter IDs
            # Since Enum values are now the full OpenRouter IDs, we can use them directly
            or_model = params.model.value
            analysis = await analyze_video_with_openrouter(video_data, prompt, model=or_model)
        else:
            # Fallback to direct Gemini API
            analysis = await analyze_video_with_gemini(video_data, prompt, model=params.model.value)
        
        return analysis
        
    except Exception as e:
        return _handle_error(e, "Video Analysis")


@mcp.tool(
    name="video_download_and_analyze",
    annotations={
        "title": "Download and Analyze Social Video",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def video_download_and_analyze(params: DownloadAndAnalyzeInput) -> str:
    """Complete workflow: Download video from Instagram/TikTok and analyze with AI.
    
    This is a convenience tool that combines downloading and analysis in one call.
    It auto-detects the platform, downloads the video via Apify, and analyzes it
    with Google Gemini Flash.
    
    Args:
        params (DownloadAndAnalyzeInput): Input containing:
            - url (str): Instagram or TikTok video URL
            - analysis_type (AnalysisType): Type of analysis ('full', 'summary', etc.)
            - custom_prompt (Optional[str]): Additional analysis instructions
            - language (str): Response language (default: 'italiano')
            - response_format (ResponseFormat): Output format
    
    Returns:
        str: Combined metadata and AI analysis in markdown or JSON format
    
    Example:
        Use when: "Analizza questo video Instagram/TikTok"
        Single call does: detect platform -> download video -> AI analysis
    """
    try:
        # Detect platform
        platform = detect_platform(params.url)
        
        # Download based on platform
        if platform == Platform.INSTAGRAM:
            download_input = DownloadVideoInput(url=params.url, platform=Platform.INSTAGRAM)
            download_result = await video_download_instagram(download_input)
        else:
            download_input = DownloadVideoInput(url=params.url, platform=Platform.TIKTOK)
            download_result = await video_download_tiktok(download_input)
        
        # Parse download result
        try:
            download_data = json.loads(download_result)
        except json.JSONDecodeError:
            return f"Error during download: {download_result}"
        
        if download_data.get("status") != "success":
            return f"Download failed: {download_result}"
        
        video_url = download_data.get("video_url")
        if not video_url:
            return "Error: Could not extract video URL from download response."
        
        # Analyze video
        analyze_input = AnalyzeVideoInput(
            video_url=video_url,
            analysis_type=params.analysis_type,
            model=params.model,
            custom_prompt=params.custom_prompt,
            language=params.language,
            response_format=params.response_format
        )
        analysis_result = await video_analyze(analyze_input)
        
        # Combine results
        if params.response_format == ResponseFormat.JSON:
            combined = {
                "metadata": download_data,
                "analysis": analysis_result
            }
            return json.dumps(combined, indent=2, ensure_ascii=False)
        else:
            # Markdown format
            output_lines = [
                f"# Video Analysis: {platform.value.title()}",
                "",
                "## Metadata",
                f"- **Autore**: @{download_data.get('author', 'N/A')}",
                f"- **Piattaforma**: {platform.value}",
                f"- **Like**: {download_data.get('likes', 'N/A'):,}" if isinstance(download_data.get('likes'), int) else f"- **Like**: {download_data.get('likes', 'N/A')}",
                f"- **Commenti**: {download_data.get('comments', 'N/A'):,}" if isinstance(download_data.get('comments'), int) else f"- **Commenti**: {download_data.get('comments', 'N/A')}",
            ]
            
            if platform == Platform.TIKTOK:
                plays = download_data.get('plays')
                if plays:
                    output_lines.append(f"- **Views**: {plays:,}" if isinstance(plays, int) else f"- **Views**: {plays}")
                shares = download_data.get('shares')
                if shares:
                    output_lines.append(f"- **Condivisioni**: {shares:,}" if isinstance(shares, int) else f"- **Condivisioni**: {shares}")
            else:
                views = download_data.get('views')
                if views:
                    output_lines.append(f"- **Views**: {views:,}" if isinstance(views, int) else f"- **Views**: {views}")
            
            duration = download_data.get('duration')
            if duration:
                output_lines.append(f"- **Durata**: {duration}s")
            
            caption = download_data.get('caption', '')
            if caption:
                output_lines.extend([
                    "",
                    "## Caption",
                    f"> {caption[:300]}{'...' if len(caption) > 300 else ''}"
                ])
            
            output_lines.extend([
                "",
                "---",
                "",
                "## AI Analysis",
                "",
                analysis_result
            ])
            
            return "\n".join(output_lines)
        
    except Exception as e:
        return _handle_error(e, "Download & Analyze")


@mcp.tool(
    name="batch_video_analyze",
    annotations={
        "title": "Batch Analyze Videos",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def batch_video_analyze(params: BatchAnalyzeInput) -> str:
    """Analyze multiple videos in parallel.
    
    Arguments:
        params (BatchAnalyzeInput): Input containing list of URLs and analysis options.
        
    Returns:
        str: Combined analysis report for all videos.
    """
    
    async def process_single_video(url: str) -> str:
        try:
            # Re-use existing workflow function for consistency
            single_input = DownloadAndAnalyzeInput(
                url=url,
                analysis_type=params.analysis_type,
                model=params.model,
                custom_prompt=params.custom_prompt,
                language=params.language,
                response_format=params.response_format
            )
            return await video_download_and_analyze(single_input)
        except Exception as e:
            return f"Error analyzing {url}: {str(e)}"

    # Run all analyses in parallel
    results = await asyncio.gather(*[process_single_video(url) for url in params.urls])
    
    # Combine results
    if params.response_format == ResponseFormat.JSON:
        combined_json = []
        for res in results:
            try:
                combined_json.append(json.loads(res))
            except:
                combined_json.append({"error": res})
        return json.dumps(combined_json, indent=2, ensure_ascii=False)
    else:
        # Markdown
        final_report = [f"# Batch Analysis Report ({len(results)} videos)", ""]
        for i, res in enumerate(results):
            final_report.append(f"## Video {i+1} : {params.urls[i]}")
            final_report.append(res)
            final_report.append("\n---\n")
        return "\n".join(final_report)


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    mcp.run()

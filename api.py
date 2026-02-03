#!/usr/bin/env python3
"""
REST API for FORGE v1
====================

FastAPI-based REST API for programmatic access to all FORGE operations:
- Stem separation
- Loop extraction
- Vocal chop generation
- MIDI extraction
- Drum one-shot generation
- Video rendering
- Batch operations

Author: NeuralWorkstation Team
License: MIT
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil
import uuid
from datetime import datetime
import os

# Import core functions
from forgev1 import (
    separate_stems_demucs,
    extract_loops,
    generate_vocal_chops,
    extract_midi,
    generate_drum_oneshots,
    render_video,
    setup_directories,
    Config
)

# Initialize FastAPI app
app = FastAPI(
    title="FORGE v1 API",
    description="REST API for Neural Audio Workstation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# API Key authentication (simple implementation)
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("FORGE_API_KEY", "forge-dev-key-change-in-production")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key: str = Security(api_key_header)):
    """Validate API key."""
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=403,
        detail="Invalid API key"
    )


# Setup directories on startup
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    setup_directories()
    print("🚀 FORGE API started")


# Pydantic models for request/response
class StemSeparationRequest(BaseModel):
    model: str = Field(default="htdemucs", description="Demucs model name")
    use_cache: bool = Field(default=True, description="Use cached results")


class StemSeparationResponse(BaseModel):
    job_id: str
    stems: Dict[str, str]
    message: str


class LoopExtractionRequest(BaseModel):
    loop_duration: float = Field(default=4.0, ge=1.0, le=16.0, description="Loop duration in seconds")
    aperture: float = Field(default=0.5, ge=0.0, le=1.0, description="Aperture control")
    num_loops: int = Field(default=5, ge=1, le=20, description="Number of loops to extract")


class LoopExtractionResponse(BaseModel):
    job_id: str
    loops: List[Dict[str, Any]]
    message: str


class VocalChopRequest(BaseModel):
    mode: str = Field(default="onset", description="Detection mode: silence, onset, or hybrid")
    min_duration: float = Field(default=0.1, ge=0.05, le=5.0, description="Minimum chop duration")
    max_duration: float = Field(default=2.0, ge=0.1, le=10.0, description="Maximum chop duration")
    threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Detection threshold")


class VocalChopResponse(BaseModel):
    job_id: str
    chops: List[str]
    message: str


class MidiExtractionResponse(BaseModel):
    job_id: str
    midi_file: str
    message: str


class DrumOneShotRequest(BaseModel):
    min_duration: float = Field(default=0.05, ge=0.01, le=2.0)
    max_duration: float = Field(default=0.5, ge=0.1, le=5.0)
    apply_fadeout: bool = Field(default=True)


class DrumOneShotResponse(BaseModel):
    job_id: str
    oneshots: List[str]
    message: str


class VideoRenderRequest(BaseModel):
    aspect_ratio: str = Field(default="16:9", description="Aspect ratio")
    visualization: str = Field(default="waveform", description="Visualization type")


class VideoRenderResponse(BaseModel):
    job_id: str
    video_file: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None


# Helper function to save uploaded file
async def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location."""
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    file_id = str(uuid.uuid4())
    file_extension = Path(upload_file.filename).suffix
    file_path = temp_dir / f"{file_id}{file_extension}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return str(file_path)


# Mock progress tracker for API
class MockProgress:
    def __call__(self, *args, **kwargs):
        pass


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "FORGE v1 API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/stem-separation", response_model=StemSeparationResponse)
async def stem_separation(
    file: UploadFile = File(...),
    request: StemSeparationRequest = StemSeparationRequest(),
    api_key: APIKey = Depends(get_api_key)
):
    """
    Separate audio into stems using Demucs.
    
    - **file**: Audio file to process
    - **model**: Demucs model to use
    - **use_cache**: Whether to use cached results
    """
    try:
        # Save uploaded file
        file_path = await save_upload_file(file)
        
        # Process
        job_id = str(uuid.uuid4())
        stems = separate_stems_demucs(
            audio_path=file_path,
            model=request.model,
            use_cache=request.use_cache,
            progress=MockProgress()
        )
        
        # Cleanup temp file
        Path(file_path).unlink(missing_ok=True)
        
        return StemSeparationResponse(
            job_id=job_id,
            stems=stems,
            message=f"Stems separated successfully using {request.model}"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/loop-extraction", response_model=LoopExtractionResponse)
async def loop_extraction(
    file: UploadFile = File(...),
    request: LoopExtractionRequest = LoopExtractionRequest(),
    api_key: APIKey = Depends(get_api_key)
):
    """
    Extract loops from audio file.
    
    - **file**: Audio file to process
    - **loop_duration**: Duration of each loop in seconds
    - **aperture**: Aperture control (0-1)
    - **num_loops**: Number of loops to extract
    """
    try:
        file_path = await save_upload_file(file)
        
        job_id = str(uuid.uuid4())
        loops = extract_loops(
            audio_path=file_path,
            loop_duration=request.loop_duration,
            aperture=request.aperture,
            num_loops=request.num_loops,
            progress=MockProgress()
        )
        
        Path(file_path).unlink(missing_ok=True)
        
        return LoopExtractionResponse(
            job_id=job_id,
            loops=loops,
            message=f"Extracted {len(loops)} loops successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/vocal-chops", response_model=VocalChopResponse)
async def vocal_chops(
    file: UploadFile = File(...),
    request: VocalChopRequest = VocalChopRequest(),
    api_key: APIKey = Depends(get_api_key)
):
    """
    Generate vocal chops from audio.
    
    - **file**: Audio file (preferably vocals)
    - **mode**: Detection mode (silence, onset, hybrid)
    - **min_duration**: Minimum chop duration
    - **max_duration**: Maximum chop duration
    - **threshold**: Detection threshold
    """
    try:
        file_path = await save_upload_file(file)
        
        job_id = str(uuid.uuid4())
        chops = generate_vocal_chops(
            audio_path=file_path,
            mode=request.mode,
            min_duration=request.min_duration,
            max_duration=request.max_duration,
            threshold=request.threshold,
            progress=MockProgress()
        )
        
        Path(file_path).unlink(missing_ok=True)
        
        return VocalChopResponse(
            job_id=job_id,
            chops=chops,
            message=f"Generated {len(chops)} vocal chops successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/midi-extraction", response_model=MidiExtractionResponse)
async def midi_extraction(
    file: UploadFile = File(...),
    api_key: APIKey = Depends(get_api_key)
):
    """
    Extract MIDI from audio file.
    
    - **file**: Audio file to transcribe
    """
    try:
        file_path = await save_upload_file(file)
        
        job_id = str(uuid.uuid4())
        midi_file = extract_midi(
            audio_path=file_path,
            progress=MockProgress()
        )
        
        Path(file_path).unlink(missing_ok=True)
        
        return MidiExtractionResponse(
            job_id=job_id,
            midi_file=midi_file,
            message="MIDI extracted successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/drum-oneshots", response_model=DrumOneShotResponse)
async def drum_oneshots(
    file: UploadFile = File(...),
    request: DrumOneShotRequest = DrumOneShotRequest(),
    api_key: APIKey = Depends(get_api_key)
):
    """
    Generate drum one-shots from audio.
    
    - **file**: Audio file (drums)
    - **min_duration**: Minimum one-shot duration
    - **max_duration**: Maximum one-shot duration
    - **apply_fadeout**: Apply fadeout to one-shots
    """
    try:
        file_path = await save_upload_file(file)
        
        job_id = str(uuid.uuid4())
        oneshots = generate_drum_oneshots(
            audio_path=file_path,
            min_duration=request.min_duration,
            max_duration=request.max_duration,
            apply_fadeout=request.apply_fadeout,
            progress=MockProgress()
        )
        
        Path(file_path).unlink(missing_ok=True)
        
        return DrumOneShotResponse(
            job_id=job_id,
            oneshots=oneshots,
            message=f"Generated {len(oneshots)} drum one-shots successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/download/{filename}")
async def download_file(
    filename: str,
    api_key: APIKey = Depends(get_api_key)
):
    """
    Download a generated file.
    
    - **filename**: Name of the file to download
    """
    # Search for file in output directories
    search_dirs = [
        Path('output/stems'),
        Path('output/loops'),
        Path('output/chops'),
        Path('output/midi'),
        Path('output/drums'),
        Path('output/videos')
    ]
    
    for directory in search_dirs:
        file_path = directory / filename
        if file_path.exists():
            return FileResponse(file_path)
    
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/api/v1/models")
async def list_models(api_key: APIKey = Depends(get_api_key)):
    """List available Demucs models."""
    return {
        "models": Config.DEMUCS_MODELS,
        "default": "htdemucs"
    }


@app.get("/api/v1/config")
async def get_config(api_key: APIKey = Depends(get_api_key)):
    """Get current configuration."""
    return {
        "sample_rate": Config.SAMPLE_RATE,
        "hop_length": Config.HOP_LENGTH,
        "n_fft": Config.N_FFT,
        "demucs_models": Config.DEMUCS_MODELS,
        "video_fps": Config.VIDEO_FPS,
        "aspect_ratios": list(Config.ASPECT_RATIOS.keys())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

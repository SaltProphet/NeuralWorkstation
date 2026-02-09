#!/usr/bin/env python3
"""
Compatibility wrapper for legacy tests and batch processor usage.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app import (
    FORGEAudioProcessor,
    FORGEConfig,
    FORGEFeedback,
    FORGEVideoRenderer,
)


class Config(FORGEConfig):
    """Legacy alias for configuration."""


_PROCESSOR: Optional[FORGEAudioProcessor] = None


def _get_processor() -> FORGEAudioProcessor:
    global _PROCESSOR
    if _PROCESSOR is None:
        _PROCESSOR = FORGEAudioProcessor(Config())
    return _PROCESSOR


def sanitize_filename(name: str) -> str:
    """Sanitize file names for safe output paths."""
    parts = re.split(r"[\\/]+", name)
    parts = [part for part in parts if part not in ("", ".", "..")]
    combined = "_".join(parts) if parts else "file"
    combined = re.sub(r"[^A-Za-z0-9]+", "_", combined)
    combined = re.sub(r"_+", "_", combined)
    combined = combined.strip("_")
    if not combined:
        combined = "file"
    return combined[:100]


def get_audio_hash(audio_path: str) -> str:
    """Get stable MD5 hash for an audio file."""
    return _get_processor()._get_audio_hash(audio_path)


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.mmm."""
    return _get_processor()._format_timestamp(seconds)


def db_to_amplitude(db: float) -> float:
    """Convert decibels to amplitude."""
    return _get_processor()._db_to_amplitude(db)


def amplitude_to_db(amplitude: float) -> float:
    """Convert amplitude to decibels."""
    return _get_processor()._amplitude_to_db(amplitude)


def setup_directories() -> List[str]:
    """Create required application directories."""
    return Config.setup_directories()


def save_config(config_dict: dict, name: str = "default") -> None:
    """Save configuration to JSON file."""
    Config.save_config(config_dict, name)


def load_config(name: str = "default") -> dict:
    """Load configuration from JSON file."""
    return Config.load_config(name)


def separate_stems_demucs(
    audio_path: str,
    model: str = "htdemucs",
    use_cache: bool = True,
    progress=None,
) -> Dict[str, str]:
    """Separate stems with Demucs."""
    return _get_processor().separate_stems_demucs(
        audio_path=audio_path,
        model=model,
        use_cache=use_cache,
        progress=progress,
    )


def separate_stems_audiosep(
    audio_path: str,
    query: str,
    progress=None,
) -> str:
    """Separate stems with AudioSep."""
    return _get_processor().separate_stems_audiosep(
        audio_path=audio_path,
        query=query,
        progress=progress,
    )


def extract_loops(
    audio_path: str,
    loop_duration: float = 4.0,
    aperture: float = 0.5,
    num_loops: int = 10,
    progress=None,
) -> List[Dict[str, Any]]:
    """Extract and rank loops from audio."""
    return _get_processor().extract_loops(
        audio_path=audio_path,
        loop_duration=loop_duration,
        aperture=aperture,
        num_loops=num_loops,
        progress=progress,
    )


def generate_vocal_chops(
    audio_path: str,
    mode: str = "onset",
    min_duration: float = 0.1,
    max_duration: float = 2.0,
    threshold: float = 0.3,
    progress=None,
) -> List[str]:
    """Generate vocal chops from audio."""
    return _get_processor().generate_vocal_chops(
        audio_path=audio_path,
        mode=mode,
        min_duration=min_duration,
        max_duration=max_duration,
        threshold=threshold,
        progress=progress,
    )


def extract_midi(
    audio_path: str,
    progress=None,
) -> str:
    """Extract MIDI from audio."""
    return _get_processor().extract_midi(
        audio_path=audio_path,
        progress=progress,
    )


def generate_drum_oneshots(
    audio_path: str,
    min_duration: float = 0.05,
    max_duration: float = 1.0,
    progress=None,
) -> List[str]:
    """Generate drum one-shots from audio."""
    return _get_processor().generate_drum_oneshots(
        audio_path=audio_path,
        min_duration=min_duration,
        max_duration=max_duration,
        progress=progress,
    )


def render_video(
    audio_path: str,
    aspect_ratio: str = "16:9",
    visualization_type: str = "waveform",
    progress=None,
) -> str:
    """Render video visualization for audio."""
    renderer = FORGEVideoRenderer(Config())
    return renderer.render_video(
        audio_path=audio_path,
        aspect_ratio=aspect_ratio,
        visualization_type=visualization_type,
        progress=progress,
    )


def save_feedback(
    feature: str,
    rating: int,
    comments: str,
    email: Optional[str] = None,
) -> str:
    """Save user feedback to disk."""
    return FORGEFeedback.save_feedback(feature, rating, comments, email)

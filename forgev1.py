#!/usr/bin/env python3
"""
FORGE v1 - Unified Neural Audio Workstation
============================================

A comprehensive audio processing workstation that integrates:
- Night Pulse: Demucs-based stem separation with caching
- FORGE: AudioSep query-based advanced stem extraction
- Loop slicing and ranking with dynamic 'Aperture' control
- Vocal chop generator (silence, onset, hybrid modes)
- MIDI extraction using basic_pitch
- Drum one-shot generator
- Video rendering with FFmpeg (dynamic visuals, multiple aspect ratios)
- User feedback mechanism with JSON storage

Author: NeuralWorkstation Team
License: MIT
"""

import os
import sys
import json
import hashlib
import shutil
import warnings
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import librosa
import soundfile as sf
import gradio as gr

warnings.filterwarnings('ignore')

# ============================================================================
# OPTIONAL FEATURES DETECTION
# ============================================================================

# Check if AudioSep is available
try:
    import audiosep
    AUDIOSEP_AVAILABLE = True
except ImportError:
    AUDIOSEP_AVAILABLE = False

# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def setup_directories():
    """
    Create necessary directories for the application.
    Ensures all output paths exist before processing.
    """
    directories = [
        'runs',
        'output',
        'cache',
        'config',
        'checkpoint',
        'feedback',
        'output/stems',
        'output/loops',
        'output/chops',
        'output/midi',
        'output/drums',
        'output/videos',
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    return directories

# ============================================================================
# CONFIGURATION AND UTILITIES
# ============================================================================

class Config:
    """
    Configuration manager for audio processing parameters.
    Stores default values and provides methods for loading/saving configs.
    """
    
    # Demucs models
    DEMUCS_MODELS = ['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'mdx_extra', 'mdx_extra_q']
    
    # Audio parameters
    SAMPLE_RATE = 44100
    HOP_LENGTH = 512
    N_FFT = 2048
    
    # Loop parameters
    DEFAULT_LOOP_LENGTH = 4  # bars
    MIN_LOOP_DURATION = 1.0  # seconds
    MAX_LOOP_DURATION = 16.0  # seconds
    
    # Output directories
    OUTPUT_DIR_STEMS = Path('output/stems')
    OUTPUT_DIR_LOOPS = Path('output/loops')
    OUTPUT_DIR_CHOPS = Path('output/chops')
    OUTPUT_DIR_MIDI = Path('output/midi')
    OUTPUT_DIR_DRUMS = Path('output/drums')
    OUTPUT_DIR_VIDEOS = Path('output/videos')
    
    # Video parameters
    VIDEO_FPS = 30
    VIDEO_BITRATE = '2M'
    ASPECT_RATIOS = {
        '16:9': (1920, 1080),
        '4:3': (1024, 768),
        '1:1': (1080, 1080),
        '9:16': (1080, 1920),
    }
    
    @staticmethod
    def save_config(config_dict: dict, name: str = 'default'):
        """Save configuration to JSON file."""
        config_path = Path('config') / f'{name}.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @staticmethod
    def load_config(name: str = 'default') -> dict:
        """Load configuration from JSON file."""
        config_path = Path('config') / f'{name}.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and ensure filesystem safety.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        Sanitized filename safe for filesystem use
    """
    # Remove or replace unsafe characters
    # Keep only alphanumeric, spaces, hyphens, and underscores
    import re
    sanitized = re.sub(r'[^\w\s\-]', '_', filename)
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Limit length to prevent issues
    return sanitized[:100]


def get_audio_hash(audio_path: str) -> str:
    """
    Generate MD5 hash of audio file for caching purposes.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(audio_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS.mmm timestamp."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def db_to_amplitude(db: float) -> float:
    """Convert decibels to amplitude."""
    return 10 ** (db / 20)


def amplitude_to_db(amplitude: float) -> float:
    """Convert amplitude to decibels."""
    return 20 * np.log10(max(amplitude, 1e-10))

# ============================================================================
# PHASE 1: STEM SEPARATION (Night Pulse - Demucs with Caching)
# ============================================================================

def separate_stems_demucs(
    audio_path: str,
    model: str = 'htdemucs',
    use_cache: bool = True,
    progress=gr.Progress()
) -> Dict[str, str]:
    """
    Separate audio into stems using Demucs with intelligent caching.
    
    Args:
        audio_path: Path to input audio file
        model: Demucs model to use
        use_cache: Whether to use cached results
        progress: Gradio progress tracker
        
    Returns:
        Dictionary mapping stem names to file paths
    """
    try:
        progress(0, desc="Initializing Demucs separation...")
        
        # Check if demucs is available
        import subprocess
        result = subprocess.run(['which', 'demucs'], capture_output=True, text=True)
        demucs_available = result.returncode == 0
        
        if not demucs_available:
            raise RuntimeError("Demucs not found. Please install: pip install demucs")
        
        # Generate cache key
        audio_hash = get_audio_hash(audio_path)
        cache_key = f"{audio_hash}_{model}"
        cache_dir = Path('cache') / cache_key
        
        # Check cache
        if use_cache and cache_dir.exists():
            progress(0.9, desc="Loading from cache...")
            cached_stems = {}
            for stem_file in cache_dir.glob('*.wav'):
                stem_name = stem_file.stem
                cached_stems[stem_name] = str(stem_file)
            
            if cached_stems:
                progress(1.0, desc="Loaded from cache!")
                return cached_stems
        
        # Run Demucs
        progress(0.3, desc=f"Running Demucs ({model})...")
        output_dir = Path('output/stems')
        
        cmd = [
            'demucs',
            '-n', model,
            '-o', str(output_dir),
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Demucs failed: {result.stderr}")
        
        progress(0.7, desc="Processing stems...")
        
        # Find generated stems
        audio_name = Path(audio_path).stem
        stem_dir = output_dir / model / audio_name
        
        stems = {}
        if stem_dir.exists():
            for stem_file in stem_dir.glob('*.wav'):
                stem_name = stem_file.stem
                output_path = output_dir / f"{audio_name}_{stem_name}.wav"
                shutil.copy(stem_file, output_path)
                stems[stem_name] = str(output_path)
                
                # Cache the stem
                if use_cache:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_path = cache_dir / f"{stem_name}.wav"
                    shutil.copy(stem_file, cache_path)
        
        progress(1.0, desc="Separation complete!")
        return stems
        
    except Exception as e:
        raise RuntimeError(f"Demucs separation failed: {str(e)}\n{traceback.format_exc()}")


# ============================================================================
# PHASE 1.5: AUDIOSEP (Advanced Query-Based Separation)
# ============================================================================

def separate_stems_audiosep(
    audio_path: str,
    query: str,
    progress=gr.Progress()
) -> str:
    """
    Perform query-based stem separation using AudioSep.
    
    Args:
        audio_path: Path to input audio file
        query: Natural language query (e.g., "bass guitar", "snare drum")
        progress: Gradio progress tracker
        
    Returns:
        Path to separated audio
    """
    try:
        progress(0, desc="Initializing AudioSep...")
        
        # Check if audiosep is available using the global flag
        if not AUDIOSEP_AVAILABLE:
            raise RuntimeError(
                "AudioSep is not installed.\n\n"
                "To install: pip install audiosep\n\n"
                "Note: AudioSep requires GPU and model checkpoints for optimal performance. "
                "CPU inference is possible but may be slow."
            )
        
        # Import AudioSep (we know it's available now)
        from audiosep import AudioSep
        
        progress(0.2, desc=f"Loading AudioSep model...")
        
        # Initialize AudioSep model
        # Use GPU if available, otherwise fall back to CPU
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except (ImportError, AttributeError) as e:
            # PyTorch not available or doesn't have CUDA support
            device = 'cpu'
        
        separator = AudioSep(device=device)
        
        progress(0.4, desc=f"Loading audio file...")
        
        # Load audio - AudioSep typically works with mono
        audio, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True)
        
        progress(0.6, desc=f"Separating: {query}...")
        
        # Perform separation
        separated_audio = separator.separate(audio, query)
        
        # Validate output
        if separated_audio is None:
            raise RuntimeError("AudioSep returned None. The query may not match any audio content.")
        
        if not isinstance(separated_audio, np.ndarray) or separated_audio.size == 0:
            raise RuntimeError("AudioSep returned invalid or empty audio data.")
        
        progress(0.8, desc="Saving output...")
        
        # Save output - use Config constant and sanitize query for filename
        output_dir = Config.OUTPUT_DIR_STEMS
        audio_name = Path(audio_path).stem
        safe_query = sanitize_filename(query)
        output_path = output_dir / f"{audio_name}_audiosep_{safe_query}.wav"
        
        # Handle output shape - AudioSep may return different shapes
        # Ensure it's in the right format for soundfile
        if separated_audio.ndim == 1:
            # Mono audio
            sf.write(output_path, separated_audio, Config.SAMPLE_RATE)
        elif separated_audio.ndim == 2:
            # Stereo or multi-channel
            # Note: We assume shape[0] < shape[1] means (channels, samples) format
            # This heuristic works for most audio (samples >> channels for any reasonable duration)
            # e.g., 1 second at 44.1kHz = 44100 samples, but channels are typically 1-8
            if separated_audio.shape[0] < separated_audio.shape[1]:
                separated_audio = separated_audio.T
            sf.write(output_path, separated_audio, Config.SAMPLE_RATE)
        else:
            raise RuntimeError(f"Unexpected audio shape from AudioSep: {separated_audio.shape}")
        
        progress(1.0, desc="AudioSep complete!")
        return str(output_path)
        
    except RuntimeError as e:
        # Re-raise RuntimeError (installation/availability issues)
        raise
    except Exception as e:
        # Other errors (model loading, processing, etc.)
        raise RuntimeError(
            f"AudioSep separation failed: {str(e)}\n\n"
            f"Common issues:\n"
            f"- Model checkpoints not downloaded (AudioSep downloads on first use)\n"
            f"- Insufficient memory (try shorter audio clips)\n"
            f"- Invalid audio format (use WAV, MP3, or FLAC)\n"
            f"- GPU out of memory (AudioSep is memory-intensive)"
        )


# ============================================================================
# PHASE 2: LOOP GENERATION & RANKING
# ============================================================================

def extract_loops(
    audio_path: str,
    loop_duration: float = 4.0,
    aperture: float = 0.5,
    num_loops: int = 10,
    progress=gr.Progress()
) -> List[Dict[str, Any]]:
    """
    Extract and rank loops from audio using RMS, onset, and spectral features.
    
    Args:
        audio_path: Path to audio file
        loop_duration: Duration of each loop in seconds
        aperture: Control parameter (0-1) for feature sensitivity
        num_loops: Number of top loops to return
        progress: Gradio progress tracker
        
    Returns:
        List of loop dictionaries with metadata
    """
    try:
        progress(0, desc="Loading audio...")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
        duration = librosa.get_duration(y=y, sr=sr)
        
        progress(0.2, desc="Analyzing audio features...")
        
        # Calculate features
        rms = librosa.feature.rms(y=y, hop_length=Config.HOP_LENGTH)[0]
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Normalize features
        rms_norm = rms / (np.max(rms) + 1e-10)
        onset_norm = onset_env / (np.max(onset_env) + 1e-10)
        centroid_norm = centroid / (np.max(centroid) + 1e-10)
        
        progress(0.4, desc="Slicing loops...")
        
        # Generate loop candidates
        loop_samples = int(loop_duration * sr)
        hop_samples = loop_samples // 2  # 50% overlap
        
        loops = []
        for i in range(0, len(y) - loop_samples, hop_samples):
            start_time = i / sr
            end_time = (i + loop_samples) / sr
            
            if end_time > duration:
                break
            
            # Extract loop
            loop_audio = y[i:i + loop_samples]
            
            # Calculate loop score using aperture-weighted features
            start_frame = librosa.time_to_frames(start_time, sr=sr, hop_length=Config.HOP_LENGTH)
            end_frame = librosa.time_to_frames(end_time, sr=sr, hop_length=Config.HOP_LENGTH)
            
            # Ensure indices are within bounds
            start_frame = max(0, min(start_frame, len(rms_norm) - 1))
            end_frame = max(start_frame + 1, min(end_frame, len(rms_norm)))
            
            # Aperture controls feature weighting (0 = RMS, 0.5 = balanced, 1 = spectral)
            rms_weight = 1.0 - aperture
            onset_weight = 0.5
            centroid_weight = aperture
            
            rms_score = np.mean(rms_norm[start_frame:end_frame])
            onset_score = np.mean(onset_norm[start_frame:end_frame])
            centroid_score = np.mean(centroid_norm[start_frame:end_frame])
            
            total_score = (rms_weight * rms_score + 
                          onset_weight * onset_score + 
                          centroid_weight * centroid_score)
            
            loops.append({
                'start_time': start_time,
                'end_time': end_time,
                'score': total_score,
                'rms_score': rms_score,
                'onset_score': onset_score,
                'centroid_score': centroid_score,
                'audio': loop_audio,
            })
        
        progress(0.7, desc="Ranking loops...")
        
        # Sort by score and select top N
        loops.sort(key=lambda x: x['score'], reverse=True)
        top_loops = loops[:num_loops]
        
        progress(0.9, desc="Exporting loops...")
        
        # Save loops
        output_dir = Path('output/loops')
        audio_name = Path(audio_path).stem
        
        saved_loops = []
        for idx, loop in enumerate(top_loops):
            loop_filename = f"{audio_name}_loop_{idx+1}_score_{loop['score']:.3f}.wav"
            loop_path = output_dir / loop_filename
            
            sf.write(loop_path, loop['audio'], sr)
            
            saved_loops.append({
                'path': str(loop_path),
                'start_time': loop['start_time'],
                'end_time': loop['end_time'],
                'score': loop['score'],
                'rank': idx + 1,
            })
        
        progress(1.0, desc=f"Extracted {len(saved_loops)} loops!")
        return saved_loops
        
    except Exception as e:
        raise RuntimeError(f"Loop extraction failed: {str(e)}\n{traceback.format_exc()}")


# ============================================================================
# PHASE 2: VOCAL CHOP GENERATOR
# ============================================================================

def generate_vocal_chops(
    audio_path: str,
    mode: str = 'onset',
    min_duration: float = 0.1,
    max_duration: float = 2.0,
    threshold: float = 0.3,
    progress=gr.Progress()
) -> List[str]:
    """
    Generate vocal chops from audio using different detection methods.
    
    Args:
        audio_path: Path to audio file (preferably vocals stem)
        mode: Detection mode - 'silence', 'onset', or 'hybrid'
        min_duration: Minimum chop duration in seconds
        max_duration: Maximum chop duration in seconds
        threshold: Detection threshold (0-1)
        progress: Gradio progress tracker
        
    Returns:
        List of paths to generated chop files
    """
    try:
        progress(0, desc="Loading audio...")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
        
        progress(0.3, desc=f"Detecting chops ({mode})...")
        
        chop_boundaries = []
        
        if mode == 'silence':
            # Silence-based detection
            intervals = librosa.effects.split(
                y,
                top_db=20 + (threshold * 20),
                frame_length=2048,
                hop_length=512
            )
            chop_boundaries = [(start / sr, end / sr) for start, end in intervals]
            
        elif mode == 'onset':
            # Onset-based detection
            onset_frames = librosa.onset.onset_detect(
                y=y,
                sr=sr,
                hop_length=Config.HOP_LENGTH,
                backtrack=True,
                delta=threshold * 0.5
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=Config.HOP_LENGTH)
            
            # Create segments between onsets
            for i in range(len(onset_times) - 1):
                start = onset_times[i]
                end = onset_times[i + 1]
                chop_boundaries.append((start, end))
                
        else:  # hybrid
            # Combine both methods
            onset_frames = librosa.onset.onset_detect(
                y=y,
                sr=sr,
                hop_length=Config.HOP_LENGTH,
                delta=threshold * 0.3
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=Config.HOP_LENGTH)
            
            intervals = librosa.effects.split(
                y,
                top_db=20 + (threshold * 15),
                frame_length=2048,
                hop_length=512
            )
            silence_boundaries = [(start / sr, end / sr) for start, end in intervals]
            
            # Merge boundaries
            all_times = sorted(set([b for seg in silence_boundaries for b in seg] + list(onset_times)))
            chop_boundaries = [(all_times[i], all_times[i+1]) for i in range(len(all_times)-1)]
        
        progress(0.6, desc="Filtering chops...")
        
        # Filter by duration
        valid_chops = [
            (start, end) for start, end in chop_boundaries
            if min_duration <= (end - start) <= max_duration
        ]
        
        progress(0.8, desc="Exporting chops...")
        
        # Export chops
        output_dir = Path('output/chops')
        audio_name = Path(audio_path).stem
        
        chop_paths = []
        for idx, (start, end) in enumerate(valid_chops):
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            chop_audio = y[start_sample:end_sample]
            
            chop_filename = f"{audio_name}_chop_{idx+1:03d}.wav"
            chop_path = output_dir / chop_filename
            
            sf.write(chop_path, chop_audio, sr)
            chop_paths.append(str(chop_path))
        
        progress(1.0, desc=f"Generated {len(chop_paths)} chops!")
        return chop_paths
        
    except Exception as e:
        raise RuntimeError(f"Vocal chop generation failed: {str(e)}\n{traceback.format_exc()}")


# ============================================================================
# PHASE 2: MIDI EXTRACTION
# ============================================================================

def extract_midi(
    audio_path: str,
    progress=gr.Progress()
) -> str:
    """
    Extract MIDI notes from audio using basic_pitch.
    
    Args:
        audio_path: Path to audio file
        progress: Gradio progress tracker
        
    Returns:
        Path to generated MIDI file
    """
    try:
        progress(0, desc="Initializing MIDI extraction...")
        
        # Check if basic_pitch is available
        try:
            from basic_pitch.inference import predict
        except ImportError:
            raise RuntimeError("basic_pitch not available. Install with: pip install basic-pitch")
        
        progress(0.3, desc="Running pitch detection...")
        
        # Perform MIDI extraction
        model_output, midi_data, note_events = predict(audio_path)
        
        progress(0.7, desc="Generating MIDI file...")
        
        # Save MIDI
        output_dir = Path('output/midi')
        audio_name = Path(audio_path).stem
        midi_path = output_dir / f"{audio_name}.mid"
        
        midi_data.write(str(midi_path))
        
        progress(1.0, desc="MIDI extraction complete!")
        return str(midi_path)
        
    except Exception as e:
        raise RuntimeError(f"MIDI extraction failed: {str(e)}\nNote: basic_pitch may require additional dependencies")


# ============================================================================
# PHASE 2: DRUM ONE-SHOT GENERATOR
# ============================================================================

def generate_drum_oneshots(
    audio_path: str,
    min_duration: float = 0.05,
    max_duration: float = 1.0,
    progress=gr.Progress()
) -> List[str]:
    """
    Extract drum one-shots from audio (typically drums stem).
    
    Args:
        audio_path: Path to drums audio file
        min_duration: Minimum one-shot duration
        max_duration: Maximum one-shot duration
        progress: Gradio progress tracker
        
    Returns:
        List of paths to drum one-shot files
    """
    try:
        progress(0, desc="Loading drums...")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
        
        progress(0.3, desc="Detecting transients...")
        
        # Detect onsets (drum hits)
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            hop_length=Config.HOP_LENGTH,
            backtrack=True,
            delta=0.2,
            wait=10
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=Config.HOP_LENGTH)
        
        progress(0.6, desc="Extracting one-shots...")
        
        # Extract one-shots
        output_dir = Path('output/drums')
        audio_name = Path(audio_path).stem
        
        oneshot_paths = []
        for idx, onset_time in enumerate(onset_times):
            # Determine one-shot duration
            if idx < len(onset_times) - 1:
                duration = min(onset_times[idx + 1] - onset_time, max_duration)
            else:
                duration = max_duration
            
            if duration < min_duration:
                continue
            
            # Extract one-shot
            start_sample = int(onset_time * sr)
            end_sample = int((onset_time + duration) * sr)
            oneshot_audio = y[start_sample:end_sample]
            
            # Apply fade out
            fade_samples = int(0.01 * sr)  # 10ms fade
            if len(oneshot_audio) > fade_samples:
                fade = np.linspace(1, 0, fade_samples)
                oneshot_audio[-fade_samples:] *= fade
            
            # Save one-shot
            oneshot_filename = f"{audio_name}_hit_{idx+1:03d}.wav"
            oneshot_path = output_dir / oneshot_filename
            
            sf.write(oneshot_path, oneshot_audio, sr)
            oneshot_paths.append(str(oneshot_path))
        
        progress(1.0, desc=f"Generated {len(oneshot_paths)} drum one-shots!")
        return oneshot_paths
        
    except Exception as e:
        raise RuntimeError(f"Drum one-shot generation failed: {str(e)}\n{traceback.format_exc()}")


# ============================================================================
# PHASE 3: VIDEO RENDERING
# ============================================================================

def render_video(
    audio_path: str,
    aspect_ratio: str = '16:9',
    visualization_type: str = 'waveform',
    progress=gr.Progress()
) -> str:
    """
    Render video with audio visualization using FFmpeg.
    
    Args:
        audio_path: Path to audio file
        aspect_ratio: Video aspect ratio ('16:9', '4:3', '1:1', '9:16')
        visualization_type: Type of visualization ('waveform', 'spectrum', 'both')
        progress: Gradio progress tracker
        
    Returns:
        Path to rendered video file
    """
    try:
        progress(0, desc="Initializing video renderer...")
        
        # Check FFmpeg
        import subprocess
        result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg")
        
        # Get video dimensions
        width, height = Config.ASPECT_RATIOS.get(aspect_ratio, (1920, 1080))
        
        progress(0.3, desc=f"Rendering video ({aspect_ratio})...")
        
        # Prepare output path
        output_dir = Path('output/videos')
        audio_name = Path(audio_path).stem
        video_path = output_dir / f"{audio_name}_{aspect_ratio.replace(':', 'x')}.mp4"
        
        # Build FFmpeg command based on visualization type
        if visualization_type == 'waveform':
            filter_complex = (
                f"[0:a]showwaves=s={width}x{height}:mode=line:colors=cyan[v]"
            )
        elif visualization_type == 'spectrum':
            filter_complex = (
                f"[0:a]showfreqs=s={width}x{height}:mode=line:colors=magenta[v]"
            )
        else:  # both
            half_height = height // 2
            filter_complex = (
                f"[0:a]asplit[a1][a2];"
                f"[a1]showwaves=s={width}x{half_height}:mode=line:colors=cyan[v1];"
                f"[a2]showfreqs=s={width}x{half_height}:mode=line:colors=magenta[v2];"
                f"[v1][v2]vstack[v]"
            )
        
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-i', audio_path,
            '-filter_complex', filter_complex,
            '-map', '[v]',
            '-map', '0:a',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-b:v', Config.VIDEO_BITRATE,
            '-c:a', 'aac',
            '-b:a', '192k',
            '-r', str(Config.VIDEO_FPS),
            str(video_path)
        ]
        
        progress(0.5, desc="Processing video...")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        progress(1.0, desc="Video rendering complete!")
        return str(video_path)
        
    except Exception as e:
        raise RuntimeError(f"Video rendering failed: {str(e)}\n{traceback.format_exc()}")


# ============================================================================
# FEEDBACK MECHANISM
# ============================================================================

def save_feedback(
    feature: str,
    rating: int,
    comments: str,
    email: Optional[str] = None
) -> str:
    """
    Save user feedback to JSON file.
    
    Args:
        feature: Feature being reviewed
        rating: Rating (1-5)
        comments: User comments
        email: Optional user email
        
    Returns:
        Confirmation message
    """
    try:
        feedback_dir = Path('feedback')
        
        # Create timestamped feedback entry
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        feedback_data = {
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            'feature': feature,
            'rating': rating,
            'comments': comments,
            'email': email
        }
        
        # Save to file
        feedback_file = feedback_dir / f"feedback_{timestamp}.json"
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        return f"✅ Feedback saved successfully! Thank you for your input.\nFile: {feedback_file.name}"
        
    except Exception as e:
        return f"❌ Failed to save feedback: {str(e)}"


# ============================================================================
# GRADIO UI
# ============================================================================

# Custom CSS for FORGE Neural Workstation theme (module-level for reusability)
CUSTOM_CSS = """
/* ==================== FORGE NEURAL WORKSTATION THEME ==================== */

/* Global container styling - Dark theme */
.gradio-container {
    font-family: 'Courier New', 'Consolas', monospace !important;
    background: #0a0a0a !important;
}

/* Main content area */
.main {
    background: #0a0a0a !important;
}

/* Header styling */
.forge-header {
    background: #1a1a1a !important;
    border-bottom: 2px solid #ff6b35 !important;
    padding: 20px !important;
    margin-bottom: 0 !important;
}

.forge-title {
    color: #ffffff !important;
    font-size: 28px !important;
    font-weight: bold !important;
    letter-spacing: 2px !important;
    margin: 0 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .forge-subtitle {
        color: #888888 !important;
        font-size: 12px !important;
        margin: 5px 0 0 0 !important;
    }
    
    .system-status {
        color: #ff6b35 !important;
        font-size: 12px !important;
        float: right !important;
        margin-top: -35px !important;
    }
    
    /* Tab styling - Orange accents */
    .tab-nav {
        background: #1a1a1a !important;
        border-bottom: 1px solid #333333 !important;
    }
    
    .tab-nav button {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #888888 !important;
        background: transparent !important;
        border: none !important;
        padding: 15px 25px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .tab-nav button:hover {
        color: #ff6b35 !important;
        background: #222222 !important;
    }
    
    .tab-nav button.selected {
        color: #ff6b35 !important;
        border-bottom: 3px solid #ff6b35 !important;
        background: #1a1a1a !important;
    }
    
    /* Card/Section styling */
    .forge-card {
        background: #1a1a1a !important;
        border: 1px solid #333333 !important;
        border-radius: 4px !important;
        padding: 20px !important;
        margin: 15px 0 !important;
    }
    
    .forge-card-header {
        color: #ff6b35 !important;
        font-size: 16px !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        margin-bottom: 15px !important;
        padding-bottom: 10px !important;
        border-bottom: 2px solid #ff6b35 !important;
    }
    
    /* Console styling */
    .system-console {
        background: #0d0d0d !important;
        border: 1px solid #ff6b35 !important;
        border-radius: 4px !important;
        padding: 15px !important;
        font-family: 'Courier New', monospace !important;
        font-size: 12px !important;
        color: #00ff00 !important;
        height: 400px !important;
        overflow-y: auto !important;
    }
    
    .console-header {
        color: #ff6b35 !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        margin-bottom: 10px !important;
        padding-bottom: 8px !important;
        border-bottom: 1px solid #ff6b35 !important;
    }
    
    /* Session output styling */
    .session-output {
        background: #0d0d0d !important;
        border: 1px solid #333333 !important;
        border-radius: 4px !important;
        padding: 15px !important;
        margin-top: 15px !important;
    }
    
    .session-header {
        color: #888888 !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        font-size: 12px !important;
        letter-spacing: 1.5px !important;
        margin-bottom: 10px !important;
    }
    
    /* Button styling - Orange primary buttons */
    .gr-button-primary {
        background: #ff6b35 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        padding: 15px 30px !important;
        font-size: 14px !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .gr-button-primary:hover {
        background: #ff8552 !important;
    }
    
    /* Input/Textbox styling */
    input, textarea, select {
        background: #0d0d0d !important;
        border: 1px solid #333333 !important;
        color: #ffffff !important;
        font-family: 'Courier New', monospace !important;
    }
    
    input:focus, textarea:focus, select:focus {
        border-color: #ff6b35 !important;
    }
    
    /* Label styling */
    label {
        color: #888888 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-family: 'Courier New', monospace !important;
    }
    
    /* Dropdown styling */
    .gr-dropdown {
        background: #0d0d0d !important;
        border: 1px solid #333333 !important;
    }
    
    /* Slider styling */
    .gr-slider input[type="range"] {
        background: #333333 !important;
    }
    
    .gr-slider input[type="range"]::-webkit-slider-thumb {
        background: #ff6b35 !important;
    }
    
    /* Checkbox styling */
    input[type="checkbox"] {
        accent-color: #ff6b35 !important;
    }
    
    /* File upload area styling */
    .file-upload {
        background: #0d0d0d !important;
        border: 2px dashed #333333 !important;
        border-radius: 4px !important;
        padding: 30px !important;
        text-align: center !important;
    }
    
    .file-upload:hover {
        border-color: #ff6b35 !important;
    }
    
    /* Status/Output text styling */
    .gr-textbox {
        background: #0d0d0d !important;
        border: 1px solid #333333 !important;
        color: #00ff00 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 12px !important;
    }
    
    /* JSON output styling */
    .gr-json {
        background: #0d0d0d !important;
        border: 1px solid #333333 !important;
        color: #ffffff !important;
    }
    
    /* Progress bar */
    .progress-bar {
        background: #ff6b35 !important;
        height: 4px !important;
    }
    
    /* Markdown text */
    .markdown-text {
        color: #cccccc !important;
    }
    
    /* Remove default gradio margins */
    .gap {
        gap: 0 !important;
    }
    
    /* Row spacing */
    .gr-row {
        margin: 10px 0 !important;
    }
    
/* Column spacing */
.gr-column {
    padding: 10px !important;
}
"""

def create_gradio_interface():
    """
    Create the main Gradio interface with FORGE Neural Workstation styling.
    Implements dark theme with orange accents, multi-phase tabs, and persistent console.
    """
    
    with gr.Blocks(title="FORGE // NEURAL WORKSTATION") as app:
        
        # ==================== HEADER BAR ====================
        with gr.Row(elem_classes="forge-header"):
            with gr.Column(scale=8):
                gr.HTML("""
                    <div class="forge-title">FORGE // NEURAL WORKSTATION</div>
                    <div class="forge-subtitle">v1.1.5 (Daily Driver + AudioSep)</div>
                """)
            with gr.Column(scale=2):
                gr.HTML('<div class="system-status">⚡ SYSTEM READY</div>')
        
        # ==================== MAIN LAYOUT: TABS + CONSOLE ====================
        with gr.Row():
            # Left side: Main content tabs (70% width)
            with gr.Column(scale=7):
                with gr.Tabs():
                    
                    # ==================== PHASE 1: SOURCE ====================
                    with gr.Tab("PHASE 1: SOURCE"):
                        gr.HTML('<div class="forge-card-header">1.1 AUDIO INPUT</div>')
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                demucs_audio = gr.Audio(label="Drag & drop or browse to upload", type="filepath")
                                gr.Markdown("*YouTube / SoundCloud URL*")
                                url_input = gr.Textbox(label="", placeholder="Enter URL...", show_label=False)
                        
                        gr.HTML('<div class="forge-card-header">1.2 SEPARATION SETTINGS</div>')
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column(scale=1):
                                demucs_model = gr.Dropdown(
                                    choices=Config.DEMUCS_MODELS,
                                    value='htdemucs',
                                    label="Source Model",
                                    info="Select separation model"
                                )
                            with gr.Column(scale=1):
                                stem_selection = gr.Dropdown(
                                    choices=['Auto', 'Vocals', 'Drums', 'Bass', 'Other'],
                                    value='Auto',
                                    label="Select Stem (Optional)",
                                    info="Choose specific stem"
                                )
                        
                        with gr.Row(elem_classes="forge-card"):
                            demucs_cache = gr.Checkbox(label="Force Re-run Demucs (ignore cache)", value=False)
                        
                        with gr.Row():
                            demucs_btn = gr.Button("🚀 PHASE 1: ANALYZE + SEPARATE", variant="primary", size="lg")
                        
                        with gr.Row():
                            demucs_output = gr.JSON(label="Separated Stems Results", visible=False)
                            demucs_status = gr.Textbox(label="Status", lines=2, interactive=False)
                        
                        def demucs_wrapper(audio, model, cache):
                            if not audio:
                                return {"error": "No audio uploaded"}, "❌ [ERROR] No audio file provided"
                            try:
                                result = separate_stems_demucs(audio, model, not cache)
                                return result, f"✅ [SUCCESS] Stems separated using {model}"
                            except Exception as e:
                                return {"error": str(e)}, f"❌ [ERROR] {str(e)}"
                        
                        demucs_btn.click(
                            fn=demucs_wrapper,
                            inputs=[demucs_audio, demucs_model, demucs_cache],
                            outputs=[demucs_output, demucs_status]
                        )
                    
                    # ==================== PHASE 1.5: AUDIOSEP ====================
                    with gr.Tab("PHASE 1.5: AUDIOSEP"):
                        gr.HTML('<div class="forge-card-header">1.5 ADVANCED STEM EXTRACTION (AUDIOSEP)</div>')
                        
                        # Show availability status
                        if AUDIOSEP_AVAILABLE:
                            gr.Markdown("*✅ AudioSep is available. Extract specific audio elements using natural language queries.*")
                        else:
                            gr.Markdown("""
                            *⚠️ AudioSep is not installed. This is an optional feature.*
                            
                            To enable AudioSep, run: `pip install audiosep`
                            
                            Note: AudioSep requires GPU and model checkpoints for best performance.
                            """)
                        
                        gr.HTML('<div class="forge-card-header">1.5.1 AUDIO INPUT</div>')
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                audiosep_audio = gr.Audio(
                                    label="Upload Audio File (manually select from Phase 1 stems folder if needed)", 
                                    type="filepath"
                                )
                        
                        gr.HTML('<div class="forge-card-header">1.5.2 QUERY SETTINGS</div>')
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                audiosep_query = gr.Textbox(
                                    label="Natural Language Query (e.g., 'bass guitar', 'snare drum', 'piano', 'saxophone')",
                                    placeholder="Describe the audio element you want to extract",
                                    value="bass guitar",
                                    lines=2
                                )
                        
                        with gr.Row():
                            audiosep_btn = gr.Button(
                                "⚡ EXTRACT" if AUDIOSEP_AVAILABLE else "⚠️ AUDIOSEP NOT INSTALLED",
                                variant="primary" if AUDIOSEP_AVAILABLE else "secondary", 
                                size="lg",
                                interactive=AUDIOSEP_AVAILABLE
                            )
                        
                        with gr.Row():
                            audiosep_output = gr.Audio(label="Extracted Audio", visible=True)
                            audiosep_status = gr.Textbox(label="Status", lines=3, interactive=False)
                        
                        def audiosep_wrapper(audio, query):
                            """Wrapper for AudioSep with proper error handling."""
                            # Check if AudioSep is available
                            if not AUDIOSEP_AVAILABLE:
                                return None, "❌ [ERROR] AudioSep is not installed.\n\nInstall with: pip install audiosep\n\nNote: Requires GPU and model checkpoints."
                            
                            # Validate inputs
                            if not audio:
                                return None, "❌ [ERROR] No audio file provided. Please upload an audio file."
                            
                            if not query or not query.strip():
                                return None, "❌ [ERROR] No query provided. Please enter a natural language query (e.g., 'bass guitar')."
                            
                            try:
                                # Call the actual AudioSep function
                                result_path = separate_stems_audiosep(audio, query.strip())
                                return result_path, f"✅ [SUCCESS] AudioSep extraction complete!\n\nQuery: {query.strip()}\nOutput: {result_path}"
                            except RuntimeError as e:
                                # Handle AudioSep-specific errors
                                return None, f"❌ [ERROR] AudioSep failed:\n\n{str(e)}"
                            except Exception as e:
                                # Handle any other errors
                                return None, f"❌ [ERROR] Unexpected error:\n\n{str(e)}\n\nPlease check your audio file and query."
                        
                        audiosep_btn.click(
                            fn=audiosep_wrapper,
                            inputs=[audiosep_audio, audiosep_query],
                            outputs=[audiosep_output, audiosep_status]
                        )
                    
                    # ==================== PHASE 2: LOOP GENERATION ====================
                    with gr.Tab("PHASE 2: LOOP GENERATION"):
                        gr.HTML('<div class="forge-card-header">2.1 AUDIO INPUT</div>')
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                loop_audio = gr.Audio(
                                    label="Upload Audio File (original track or stem from Phase 1)", 
                                    type="filepath"
                                )
                        
                        gr.HTML('<div class="forge-card-header">2.2 LOOP SETTINGS</div>')
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                loop_duration = gr.Slider(
                                    minimum=Config.MIN_LOOP_DURATION,
                                    maximum=Config.MAX_LOOP_DURATION,
                                    value=4.0,
                                    step=0.5,
                                    label="Loop Duration (seconds)",
                                    info="4s = 4 bars at 120 BPM"
                                )
                            with gr.Column():
                                loop_aperture = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.5,
                                    step=0.05,
                                    label="Aperture",
                                    info="0.0=energy focused, 1.0=spectral focused"
                                )
                        
                        with gr.Row(elem_classes="forge-card"):
                            loop_num_loops = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=10,
                                step=1,
                                label="Number of Loops to Extract"
                            )
                        
                        with gr.Row():
                            loop_extract_btn = gr.Button("🔄 EXTRACT LOOPS", variant="primary", size="lg")
                        
                        with gr.Row():
                            loop_output = gr.File(label="Generated Loops", file_count="multiple")
                            loop_status = gr.Textbox(label="Status", lines=3, interactive=False)
                        
                        def loop_wrapper(audio, duration, aperture, num_loops):
                            if not audio:
                                return None, "❌ [ERROR] No audio file provided"
                            try:
                                loops = extract_loops(audio, duration, aperture, num_loops)
                                return loops, f"✅ [SUCCESS] Extracted {len(loops)} loops\nSaved to output/loops/"
                            except Exception as e:
                                return None, f"❌ [ERROR] {str(e)}"
                        
                        loop_extract_btn.click(
                            fn=loop_wrapper,
                            inputs=[loop_audio, loop_duration, loop_aperture, loop_num_loops],
                            outputs=[loop_output, loop_status]
                        )
                    
                    # ==================== PHASE 2: VOCAL CHOPS ====================
                    with gr.Tab("PHASE 2: VOCAL CHOPS"):
                        gr.HTML('<div class="forge-card-header">2.3 AUDIO INPUT</div>')
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                chop_audio = gr.Audio(
                                    label="Upload Vocal Audio (vocals stem from Phase 1 recommended)", 
                                    type="filepath"
                                )
                        
                        gr.HTML('<div class="forge-card-header">2.4 CHOP SETTINGS</div>')
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                chop_mode = gr.Dropdown(
                                    choices=['hybrid', 'silence', 'onset'],
                                    value='hybrid',
                                    label="Detection Mode",
                                    info="Hybrid=best for most, Silence=sparse vocals, Onset=rhythmic"
                                )
                            with gr.Column():
                                chop_threshold = gr.Slider(
                                    minimum=-60,
                                    maximum=-10,
                                    value=-40,
                                    step=1,
                                    label="Silence Threshold (dB)",
                                    info="Only used in silence/hybrid modes"
                                )
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                chop_min_duration = gr.Slider(
                                    minimum=0.05,
                                    maximum=2.0,
                                    value=0.1,
                                    step=0.05,
                                    label="Min Duration (seconds)"
                                )
                            with gr.Column():
                                chop_max_duration = gr.Slider(
                                    minimum=0.5,
                                    maximum=10.0,
                                    value=3.0,
                                    step=0.5,
                                    label="Max Duration (seconds)"
                                )
                        
                        with gr.Row():
                            chop_btn = gr.Button("✂️ GENERATE CHOPS", variant="primary", size="lg")
                        
                        with gr.Row():
                            chop_output = gr.File(label="Generated Chops", file_count="multiple")
                            chop_status = gr.Textbox(label="Status", lines=3, interactive=False)
                        
                        def chop_wrapper(audio, mode, threshold, min_dur, max_dur):
                            if not audio:
                                return None, "❌ [ERROR] No audio file provided"
                            try:
                                chops = generate_vocal_chops(audio, mode, min_dur, max_dur, threshold)
                                return chops, f"✅ [SUCCESS] Generated {len(chops)} vocal chops\nSaved to output/chops/"
                            except Exception as e:
                                return None, f"❌ [ERROR] {str(e)}"
                        
                        chop_btn.click(
                            fn=chop_wrapper,
                            inputs=[chop_audio, chop_mode, chop_threshold, chop_min_duration, chop_max_duration],
                            outputs=[chop_output, chop_status]
                        )
                    
                    # ==================== PHASE 2: MIDI EXTRACTION ====================
                    with gr.Tab("PHASE 2: MIDI"):
                        gr.HTML('<div class="forge-card-header">2.5 AUDIO INPUT</div>')
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                midi_audio = gr.Audio(
                                    label="Upload Melodic Audio (vocals, instruments, or stems)", 
                                    type="filepath"
                                )
                        
                        gr.HTML('<div class="forge-card-header">2.6 MIDI EXTRACTION</div>')
                        
                        gr.Markdown("""
                        **MIDI extraction works best with:**
                        - Monophonic audio (single notes, not chords)
                        - Clean, isolated instruments or vocals
                        - Clear melodic content
                        """)
                        
                        with gr.Row():
                            midi_btn = gr.Button("🎹 EXTRACT MIDI", variant="primary", size="lg")
                        
                        with gr.Row():
                            midi_output = gr.File(label="Generated MIDI File")
                            midi_status = gr.Textbox(label="Status", lines=3, interactive=False)
                        
                        def midi_wrapper(audio):
                            if not audio:
                                return None, "❌ [ERROR] No audio file provided"
                            try:
                                midi_file = extract_midi(audio)
                                return midi_file, f"✅ [SUCCESS] MIDI extracted\nSaved to: {midi_file}"
                            except Exception as e:
                                return None, f"❌ [ERROR] {str(e)}"
                        
                        midi_btn.click(
                            fn=midi_wrapper,
                            inputs=[midi_audio],
                            outputs=[midi_output, midi_status]
                        )
                    
                    # ==================== PHASE 2: DRUM ONE-SHOTS ====================
                    with gr.Tab("PHASE 2: DRUM ONE-SHOTS"):
                        gr.HTML('<div class="forge-card-header">2.7 AUDIO INPUT</div>')
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                drum_audio = gr.Audio(
                                    label="Upload Drum Audio (drums stem from Phase 1 recommended)", 
                                    type="filepath"
                                )
                        
                        gr.HTML('<div class="forge-card-header">2.8 ONE-SHOT SETTINGS</div>')
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                drum_min_duration = gr.Slider(
                                    minimum=0.01,
                                    maximum=0.5,
                                    value=0.05,
                                    step=0.01,
                                    label="Min Duration (seconds)",
                                    info="Shorter = tighter one-shots"
                                )
                            with gr.Column():
                                drum_max_duration = gr.Slider(
                                    minimum=0.1,
                                    maximum=3.0,
                                    value=1.0,
                                    step=0.1,
                                    label="Max Duration (seconds)"
                                )
                        
                        with gr.Row():
                            drum_btn = gr.Button("🥁 EXTRACT ONE-SHOTS", variant="primary", size="lg")
                        
                        with gr.Row():
                            drum_output = gr.File(label="Generated Drum One-Shots", file_count="multiple")
                            drum_status = gr.Textbox(label="Status", lines=3, interactive=False)
                        
                        def drum_wrapper(audio, min_dur, max_dur):
                            if not audio:
                                return None, "❌ [ERROR] No audio file provided"
                            try:
                                oneshots = generate_drum_oneshots(audio, min_dur, max_dur)
                                return oneshots, f"✅ [SUCCESS] Extracted {len(oneshots)} drum one-shots\nSaved to output/drums/"
                            except Exception as e:
                                return None, f"❌ [ERROR] {str(e)}"
                        
                        drum_btn.click(
                            fn=drum_wrapper,
                            inputs=[drum_audio, drum_min_duration, drum_max_duration],
                            outputs=[drum_output, drum_status]
                        )
                    
                    # ==================== PHASE 3: VIDEO RENDERING ====================
                    with gr.Tab("PHASE 3: VIDEO"):
                        gr.HTML('<div class="forge-card-header">3.1 AUDIO INPUT</div>')
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                video_audio = gr.Audio(
                                    label="Upload Audio File for Video", 
                                    type="filepath"
                                )
                        
                        gr.HTML('<div class="forge-card-header">3.2 VIDEO SETTINGS</div>')
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                video_aspect_ratio = gr.Dropdown(
                                    choices=['16:9', '4:3', '1:1', '9:16'],
                                    value='16:9',
                                    label="Aspect Ratio",
                                    info="16:9=YouTube, 1:1=Instagram, 9:16=TikTok"
                                )
                            with gr.Column():
                                video_viz_type = gr.Dropdown(
                                    choices=['waveform', 'spectrum', 'both'],
                                    value='waveform',
                                    label="Visualization Type"
                                )
                        
                        with gr.Row():
                            video_btn = gr.Button("🎬 RENDER VIDEO", variant="primary", size="lg")
                        
                        with gr.Row():
                            video_output = gr.File(label="Generated Video")
                            video_status = gr.Textbox(label="Status", lines=3, interactive=False)
                        
                        def video_wrapper(audio, aspect_ratio, viz_type):
                            if not audio:
                                return None, "❌ [ERROR] No audio file provided"
                            try:
                                video_file = render_video(audio, aspect_ratio, viz_type)
                                return video_file, f"✅ [SUCCESS] Video rendered\nSaved to: {video_file}"
                            except Exception as e:
                                return None, f"❌ [ERROR] {str(e)}"
                        
                        video_btn.click(
                            fn=video_wrapper,
                            inputs=[video_audio, video_aspect_ratio, video_viz_type],
                            outputs=[video_output, video_status]
                        )
                    
                    # ==================== FEEDBACK ====================
                    with gr.Tab("FEEDBACK"):
                        gr.HTML('<div class="forge-card-header">USER FEEDBACK</div>')
                        
                        gr.Markdown("**Help us improve FORGE by sharing your experience**")
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                feedback_comments = gr.Textbox(
                                    label="Comments",
                                    placeholder="Share your thoughts, suggestions, or report issues...",
                                    lines=6
                                )
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                feedback_role = gr.Dropdown(
                                    choices=['Producer', 'DJ', 'Sound Designer', 'Musician', 'Hobbyist', 'Other'],
                                    label="Your Role",
                                    value='Producer'
                                )
                            with gr.Column():
                                feedback_usage = gr.Dropdown(
                                    choices=['Daily', 'Weekly', 'Monthly', 'First Time', 'Rarely'],
                                    label="Usage Frequency",
                                    value='First Time'
                                )
                        
                        with gr.Row(elem_classes="forge-card"):
                            feedback_email = gr.Textbox(
                                label="Contact Email (optional)",
                                placeholder="your@email.com"
                            )
                        
                        with gr.Row():
                            feedback_btn = gr.Button("📤 SUBMIT FEEDBACK", variant="primary", size="lg")
                        
                        with gr.Row():
                            feedback_status = gr.Textbox(label="Status", lines=3, interactive=False)
                        
                        def feedback_wrapper(comments, role, usage, email):
                            try:
                                if not comments or len(comments.strip()) < 10:
                                    return "❌ [ERROR] Please provide detailed feedback (minimum 10 characters)"
                                
                                # Call the existing save_feedback function
                                # Note: save_feedback expects (feature, rating, comments, email)
                                # We'll use role as the feature and add usage to comments
                                feature = role if role else "General Feedback"
                                full_comments = f"{comments}\n\nUsage: {usage}" if usage else comments
                                rating = 5  # Default rating since Phase 4 doesn't have rating slider
                                
                                result = save_feedback(feature, rating, full_comments, email)
                                return result
                                
                            except Exception as e:
                                return f"❌ [ERROR] {str(e)}"
                        
                        feedback_btn.click(
                            fn=feedback_wrapper,
                            inputs=[feedback_comments, feedback_role, feedback_usage, feedback_email],
                            outputs=[feedback_status]
                        )
                    
                    # ==================== BATCH PROCESSING ====================
                    with gr.Tab("BATCH PROCESSING"):
                        gr.HTML('<div class="forge-card-header">BATCH PROCESSING</div>')
                        
                        gr.Markdown("**Process multiple files at once**")
                        gr.Markdown("*Upload multiple audio files and apply operations in batch*")
                        
                        with gr.Row(elem_classes="forge-card"):
                            with gr.Column():
                                batch_files = gr.File(
                                    label="Upload Audio Files",
                                    file_count="multiple",
                                    file_types=["audio"]
                                )
                        
                        gr.HTML('<div class="forge-card-header">SELECT OPERATION</div>')
                        
                        with gr.Tabs():
                            # Batch Stem Separation
                            with gr.Tab("Stem Separation"):
                                gr.Markdown("**Batch separate stems from multiple audio files**")
                                
                                with gr.Row():
                                    batch_stem_model = gr.Dropdown(
                                        choices=Config.DEMUCS_MODELS,
                                        value='htdemucs',
                                        label="Demucs Model"
                                    )
                                with gr.Row():
                                    batch_stem_cache = gr.Checkbox(
                                        label="Use cache",
                                        value=True
                                    )
                                with gr.Row():
                                    batch_stem_btn = gr.Button(
                                        "🚀 Run Batch Stem Separation",
                                        variant="primary"
                                    )
                                with gr.Row():
                                    batch_stem_output = gr.Textbox(
                                        label="Results",
                                        lines=10,
                                        interactive=False
                                    )
                            
                            # Batch Loop Extraction
                            with gr.Tab("Loop Extraction"):
                                gr.Markdown("**Batch extract loops from multiple audio files**")
                                
                                with gr.Row():
                                    batch_loop_duration = gr.Slider(
                                        minimum=1.0,
                                        maximum=16.0,
                                        value=4.0,
                                        label="Loop Duration (seconds)"
                                    )
                                with gr.Row():
                                    batch_loop_aperture = gr.Slider(
                                        minimum=0.0,
                                        maximum=1.0,
                                        value=0.5,
                                        label="Aperture"
                                    )
                                with gr.Row():
                                    batch_loop_num = gr.Slider(
                                        minimum=1,
                                        maximum=20,
                                        value=5,
                                        step=1,
                                        label="Loops per file"
                                    )
                                with gr.Row():
                                    batch_loop_btn = gr.Button(
                                        "🚀 Run Batch Loop Extraction",
                                        variant="primary"
                                    )
                                with gr.Row():
                                    batch_loop_output = gr.Textbox(
                                        label="Results",
                                        lines=10,
                                        interactive=False
                                    )
                            
                            # Batch Vocal Chops
                            with gr.Tab("Vocal Chops"):
                                gr.Markdown("**Batch generate vocal chops from multiple files**")
                                
                                with gr.Row():
                                    batch_chop_mode = gr.Dropdown(
                                        choices=['silence', 'onset', 'hybrid'],
                                        value='onset',
                                        label="Detection Mode"
                                    )
                                with gr.Row():
                                    batch_chop_min = gr.Slider(
                                        minimum=0.05,
                                        maximum=2.0,
                                        value=0.1,
                                        label="Min Duration (s)"
                                    )
                                    batch_chop_max = gr.Slider(
                                        minimum=0.1,
                                        maximum=5.0,
                                        value=2.0,
                                        label="Max Duration (s)"
                                    )
                                with gr.Row():
                                    batch_chop_threshold = gr.Slider(
                                        minimum=0.0,
                                        maximum=1.0,
                                        value=0.3,
                                        label="Threshold"
                                    )
                                with gr.Row():
                                    batch_chop_btn = gr.Button(
                                        "🚀 Run Batch Chop Generation",
                                        variant="primary"
                                    )
                                with gr.Row():
                                    batch_chop_output = gr.Textbox(
                                        label="Results",
                                        lines=10,
                                        interactive=False
                                    )
                            
                            # Batch MIDI Extraction
                            with gr.Tab("MIDI Extraction"):
                                gr.Markdown("**Batch extract MIDI from multiple audio files**")
                                
                                with gr.Row():
                                    batch_midi_btn = gr.Button(
                                        "🚀 Run Batch MIDI Extraction",
                                        variant="primary"
                                    )
                                with gr.Row():
                                    batch_midi_output = gr.Textbox(
                                        label="Results",
                                        lines=10,
                                        interactive=False
                                    )
                            
                            # Batch Drum One-Shots
                            with gr.Tab("Drum One-Shots"):
                                gr.Markdown("**Batch generate drum one-shots from multiple files**")
                                
                                with gr.Row():
                                    batch_drum_min = gr.Slider(
                                        minimum=0.01,
                                        maximum=0.5,
                                        value=0.05,
                                        label="Min Duration (s)"
                                    )
                                    batch_drum_max = gr.Slider(
                                        minimum=0.1,
                                        maximum=2.0,
                                        value=0.5,
                                        label="Max Duration (s)"
                                    )
                                with gr.Row():
                                    batch_drum_fadeout = gr.Checkbox(
                                        label="Apply fadeout",
                                        value=True
                                    )
                                with gr.Row():
                                    batch_drum_btn = gr.Button(
                                        "🚀 Run Batch Drum One-Shot Generation",
                                        variant="primary"
                                    )
                                with gr.Row():
                                    batch_drum_output = gr.Textbox(
                                        label="Results",
                                        lines=10,
                                        interactive=False
                                    )
                        
                        # Import batch processing functions
                        from batch_processor import (
                            batch_separate_stems,
                            batch_extract_loops,
                            batch_generate_chops,
                            batch_extract_midi,
                            batch_generate_drum_oneshots
                        )
                        
                        # Wire up batch operations
                        def get_file_paths(files):
                            if files is None:
                                return []
                            return [f.name for f in files]
                        
                        batch_stem_btn.click(
                            fn=lambda files, model, cache: batch_separate_stems(
                                get_file_paths(files), model, cache
                            ),
                            inputs=[batch_files, batch_stem_model, batch_stem_cache],
                            outputs=[batch_stem_output]
                        )
                        
                        batch_loop_btn.click(
                            fn=lambda files, duration, aperture, num: batch_extract_loops(
                                get_file_paths(files), duration, aperture, int(num)
                            ),
                            inputs=[batch_files, batch_loop_duration, batch_loop_aperture, batch_loop_num],
                            outputs=[batch_loop_output]
                        )
                        
                        batch_chop_btn.click(
                            fn=lambda files, mode, min_d, max_d, thresh: batch_generate_chops(
                                get_file_paths(files), mode, min_d, max_d, thresh
                            ),
                            inputs=[batch_files, batch_chop_mode, batch_chop_min, batch_chop_max, batch_chop_threshold],
                            outputs=[batch_chop_output]
                        )
                        
                        batch_midi_btn.click(
                            fn=lambda files: batch_extract_midi(get_file_paths(files)),
                            inputs=[batch_files],
                            outputs=[batch_midi_output]
                        )
                        
                        batch_drum_btn.click(
                            fn=lambda files, min_d, max_d, fadeout: batch_generate_drum_oneshots(
                                get_file_paths(files), min_d, max_d, fadeout
                            ),
                            inputs=[batch_files, batch_drum_min, batch_drum_max, batch_drum_fadeout],
                            outputs=[batch_drum_output]
                        )
            
            # Right side: Console + Session Output (30% width)
            with gr.Column(scale=3):
                # System Console
                gr.HTML('<div class="console-header">SYSTEM CONSOLE</div>')
                with gr.Column(elem_classes="system-console"):
                    console_output = gr.Textbox(
                        value="[SYSTEM] FORGE v1.1.5 initialized\n[SYSTEM] All modules loaded\n[SYSTEM] Ready for processing...",
                        label="",
                        lines=15,
                        interactive=False,
                        show_label=False,
                        elem_classes="system-console"
                    )
                
                # Session Output
                gr.HTML('<div class="session-header" style="margin-top: 20px;">SESSION OUTPUT</div>')
                with gr.Column(elem_classes="session-output"):
                    session_files = gr.Textbox(
                        value="No files generated yet.\n\nProcessed files will appear here.",
                        label="",
                        lines=8,
                        interactive=False,
                        show_label=False
                    )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("*FORGE v1 - Neural Audio Workstation | Built with ❤️ by NeuralWorkstation Team*")
    
    return app


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main entry point for FORGE v1 application.
    Sets up directories and launches the Gradio interface.
    """
    
    print("=" * 70)
    print("FORGE v1 - Neural Audio Workstation")
    print("=" * 70)
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    print("✅ Directories ready")
    
    # Create and launch Gradio app
    print("\n🚀 Launching Gradio interface...")
    app = create_gradio_interface()
    
    # Launch with share=False for local use, share=True for public link
    # Disable SSR mode to fix "Could not get API info" errors in Gradio 5.x
    # In Gradio 6.x, theme and css are passed to launch() instead of Blocks()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        ssr_mode=False,
        theme=gr.themes.Base(),
        css=CUSTOM_CSS
    )


if __name__ == "__main__":
    main()

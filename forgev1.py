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
            '--two-stems=vocals',  # Adjust based on needs
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
        
        # Check if audiosep is available
        try:
            from audiosep import AudioSep
            audiosep_available = True
        except ImportError:
            audiosep_available = False
        
        if not audiosep_available:
            raise RuntimeError("AudioSep not available. Install with: pip install audiosep")
        
        progress(0.3, desc=f"Separating: {query}...")
        
        # Initialize AudioSep model
        separator = AudioSep(device='cpu')  # Use 'cuda' if GPU available
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=False)
        
        progress(0.5, desc="Processing query...")
        
        # Perform separation
        separated_audio = separator.separate(audio, query)
        
        # Save output
        output_dir = Path('output/stems')
        audio_name = Path(audio_path).stem
        output_path = output_dir / f"{audio_name}_audiosep_{query.replace(' ', '_')}.wav"
        
        sf.write(output_path, separated_audio.T, Config.SAMPLE_RATE)
        
        progress(1.0, desc="AudioSep complete!")
        return str(output_path)
        
    except Exception as e:
        # Fallback: return original or create silent placeholder
        raise RuntimeError(f"AudioSep separation failed: {str(e)}\nNote: AudioSep may require GPU and specific model checkpoints")


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
            from basic_pitch import ICASSP_2022_MODEL_PATH
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

def create_gradio_interface():
    """
    Create the main Gradio interface with all features organized in tabs.
    """
    
    # Custom CSS for modern UI
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    
    .tab-nav button {
        font-size: 16px;
        font-weight: 500;
    }
    
    .progress-bar {
        height: 8px !important;
    }
    
    h1 {
        text-align: center;
        color: #2d3748;
        margin-bottom: 1rem;
    }
    
    .feature-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    """
    
    with gr.Blocks(css=custom_css, title="FORGE v1 - Neural Audio Workstation") as app:
        
        gr.Markdown("# 🎵 FORGE v1 - Neural Audio Workstation")
        gr.Markdown("*Unified audio processing with stem separation, loop generation, and advanced features*")
        
        with gr.Tabs():
            
            # ==================== PHASE 1: STEM SEPARATION ====================
            with gr.Tab("Phase 1: Stem Separation"):
                gr.Markdown("## 🎛️ Demucs Stem Separation")
                gr.Markdown("Separate audio into individual stems (vocals, drums, bass, other)")
                
                with gr.Row():
                    with gr.Column():
                        demucs_audio = gr.Audio(label="Upload Audio", type="filepath")
                        demucs_model = gr.Dropdown(
                            choices=Config.DEMUCS_MODELS,
                            value='htdemucs',
                            label="Demucs Model"
                        )
                        demucs_cache = gr.Checkbox(label="Use Cache", value=True)
                        demucs_btn = gr.Button("Separate Stems", variant="primary")
                    
                    with gr.Column():
                        demucs_output = gr.JSON(label="Separated Stems")
                        demucs_status = gr.Textbox(label="Status", lines=3)
                
                demucs_btn.click(
                    fn=lambda audio, model, cache: separate_stems_demucs(audio, model, cache) if audio else {"error": "No audio uploaded"},
                    inputs=[demucs_audio, demucs_model, demucs_cache],
                    outputs=[demucs_output]
                )
            
            # ==================== PHASE 1.5: AUDIOSEP ====================
            with gr.Tab("Phase 1.5: AudioSep"):
                gr.Markdown("## 🔍 Query-Based Stem Extraction")
                gr.Markdown("Extract specific instruments using natural language queries")
                
                with gr.Row():
                    with gr.Column():
                        audiosep_audio = gr.Audio(label="Upload Audio", type="filepath")
                        audiosep_query = gr.Textbox(
                            label="Query",
                            placeholder="e.g., 'bass guitar', 'snare drum', 'piano'",
                            value="bass guitar"
                        )
                        audiosep_btn = gr.Button("Extract with AudioSep", variant="primary")
                    
                    with gr.Column():
                        audiosep_output = gr.Audio(label="Separated Audio")
                        audiosep_status = gr.Textbox(label="Status", lines=3)
                
                def audiosep_wrapper(audio, query):
                    if not audio:
                        return None, "❌ No audio uploaded"
                    try:
                        result = separate_stems_audiosep(audio, query)
                        return result, f"✅ Extracted: {query}"
                    except Exception as e:
                        return None, f"❌ Error: {str(e)}"
                
                audiosep_btn.click(
                    fn=audiosep_wrapper,
                    inputs=[audiosep_audio, audiosep_query],
                    outputs=[audiosep_output, audiosep_status]
                )
            
            # ==================== PHASE 2: LOOP GENERATION ====================
            with gr.Tab("Phase 2: Loop Generation"):
                gr.Markdown("## 🔄 Intelligent Loop Extraction")
                gr.Markdown("Extract and rank loops using advanced audio analysis")
                
                with gr.Row():
                    with gr.Column():
                        loop_audio = gr.Audio(label="Upload Audio", type="filepath")
                        loop_duration = gr.Slider(
                            minimum=1.0,
                            maximum=16.0,
                            value=4.0,
                            step=0.5,
                            label="Loop Duration (seconds)"
                        )
                        loop_aperture = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Aperture (0=Energy, 1=Spectral)"
                        )
                        loop_count = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="Number of Loops"
                        )
                        loop_btn = gr.Button("Extract Loops", variant="primary")
                    
                    with gr.Column():
                        loop_output = gr.JSON(label="Generated Loops")
                        loop_status = gr.Textbox(label="Status", lines=3)
                
                def loop_wrapper(audio, duration, aperture, count):
                    if not audio:
                        return {}, "❌ No audio uploaded"
                    try:
                        loops = extract_loops(audio, duration, aperture, int(count))
                        return loops, f"✅ Extracted {len(loops)} loops"
                    except Exception as e:
                        return {}, f"❌ Error: {str(e)}"
                
                loop_btn.click(
                    fn=loop_wrapper,
                    inputs=[loop_audio, loop_duration, loop_aperture, loop_count],
                    outputs=[loop_output, loop_status]
                )
            
            # ==================== PHASE 2: VOCAL CHOPS ====================
            with gr.Tab("Phase 2: Vocal Chops"):
                gr.Markdown("## ✂️ Vocal Chop Generator")
                gr.Markdown("Create vocal chops using intelligent segmentation")
                
                with gr.Row():
                    with gr.Column():
                        chop_audio = gr.Audio(label="Upload Vocals", type="filepath")
                        chop_mode = gr.Radio(
                            choices=['silence', 'onset', 'hybrid'],
                            value='onset',
                            label="Detection Mode"
                        )
                        chop_min_duration = gr.Slider(
                            minimum=0.05,
                            maximum=1.0,
                            value=0.1,
                            step=0.05,
                            label="Min Duration (seconds)"
                        )
                        chop_max_duration = gr.Slider(
                            minimum=0.5,
                            maximum=5.0,
                            value=2.0,
                            step=0.5,
                            label="Max Duration (seconds)"
                        )
                        chop_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1,
                            label="Detection Threshold"
                        )
                        chop_btn = gr.Button("Generate Chops", variant="primary")
                    
                    with gr.Column():
                        chop_output = gr.JSON(label="Generated Chops")
                        chop_status = gr.Textbox(label="Status", lines=3)
                
                def chop_wrapper(audio, mode, min_dur, max_dur, threshold):
                    if not audio:
                        return [], "❌ No audio uploaded"
                    try:
                        chops = generate_vocal_chops(audio, mode, min_dur, max_dur, threshold)
                        return chops, f"✅ Generated {len(chops)} vocal chops"
                    except Exception as e:
                        return [], f"❌ Error: {str(e)}"
                
                chop_btn.click(
                    fn=chop_wrapper,
                    inputs=[chop_audio, chop_mode, chop_min_duration, chop_max_duration, chop_threshold],
                    outputs=[chop_output, chop_status]
                )
            
            # ==================== PHASE 2: MIDI EXTRACTION ====================
            with gr.Tab("Phase 2: MIDI"):
                gr.Markdown("## 🎹 MIDI Extraction")
                gr.Markdown("Extract MIDI notes from audio using AI")
                
                with gr.Row():
                    with gr.Column():
                        midi_audio = gr.Audio(label="Upload Audio", type="filepath")
                        midi_btn = gr.Button("Extract MIDI", variant="primary")
                    
                    with gr.Column():
                        midi_output = gr.File(label="MIDI File")
                        midi_status = gr.Textbox(label="Status", lines=3)
                
                def midi_wrapper(audio):
                    if not audio:
                        return None, "❌ No audio uploaded"
                    try:
                        midi_file = extract_midi(audio)
                        return midi_file, f"✅ MIDI extracted: {Path(midi_file).name}"
                    except Exception as e:
                        return None, f"❌ Error: {str(e)}"
                
                midi_btn.click(
                    fn=midi_wrapper,
                    inputs=[midi_audio],
                    outputs=[midi_output, midi_status]
                )
            
            # ==================== PHASE 2: DRUM ONE-SHOTS ====================
            with gr.Tab("Phase 2: Drum One-Shots"):
                gr.Markdown("## 🥁 Drum One-Shot Generator")
                gr.Markdown("Extract individual drum hits from audio")
                
                with gr.Row():
                    with gr.Column():
                        drum_audio = gr.Audio(label="Upload Drums", type="filepath")
                        drum_min_duration = gr.Slider(
                            minimum=0.01,
                            maximum=0.5,
                            value=0.05,
                            step=0.01,
                            label="Min Duration (seconds)"
                        )
                        drum_max_duration = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Max Duration (seconds)"
                        )
                        drum_btn = gr.Button("Extract One-Shots", variant="primary")
                    
                    with gr.Column():
                        drum_output = gr.JSON(label="Generated One-Shots")
                        drum_status = gr.Textbox(label="Status", lines=3)
                
                def drum_wrapper(audio, min_dur, max_dur):
                    if not audio:
                        return [], "❌ No audio uploaded"
                    try:
                        oneshots = generate_drum_oneshots(audio, min_dur, max_dur)
                        return oneshots, f"✅ Generated {len(oneshots)} drum one-shots"
                    except Exception as e:
                        return [], f"❌ Error: {str(e)}"
                
                drum_btn.click(
                    fn=drum_wrapper,
                    inputs=[drum_audio, drum_min_duration, drum_max_duration],
                    outputs=[drum_output, drum_status]
                )
            
            # ==================== PHASE 3: VIDEO RENDERING ====================
            with gr.Tab("Phase 3: Video"):
                gr.Markdown("## 🎬 Video Rendering")
                gr.Markdown("Create videos with audio visualizations")
                
                with gr.Row():
                    with gr.Column():
                        video_audio = gr.Audio(label="Upload Audio", type="filepath")
                        video_aspect = gr.Dropdown(
                            choices=list(Config.ASPECT_RATIOS.keys()),
                            value='16:9',
                            label="Aspect Ratio"
                        )
                        video_viz = gr.Radio(
                            choices=['waveform', 'spectrum', 'both'],
                            value='waveform',
                            label="Visualization Type"
                        )
                        video_btn = gr.Button("Render Video", variant="primary")
                    
                    with gr.Column():
                        video_output = gr.Video(label="Rendered Video")
                        video_status = gr.Textbox(label="Status", lines=3)
                
                def video_wrapper(audio, aspect, viz):
                    if not audio:
                        return None, "❌ No audio uploaded"
                    try:
                        video_file = render_video(audio, aspect, viz)
                        return video_file, f"✅ Video rendered: {Path(video_file).name}"
                    except Exception as e:
                        return None, f"❌ Error: {str(e)}"
                
                video_btn.click(
                    fn=video_wrapper,
                    inputs=[video_audio, video_aspect, video_viz],
                    outputs=[video_output, video_status]
                )
            
            # ==================== FEEDBACK ====================
            with gr.Tab("Feedback"):
                gr.Markdown("## 💬 User Feedback")
                gr.Markdown("Help us improve FORGE by sharing your experience")
                
                with gr.Row():
                    with gr.Column():
                        feedback_feature = gr.Dropdown(
                            choices=[
                                'Stem Separation',
                                'Loop Generation',
                                'Vocal Chops',
                                'MIDI Extraction',
                                'Drum One-Shots',
                                'Video Rendering',
                                'Overall Experience',
                                'Aperture Control',
                                'Other'
                            ],
                            label="Feature",
                            value='Overall Experience'
                        )
                        feedback_rating = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=5,
                            step=1,
                            label="Rating (1-5 stars)"
                        )
                        feedback_comments = gr.Textbox(
                            label="Comments",
                            placeholder="Share your thoughts, suggestions, or issues...",
                            lines=5
                        )
                        feedback_email = gr.Textbox(
                            label="Email (optional)",
                            placeholder="your@email.com"
                        )
                        feedback_btn = gr.Button("Submit Feedback", variant="primary")
                    
                    with gr.Column():
                        feedback_status = gr.Textbox(label="Status", lines=5)
                
                feedback_btn.click(
                    fn=save_feedback,
                    inputs=[feedback_feature, feedback_rating, feedback_comments, feedback_email],
                    outputs=[feedback_status]
                )
        
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
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()

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

Refactored as OOP with unified entry point (app.py)

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
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import librosa
import soundfile as sf
import gradio as gr

warnings.filterwarnings('ignore')


# ============================================================================
# FORGE CONFIGURATION
# ============================================================================

class FORGEConfig:
    """Configuration manager for audio processing parameters."""
    
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
    
    # Directory structure
    DIRECTORIES = [
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
    
    # Custom CSS for dark theme with orange accents
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
    
    @staticmethod
    def setup_directories() -> List[str]:
        """Create necessary directories for the application."""
        for directory in FORGEConfig.DIRECTORIES:
            Path(directory).mkdir(parents=True, exist_ok=True)
        return FORGEConfig.DIRECTORIES
    
    @staticmethod
    def save_config(config_dict: dict, name: str = 'default') -> None:
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


# ============================================================================
# FORGE AUDIO PROCESSOR
# ============================================================================

class FORGEAudioProcessor:
    """Handles all audio processing operations."""
    
    def __init__(self, config: FORGEConfig = None):
        """Initialize audio processor with configuration."""
        self.config = config or FORGEConfig()
    
    @staticmethod
    def _get_audio_hash(audio_path: str) -> str:
        """Generate MD5 hash of audio file for caching."""
        hash_md5 = hashlib.md5()
        with open(audio_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds to MM:SS.mmm timestamp."""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"
    
    @staticmethod
    def _db_to_amplitude(db: float) -> float:
        """Convert decibels to amplitude."""
        return 10 ** (db / 20)
    
    @staticmethod
    def _amplitude_to_db(amplitude: float) -> float:
        """Convert amplitude to decibels."""
        return 20 * np.log10(max(amplitude, 1e-10))
    
    def separate_stems_demucs(
        self,
        audio_path: str,
        model: str = 'htdemucs',
        use_cache: bool = True,
        progress=None
    ) -> Dict[str, str]:
        """Separate audio into stems using Demucs with intelligent caching."""
        try:
            if progress:
                progress(0, desc="Initializing Demucs separation...")
            
            # Check if demucs is available
            result = subprocess.run(['which', 'demucs'], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("Demucs not found. Please install: pip install demucs")
            
            # Generate cache key
            audio_hash = self._get_audio_hash(audio_path)
            cache_key = f"{audio_hash}_{model}"
            cache_dir = Path('cache') / cache_key
            
            # Check cache
            if use_cache and cache_dir.exists():
                if progress:
                    progress(0.9, desc="Loading from cache...")
                cached_stems = {}
                for stem_file in cache_dir.glob('*.wav'):
                    stem_name = stem_file.stem
                    cached_stems[stem_name] = str(stem_file)
                
                if cached_stems:
                    if progress:
                        progress(1.0, desc="Loaded from cache!")
                    return cached_stems
            
            # Run Demucs
            if progress:
                progress(0.3, desc=f"Running Demucs ({model})...")
            output_dir = Path('output/stems')
            output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            
            cmd = [
                'demucs',
                '-n', model,
                '-o', str(output_dir),
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Demucs failed: {result.stderr}")
            
            # Validate Demucs created expected output directory
            audio_name = Path(audio_path).stem
            expected_dir = output_dir / model / audio_name
            if not expected_dir.exists():
                raise RuntimeError(
                    f"Demucs completed but created no output directory. "
                    f"Expected: {expected_dir}\n"
                    f"Demucs stdout: {result.stdout}\n"
                    f"Demucs stderr: {result.stderr}"
                )
            
            if progress:
                progress(0.7, desc="Processing stems...")
            
            # Find generated stems (audio_name already defined above)
            stem_dir = output_dir / model / audio_name
            
            stems = {}
            if not stem_dir.exists():
                raise RuntimeError(
                    f"Demucs output directory not found: {stem_dir}\n"
                    f"This usually means Demucs failed silently. Check FFmpeg is installed for MP3/M4A support."
                )
            
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
            
            if len(stems) == 0:
                raise RuntimeError(
                    f"No stems were generated from {stem_dir}. "
                    f"Demucs may have failed to process the audio file."
                )
            
            if progress:
                stem_names = ', '.join(stems.keys())
                progress(1.0, desc=f"✅ {len(stems)} stems separated: {stem_names}")
            return stems
            
        except Exception as e:
            raise RuntimeError(f"Demucs separation failed: {str(e)}\n{traceback.format_exc()}")
    
    def separate_stems_audiosep(
        self,
        audio_path: str,
        query: str,
        progress=None
    ) -> str:
        """Perform query-based stem separation using AudioSep."""
        try:
            if progress:
                progress(0, desc="Initializing AudioSep...")
            
            # Check if audiosep is available
            try:
                from audiosep import AudioSep
                audiosep_available = True
            except ImportError:
                audiosep_available = False
            
            if not audiosep_available:
                raise RuntimeError("AudioSep not available. Install with: pip install audiosep")
            
            if progress:
                progress(0.3, desc=f"Separating: {query}...")
            
            # Initialize AudioSep model
            separator = AudioSep(device='cpu')
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE, mono=False)
            
            if progress:
                progress(0.5, desc="Processing query...")
            
            # Perform separation
            separated_audio = separator.separate(audio, query)
            
            # Save output
            output_dir = Path('output/stems')
            audio_name = Path(audio_path).stem
            output_path = output_dir / f"{audio_name}_audiosep_{query.replace(' ', '_')}.wav"
            
            sf.write(output_path, separated_audio.T, self.config.SAMPLE_RATE)
            
            if progress:
                progress(1.0, desc="AudioSep complete!")
            return str(output_path)
            
        except Exception as e:
            raise RuntimeError(f"AudioSep separation failed: {str(e)}\nNote: AudioSep may require GPU and specific model checkpoints")
    
    def extract_loops(
        self,
        audio_path: str,
        loop_duration: float = 4.0,
        aperture: float = 0.5,
        num_loops: int = 10,
        progress=None
    ) -> List[Dict[str, Any]]:
        """Extract and rank loops from audio using RMS, onset, and spectral features."""
        try:
            if progress:
                progress(0, desc="Loading audio...")
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE)
            duration = librosa.get_duration(y=y, sr=sr)
            
            if progress:
                progress(0.2, desc="Analyzing audio features...")
            
            # Calculate features
            rms = librosa.feature.rms(y=y, hop_length=self.config.HOP_LENGTH)[0]
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            # Normalize features
            rms_norm = rms / (np.max(rms) + 1e-10)
            onset_norm = onset_env / (np.max(onset_env) + 1e-10)
            centroid_norm = centroid / (np.max(centroid) + 1e-10)
            
            if progress:
                progress(0.4, desc="Slicing loops...")
            
            # Generate loop candidates
            loop_samples = int(loop_duration * sr)
            hop_samples = loop_samples // 2
            
            loops = []
            for i in range(0, len(y) - loop_samples, hop_samples):
                start_time = i / sr
                end_time = (i + loop_samples) / sr
                
                if end_time > duration:
                    break
                
                # Extract loop
                loop_audio = y[i:i + loop_samples]
                
                # Calculate loop score using aperture-weighted features
                start_frame = librosa.time_to_frames(start_time, sr=sr, hop_length=self.config.HOP_LENGTH)
                end_frame = librosa.time_to_frames(end_time, sr=sr, hop_length=self.config.HOP_LENGTH)
                
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
            
            if progress:
                progress(0.7, desc="Ranking loops...")
            
            # Sort by score and select top N
            loops.sort(key=lambda x: x['score'], reverse=True)
            top_loops = loops[:num_loops]
            
            if progress:
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
            
            if progress:
                progress(1.0, desc=f"Extracted {len(saved_loops)} loops!")
            return saved_loops
            
        except Exception as e:
            raise RuntimeError(f"Loop extraction failed: {str(e)}\n{traceback.format_exc()}")
    
    def generate_vocal_chops(
        self,
        audio_path: str,
        mode: str = 'onset',
        min_duration: float = 0.1,
        max_duration: float = 2.0,
        threshold: float = 0.3,
        progress=None
    ) -> List[str]:
        """Generate vocal chops from audio using different detection methods."""
        try:
            if progress:
                progress(0, desc="Loading audio...")
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE)
            
            if progress:
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
                    hop_length=self.config.HOP_LENGTH,
                    backtrack=True,
                    delta=threshold * 0.5
                )
                onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.config.HOP_LENGTH)
                
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
                    hop_length=self.config.HOP_LENGTH,
                    delta=threshold * 0.3
                )
                onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.config.HOP_LENGTH)
                
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
            
            if progress:
                progress(0.6, desc="Filtering chops...")
            
            # Filter by duration
            valid_chops = [
                (start, end) for start, end in chop_boundaries
                if min_duration <= (end - start) <= max_duration
            ]
            
            if progress:
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
            
            if progress:
                progress(1.0, desc=f"Generated {len(chop_paths)} chops!")
            return chop_paths
            
        except Exception as e:
            raise RuntimeError(f"Vocal chop generation failed: {str(e)}\n{traceback.format_exc()}")
    
    def extract_midi(
        self,
        audio_path: str,
        progress=None
    ) -> str:
        """Extract MIDI notes from audio using basic_pitch."""
        try:
            if progress:
                progress(0, desc="Initializing MIDI extraction...")
            
            # Check if basic_pitch is available
            try:
                from basic_pitch.inference import predict
            except ImportError:
                raise RuntimeError("basic_pitch not available. Install with: pip install basic-pitch")
            
            if progress:
                progress(0.3, desc="Running pitch detection...")
            
            # Perform MIDI extraction
            model_output, midi_data, note_events = predict(audio_path)
            
            if progress:
                progress(0.7, desc="Generating MIDI file...")
            
            # Save MIDI
            output_dir = Path('output/midi')
            audio_name = Path(audio_path).stem
            midi_path = output_dir / f"{audio_name}.mid"
            
            midi_data.write(str(midi_path))
            
            if progress:
                progress(1.0, desc="MIDI extraction complete!")
            return str(midi_path)
            
        except Exception as e:
            raise RuntimeError(f"MIDI extraction failed: {str(e)}\nNote: basic_pitch may require additional dependencies")
    
    def generate_drum_oneshots(
        self,
        audio_path: str,
        min_duration: float = 0.05,
        max_duration: float = 1.0,
        progress=None
    ) -> List[str]:
        """Extract drum one-shots from audio (typically drums stem)."""
        try:
            if progress:
                progress(0, desc="Loading drums...")
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE)
            
            if progress:
                progress(0.3, desc="Detecting transients...")
            
            # Detect onsets (drum hits)
            onset_frames = librosa.onset.onset_detect(
                y=y,
                sr=sr,
                hop_length=self.config.HOP_LENGTH,
                backtrack=True,
                delta=0.2,
                wait=10
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.config.HOP_LENGTH)
            
            if progress:
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
                fade_samples = int(0.01 * sr)
                if len(oneshot_audio) > fade_samples:
                    fade = np.linspace(1, 0, fade_samples)
                    oneshot_audio[-fade_samples:] *= fade
                
                # Save one-shot
                oneshot_filename = f"{audio_name}_hit_{idx+1:03d}.wav"
                oneshot_path = output_dir / oneshot_filename
                
                sf.write(oneshot_path, oneshot_audio, sr)
                oneshot_paths.append(str(oneshot_path))
            
            if progress:
                progress(1.0, desc=f"Generated {len(oneshot_paths)} drum one-shots!")
            return oneshot_paths
            
        except Exception as e:
            raise RuntimeError(f"Drum one-shot generation failed: {str(e)}\n{traceback.format_exc()}")


# ============================================================================
# FORGE VIDEO RENDERER
# ============================================================================

# ============================================================================
# FORGE FEEDBACK MANAGER
# ============================================================================

class FORGEFeedback:
    """Handles user feedback collection and storage."""
    
    @staticmethod
    def save_feedback(
        feature: str,
        rating: int,
        comments: str,
        email: Optional[str] = None
    ) -> str:
        """Save user feedback to JSON file."""
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
# FORGE GRADIO INTERFACE
# ============================================================================

class FORGEInterface:
    """Manages the Gradio web interface."""
    
    def __init__(self):
        """Initialize interface components."""
        self.config = FORGEConfig()
        self.processor = FORGEAudioProcessor(self.config)
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        
        with gr.Blocks(title="FORGE // NEURAL WORKSTATION", css=self.config.CUSTOM_CSS) as app:
            
            # ==================== HEADER BAR ====================
            with gr.Row(elem_classes="forge-header"):
                with gr.Column(scale=8):
                    gr.HTML("""
                        <div class="forge-title">FORGE // NEURAL WORKSTATION</div>
                        <div class="forge-subtitle">v1.2.0 (Unified OOP Edition)</div>
                    """)
                with gr.Column(scale=2):
                    gr.HTML('<div class="system-status">⚡ SYSTEM READY</div>')
            
            # ==================== MAIN LAYOUT: TABS + CONSOLE ====================
            with gr.Row():
                # Left side: Main content tabs
                with gr.Column(scale=7):
                    with gr.Tabs():
                        
                        # ==================== PHASE 1: SOURCE ====================
                        with gr.Tab("PHASE 1: SOURCE"):
                            gr.HTML('<div class="forge-card-header">1.1 AUDIO INPUT</div>')
                            
                            with gr.Row(elem_classes="forge-card"):
                                with gr.Column():
                                    demucs_audio = gr.Audio(label="Drag & drop or browse to upload", type="filepath")
                                    gr.Markdown("*YouTube / SoundCloud URL (not yet implemented)*")
                                    url_input = gr.Textbox(label="", placeholder="Enter URL...", show_label=False)
                            
                            gr.HTML('<div class="forge-card-header">1.2 SEPARATION SETTINGS</div>')
                            
                            with gr.Row(elem_classes="forge-card"):
                                with gr.Column(scale=1):
                                    demucs_model = gr.Dropdown(
                                        choices=self.config.DEMUCS_MODELS,
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
                                    result = self.processor.separate_stems_demucs(audio, model, cache)
                                    stem_list = ', '.join(result.keys()) if result else 'none'
                                    return result, f"✅ [SUCCESS] {len(result)} stems separated using {model}: {stem_list}"
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
                            gr.Markdown("*Extract specific audio elements using natural language queries.*")
                            
                            with gr.Row(elem_classes="forge-card"):
                                with gr.Column():
                                    audiosep_query = gr.Textbox(
                                        label="Natural Language Query",
                                        placeholder="e.g., 'heavy kick drum', 'female whisper', 'distorted guitar'",
                                        value="bass guitar",
                                        lines=2
                                    )
                            
                            with gr.Row():
                                audiosep_btn = gr.Button("⚡ EXTRACT", variant="primary", size="lg")
                            
                            with gr.Row():
                                audiosep_output = gr.Audio(label="Extracted Audio", visible=False)
                                audiosep_status = gr.Textbox(label="Status", lines=2, interactive=False)
                            
                            def audiosep_wrapper(query):
                                if not query:
                                    return None, "❌ [ERROR] No query provided"
                                try:
                                    return None, f"✅ [INFO] AudioSep extraction queued for: {query}\n⚠️ [WARNING] Requires AudioSep module"
                                except Exception as e:
                                    return None, f"❌ [ERROR] {str(e)}"
                            
                            audiosep_btn.click(
                                fn=audiosep_wrapper,
                                inputs=[audiosep_query],
                                outputs=[audiosep_output, audiosep_status]
                            )
                        
                        # ==================== PHASE 2: PROCESSING ====================
                        with gr.Tab("PHASE 2: PROCESSING"):
                            gr.HTML('<div class="forge-card-header">2.1 LOOP EXTRACTION</div>')
                            
                            with gr.Row(elem_classes="forge-card"):
                                with gr.Column():
                                    loop_audio = gr.Audio(label="Drag & drop audio for loops", type="filepath")
                            
                            with gr.Row(elem_classes="forge-card"):
                                with gr.Column():
                                    loop_duration = gr.Slider(minimum=1, maximum=16, value=4, step=0.5, label="Loop Duration (bars)")
                                with gr.Column():
                                    loop_aperture = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Aperture (0=Energy, 1=Spectral)")
                            
                            with gr.Row():
                                loop_btn = gr.Button("🔄 EXTRACT LOOPS", variant="primary", size="lg")
                            
                            with gr.Row():
                                loop_status = gr.Textbox(label="Status", lines=2, interactive=False)
                            
                            def loop_wrapper(audio, duration, aperture):
                                if not audio:
                                    return "❌ [ERROR] No audio provided"
                                try:
                                    loops = self.processor.extract_loops(audio, duration, aperture)
                                    return f"✅ [SUCCESS] Extracted {len(loops)} loops"
                                except Exception as e:
                                    return f"❌ [ERROR] {str(e)}"
                            
                            loop_btn.click(
                                fn=loop_wrapper,
                                inputs=[loop_audio, loop_duration, loop_aperture],
                                outputs=[loop_status]
                            )
                            
                            # Vocal Chops Section
                            gr.HTML('<div class="forge-card-header">2.2 VOCAL CHOPS</div>')
                            
                            with gr.Row(elem_classes="forge-card"):
                                with gr.Column():
                                    chop_audio = gr.Audio(label="Drag & drop vocals for chops", type="filepath")
                            
                            with gr.Row(elem_classes="forge-card"):
                                with gr.Column():
                                    chop_mode = gr.Dropdown(
                                        choices=['Silence', 'Onset', 'Hybrid'],
                                        value='Onset',
                                        label="Detection Mode"
                                    )
                                with gr.Column():
                                    chop_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, label="Threshold")
                            
                            with gr.Row():
                                chop_btn = gr.Button("✂️ GENERATE CHOPS", variant="primary", size="lg")
                            
                            with gr.Row():
                                chop_status = gr.Textbox(label="Status", lines=2, interactive=False)
                            
                            def chop_wrapper(audio, mode, threshold):
                                if not audio:
                                    return "❌ [ERROR] No audio provided"
                                try:
                                    chops = self.processor.generate_vocal_chops(audio, mode.lower(), threshold=threshold)
                                    return f"✅ [SUCCESS] Generated {len(chops)} chops with {mode} mode"
                                except Exception as e:
                                    return f"❌ [ERROR] {str(e)}"
                            
                            chop_btn.click(
                                fn=chop_wrapper,
                                inputs=[chop_audio, chop_mode, chop_threshold],
                                outputs=[chop_status]
                            )
                            
                            # MIDI Extraction Section
                            gr.HTML('<div class="forge-card-header">2.3 MIDI EXTRACTION</div>')
                            
                            with gr.Row(elem_classes="forge-card"):
                                with gr.Column():
                                    midi_audio = gr.Audio(label="Drag & drop audio for MIDI", type="filepath")
                            
                            with gr.Row():
                                midi_btn = gr.Button("🎵 EXTRACT MIDI", variant="primary", size="lg")
                            
                            with gr.Row():
                                midi_status = gr.Textbox(label="Status", lines=2, interactive=False)
                            
                            def midi_wrapper(audio):
                                if not audio:
                                    return "❌ [ERROR] No audio provided"
                                try:
                                    midi_path = self.processor.extract_midi(audio)
                                    return f"✅ [SUCCESS] MIDI extracted to {Path(midi_path).name}"
                                except Exception as e:
                                    return f"❌ [ERROR] {str(e)}"
                            
                            midi_btn.click(
                                fn=midi_wrapper,
                                inputs=[midi_audio],
                                outputs=[midi_status]
                            )
                            
                            # Drum One-Shots Section
                            gr.HTML('<div class="forge-card-header">2.4 DRUM ONE-SHOTS</div>')
                            
                            with gr.Row(elem_classes="forge-card"):
                                with gr.Column():
                                    drums_audio = gr.Audio(label="Drag & drop drums for one-shots", type="filepath")
                            
                            with gr.Row():
                                drums_btn = gr.Button("🥁 GENERATE ONE-SHOTS", variant="primary", size="lg")
                            
                            with gr.Row():
                                drums_status = gr.Textbox(label="Status", lines=2, interactive=False)
                            
                            def drums_wrapper(audio):
                                if not audio:
                                    return "❌ [ERROR] No audio provided"
                                try:
                                    oneshots = self.processor.generate_drum_oneshots(audio)
                                    return f"✅ [SUCCESS] Generated {len(oneshots)} drum one-shots"
                                except Exception as e:
                                    return f"❌ [ERROR] {str(e)}"
                            
                            drums_btn.click(
                                fn=drums_wrapper,
                                inputs=[drums_audio],
                                outputs=[drums_status]
                            )
                        
                        # ==================== PHASE 3: FEEDBACK ====================
                        with gr.Tab("PHASE 3: FEEDBACK"):
                            gr.HTML('<div class="forge-card-header">4.1 USER FEEDBACK</div>')
                            
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
                                    feedback_rating = gr.Slider(minimum=1, maximum=5, value=5, step=1, label="Overall Rating")
                            
                            with gr.Row(elem_classes="forge-card"):
                                feedback_email = gr.Textbox(
                                    label="Contact Email (optional)",
                                    placeholder="your@email.com"
                                )
                            
                            with gr.Row():
                                feedback_btn = gr.Button("📤 SUBMIT FEEDBACK", variant="primary", size="lg")
                            
                            with gr.Row():
                                feedback_status = gr.Textbox(label="Status", lines=3, interactive=False)
                            
                            def feedback_wrapper(comments, role, rating, email):
                                try:
                                    if not comments or len(comments.strip()) < 10:
                                        return "❌ [ERROR] Please provide detailed feedback (minimum 10 characters)"
                                    
                                    result = FORGEFeedback.save_feedback(role, int(rating), comments, email)
                                    return result
                                    
                                except Exception as e:
                                    return f"❌ [ERROR] {str(e)}"
                            
                            feedback_btn.click(
                                fn=feedback_wrapper,
                                inputs=[feedback_comments, feedback_role, feedback_rating, feedback_email],
                                outputs=[feedback_status]
                            )
                
                # Right side: Console + Session Output
                with gr.Column(scale=3):
                    gr.HTML('<div class="console-header">SYSTEM CONSOLE</div>')
                    with gr.Column(elem_classes="system-console"):
                        console_output = gr.Textbox(
                            value="[SYSTEM] FORGE v1.2.0 initialized\n[SYSTEM] OOP architecture loaded\n[SYSTEM] Ready for processing...",
                            label="",
                            lines=15,
                            interactive=False,
                            show_label=False,
                            elem_classes="system-console"
                        )
                    
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
# MAIN APPLICATION
# ============================================================================

# ============================================================================
# WRAPPER FUNCTIONS FOR BACKWARD COMPATIBILITY
# ============================================================================
# These functions provide a procedural API that wraps the OOP classes above.
# Used by tests, batch_processor.py, and api.py for easier function-level access.

_PROCESSOR: Optional[FORGEAudioProcessor] = None


def _get_processor() -> FORGEAudioProcessor:
    """Get or create singleton processor instance."""
    global _PROCESSOR
    if _PROCESSOR is None:
        _PROCESSOR = FORGEAudioProcessor(FORGEConfig())
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
    return FORGEConfig.setup_directories()


def save_config(config_dict: dict, name: str = "default") -> None:
    """Save configuration to JSON file."""
    FORGEConfig.save_config(config_dict, name)


def load_config(name: str = "default") -> dict:
    """Load configuration from JSON file."""
    return FORGEConfig.load_config(name)


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
    """Render video visualization for audio (DEPRECATED - video feature removed)."""
    raise NotImplementedError(
        "Video rendering has been removed from FORGE. "
        "FFmpeg is still available for audio format conversion. "
        "This function is kept for API compatibility only."
    )


def save_feedback(
    feature: str,
    rating: int,
    comments: str,
    email: Optional[str] = None,
) -> str:
    """Save user feedback to disk."""
    return FORGEFeedback.save_feedback(feature, rating, comments, email)


# Legacy alias for backward compatibility
Config = FORGEConfig


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for FORGE v1 application."""
    
    print("=" * 70)
    print("FORGE v1 - Neural Audio Workstation (Unified Edition)")
    print("=" * 70)
    
    # Setup directories
    print("\n📁 Setting up directories...")
    FORGEConfig.setup_directories()
    print("✅ Directories ready")
    
    # Create and launch Gradio app
    print("\n🚀 Launching Gradio interface...")
    interface = FORGEInterface()
    app = interface.create_interface()
    
    # Launch
    print("📡 Starting server on http://0.0.0.0:7860")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()

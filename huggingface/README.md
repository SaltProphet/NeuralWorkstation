---
title: FORGE v1 - Neural Audio Workstation
emoji: 🎵
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 5.11.0
app_file: app.py
pinned: false
license: mit
python_version: "3.10"
---

# 🎵 FORGE v1 - Neural Audio Workstation

A unified, comprehensive audio processing workstation combining stem separation, loop extraction, vocal chopping, MIDI extraction, drum one-shot generation, and video rendering into a single, intuitive Gradio interface.

## Features

### Phase 1: Stem Separation
- **Demucs Integration**: Industry-leading stem separation with multiple model options
  - Support for htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra, and mdx_extra_q
  - Intelligent caching system using MD5 hashing for faster reprocessing
  - Separates audio into vocals, drums, bass, and other stems

### Phase 2: Audio Processing Tools

#### Loop Generation
- **Intelligent Loop Extraction**: AI-powered loop ranking system
  - RMS energy analysis
  - Onset detection for rhythmic content
  - Spectral centroid for tonal characteristics
  - **Aperture Control**: Dynamic weighting between energy (0.0) and spectral (1.0) features
  - Configurable loop duration (1-16 seconds)
  - Extract and rank up to 20 loops

#### Vocal Chop Generator
- **Three Detection Modes**:
  - **Silence**: Split based on quiet sections
  - **Onset**: Split based on transient detection
  - **Hybrid**: Combined approach for best results
- Configurable duration ranges and detection thresholds
- Perfect for creating sample packs and remixes

#### MIDI Extraction
- **AI-Powered Transcription**: Convert audio to MIDI using basic_pitch
- Extracts melodies, harmonies, and rhythms
- Compatible with any DAW

#### Drum One-Shot Generator
- **Transient Detection**: Automatically isolate individual drum hits
- Configurable duration ranges
- Apply fade-outs for clean samples
- Ideal for creating drum sample libraries

### Phase 3: Video Rendering
- **FFmpeg-Powered Visualization**: Create stunning visualizations
  - Multiple aspect ratios: 16:9, 4:3, 1:1, 9:16
  - Waveform, spectrum, or combined visualizations
  - Customizable colors and styles

## Usage

1. **Upload an audio file** (WAV, MP3, FLAC, etc.)
2. **Choose your processing phase**:
   - Phase 1: Separate stems using Demucs
   - Phase 2: Generate loops, chops, MIDI, or drum samples
   - Phase 3: Create video visualizations
3. **Download your processed files**

## Technical Details

- **Stem Separation**: Demucs 4.0+ with multiple model options
- **Audio Processing**: librosa, soundfile, scipy
- **Deep Learning**: PyTorch 2.0+
- **MIDI Extraction**: basic-pitch
- **Video Rendering**: FFmpeg
- **UI Framework**: Gradio 4.0+

## Performance Notes

- Processing time depends on audio length and chosen model
- GPU acceleration available for faster processing
- Caching system speeds up repeated operations on the same audio

## Known Issues & Fixes

✅ **"Could not get API info" Error - FIXED**: This Gradio 5.x compatibility issue has been resolved by setting `ssr_mode=False` in the launch configuration. The latest version includes this fix.

## License

MIT License - See LICENSE file for details

## Repository

Source code: [SaltProphet/NeuralWorkstation](https://github.com/SaltProphet/NeuralWorkstation)

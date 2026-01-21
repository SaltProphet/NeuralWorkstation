# GitHub Copilot Instructions for NeuralWorkstation

## Project Overview

**FORGE v1** is a comprehensive Neural Audio Workstation that combines audio processing, stem separation, loop extraction, vocal chopping, MIDI extraction, drum one-shot generation, and video rendering into a unified Gradio-based web interface.

## Tech Stack

- **Language**: Python 3.8+
- **UI Framework**: Gradio 4.0+
- **Audio Processing**: librosa, soundfile, scipy, numpy
- **Deep Learning**: PyTorch, torchaudio
- **Stem Separation**: Demucs 4.0+
- **MIDI Extraction**: basic-pitch
- **Video Processing**: FFmpeg (system dependency)
- **Optional**: AudioSep (advanced query-based separation)

## Project Structure

```
NeuralWorkstation/
├── forgev1.py          # Main application - all logic in single file
├── requirements.txt    # Python dependencies
├── README.md          # Comprehensive user documentation
├── LICENSE            # MIT License
├── .gitignore         # Git exclusions
├── runs/              # Processing runs metadata (generated)
├── cache/             # Cached stem separations (generated)
├── config/            # Configuration files (generated)
├── checkpoint/        # Model checkpoints for AudioSep (generated)
├── feedback/          # User feedback JSON files (generated)
└── output/            # All generated outputs (generated)
    ├── stems/         # Separated audio stems
    ├── loops/         # Extracted loops
    ├── chops/         # Vocal chops
    ├── midi/          # MIDI files
    ├── drums/         # Drum one-shots
    └── videos/        # Rendered videos
```

## Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install AudioSep for advanced features
pip install audiosep
```

### Running the Application
```bash
# Launch Gradio interface (opens on http://localhost:7860)
python forgev1.py
```

### Testing Audio Processing
```bash
# The application is primarily tested through the Gradio UI
# Manual testing involves uploading audio files and verifying outputs
```

## Code Style and Conventions

### Python Style
- Follow PEP 8 conventions with clear docstrings
- Use type hints for function signatures: `def function(param: str) -> Optional[Dict]:`
- Organize code into logical sections with clear comment headers
- Example from codebase:

```python
from typing import Dict

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
        model: Demucs model name (htdemucs, htdemucs_ft, etc.)
        use_cache: Whether to use cached results if available
        progress: Gradio progress callback for UI updates
    
    Returns:
        Dictionary mapping stem names to output file paths
    """
```

### Audio Processing Guidelines
- **Sample Rate**: Use 44.1kHz or 48kHz (Config.SAMPLE_RATE = 44100)
- **Caching**: Use MD5 hashing for audio file caching to avoid reprocessing
- **Error Handling**: Wrap all processing in try-except with informative error messages
- **Paths**: Use `Path` objects from pathlib, ensure all output directories exist
- **Return Values**: Return file paths as strings, use Optional[Type] when result may be None

### Gradio Interface Patterns
- Organize UI into clear tabs for each processing phase
- Always include status messages and progress indicators
- Provide sensible default values for all parameters
- Return both file paths and status messages for user feedback

### Configuration
- Store all default parameters in the `Config` class
- Make configuration values easily adjustable
- Document parameter meanings with inline comments

## Key Features and Components

### Phase 1: Stem Separation
- **Demucs Integration**: Multiple models (htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra, mdx_extra_q)
- **Caching System**: MD5-based caching to avoid reprocessing same audio
- **Output**: vocals, drums, bass, other stems

### Phase 2: Audio Processing
- **Loop Generation**: AI-powered ranking with "Aperture" control (0.0=energy, 1.0=spectral)
- **Vocal Chops**: Three modes (silence, onset, hybrid)
- **MIDI Extraction**: Uses basic_pitch
- **Drum One-Shots**: Transient detection for individual hits

### Phase 3: Video Rendering
- **FFmpeg-Powered**: Multiple aspect ratios (16:9, 4:3, 1:1, 9:16)
- **Visualization Types**: waveform, spectrum, or both

### Feedback System
- JSON-based storage of user ratings and comments
- Timestamped feedback files in `feedback/` directory

## Important Boundaries

### DO NOT MODIFY
- **LICENSE**: MIT License, keep as-is
- **Generated directories**: Do not commit contents of `cache/`, `output/`, `runs/`, `checkpoint/`, `feedback/`
- **.gitignore**: Properly configured to exclude generated content
- **README.md**: Only update if adding significant new features that change user workflow

### BE CAREFUL WITH
- **forgev1.py**: This is a monolithic file (~1000+ lines). Make surgical changes only.
- **Demucs model names**: Must match official Demucs model identifiers
- **Audio file handling**: Always check file existence before processing
- **FFmpeg commands**: Ensure compatibility across platforms (Linux, macOS, Windows)

## Dependencies

### System Requirements
- **FFmpeg**: Required for video rendering and some audio operations
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: Download from https://ffmpeg.org/download.html

### Python Dependencies
- Core: numpy>=1.21.0, scipy>=1.7.0, librosa>=0.10.0, soundfile>=0.12.0, torch>=2.0.0, torchaudio>=2.0.0
- Stem Separation: demucs>=4.0.0
- MIDI: basic-pitch>=0.2.0
- UI: gradio>=4.0.0
- Utilities: tqdm>=4.65.0
- Optional: audiosep (requires model checkpoints and GPU)

### Installation Notes
- PyTorch installation may vary by platform (CPU vs GPU)
- AudioSep is optional and requires significant additional setup
- basic-pitch may require additional system dependencies on some platforms (libsndfile, ffmpeg, audio codecs)

## Common Tasks

### Adding a New Audio Processing Feature
1. Add function with clear docstring and type hints
2. Integrate into appropriate Gradio tab (Phase 1, 2, or 3)
3. Ensure output directory exists (use `setup_directories()`)
4. Implement error handling with user-friendly messages
5. Return both file path and status message
6. Update README.md with feature documentation

### Modifying UI
1. All UI is defined in the Gradio blocks at the bottom of `forgev1.py`
2. Follow existing tab structure (Phase 1, 2, 3, Feedback)
3. Use clear labels and help text for parameters
4. Test with various audio files to ensure robustness

### Performance Considerations
- Large audio files (>5 minutes) may cause memory issues
- Demucs processing is CPU/GPU intensive
- Consider adding progress callbacks for long operations
- Cache results when possible using MD5 hashing pattern

## Testing Strategy

### Manual Testing
- Upload various audio formats (WAV, MP3, FLAC)
- Test with different audio lengths (short clips, full songs)
- Verify output files are created in correct directories
- Check that caching works (second run should be faster)
- Test error handling with invalid inputs

### Audio Quality Checks
- Listen to generated stems, loops, and chops
- Verify MIDI output in a DAW
- Review video visualizations for sync issues
- Check that audio parameters (sample rate, bit depth) are preserved

## Troubleshooting

### Common Issues
- **Demucs errors**: Ensure demucs is installed, check FFmpeg availability
- **Memory issues**: Process shorter clips, use lower quality models
- **MIDI extraction**: Works best on melodic, monophonic audio
- **Video rendering**: Verify FFmpeg is in PATH

### Debug Approach
1. Check console output for error messages
2. Verify input file exists and is valid audio
3. Check that output directories were created
4. Ensure all dependencies are installed
5. Test with a simple, known-good audio file

## Contributing Guidelines

- Keep changes minimal and focused
- Add clear docstrings for new functions
- Update README.md if user workflow changes
- Test manually with the Gradio interface
- Consider backward compatibility
- Follow existing code organization patterns

## Safety and Security

- **No secrets**: Never commit API keys, tokens, or sensitive data
- **Input validation**: Always validate file paths and user inputs
- **Resource limits**: Be mindful of memory usage with large files
- **Dependency updates**: Test thoroughly when updating major dependencies

## Additional Resources

- **Demucs Documentation**: https://github.com/facebookresearch/demucs
- **Gradio Documentation**: https://www.gradio.app/docs/
- **librosa Documentation**: https://librosa.org/doc/latest/
- **basic-pitch**: https://github.com/spotify/basic-pitch

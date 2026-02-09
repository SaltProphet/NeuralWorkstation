# 🎵 FORGE v1 - Neural Audio Workstation

A unified, comprehensive audio processing workstation combining the power of Night Pulse and FORGE into a single, intuitive Gradio interface.

## Features

### Phase 1: Stem Separation

- **Demucs Integration**: Industry-leading stem separation with multiple model options
  - Support for htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra, and mdx_extra_q
  - Intelligent caching system using MD5 hashing for faster reprocessing
  - Separates audio into vocals, drums, bass, and other stems

### Phase 1.5: AudioSep (Advanced)

- **Query-Based Extraction**: Use natural language to extract specific instruments
  - Examples: "bass guitar", "snare drum", "piano", "saxophone"
  - Powered by AudioSep AI model (requires additional setup)

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

### Feedback System

- **User Feedback Collection**: Help improve FORGE
  - Rate individual features (1-5 stars)
  - Provide detailed comments
  - Optional email for follow-up
  - All feedback stored as timestamped JSON files

### Batch Processing

- **Process Multiple Files at Once**: Efficient batch operations
  - Batch stem separation across multiple audio files
  - Batch loop extraction with custom parameters
  - Batch vocal chop generation
  - Batch MIDI extraction
  - Batch drum one-shot generation
  - Progress tracking and JSON reports for each batch

### Performance Optimizations

- **Enhanced Processing Speed**: Optimized for efficiency
  - Parallel processing for batch operations
  - Intelligent cache management with expiration
  - Configurable quality presets (draft, balanced, high)
  - Resource monitoring and limits
  - Memory-mapped audio loading for large files

### REST API

- **Programmatic Access**: Use FORGE via REST API
  - FastAPI-based REST endpoints for all operations
  - OpenAPI/Swagger documentation at `/docs`
  - API key authentication
  - File upload and download endpoints
  - Python client library included

## Installation

### Prerequisites

1. **Python 3.8+**
2. **FFmpeg** (required for MP3/M4A decoding and audio processing)

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
 
# Windows
# Download from https://ffmpeg.org/download.html
```

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/SaltProphet/NeuralWorkstation.git
   cd NeuralWorkstation
   ```

2. **Install Python dependencies**:

   ```bash
  pip install -r requirements.txt
  ```

3. **Optional: Install AudioSep** (for advanced query-based separation):

  ```bash
  pip install audiosep
  # Note: Requires model checkpoints and GPU recommended
  ```

  See [OPTIONAL_FEATURES_GUIDE.md](OPTIONAL_FEATURES_GUIDE.md) for detailed instructions on enabling and using optional features.

## Usage

### Launch the Application

```bash
python app.py
```

The Gradio interface will launch at `http://localhost:7860` (or `http://0.0.0.0:7860`).

### Quick Start Guide

1. **Stem Separation**:
   - Navigate to "Phase 1: Stem Separation" tab
   - Upload your audio file
   - Select a Demucs model (htdemucs recommended for most cases)
   - Click "Separate Stems"
   - Results will be saved in `output/stems/`

2. **Loop Generation**:
   - Navigate to "Phase 2: Loop Generation" tab
   - Upload your audio file (or use a stem from Phase 1)
   - Adjust loop duration (4 seconds is typical for 4-bar loops at 120 BPM)
   - Experiment with **Aperture** control:
     - 0.0 = Prioritize energy/loudness
     - 0.5 = Balanced
     - 1.0 = Prioritize spectral/tonal content
   - Click "Extract Loops"
   - Top-ranked loops saved in `output/loops/`

3. **Vocal Chops**:
   - Navigate to "Phase 2: Vocal Chops" tab
   - Upload vocal stem (or any audio)
   - Select detection mode:
     - **Onset**: Best for rhythmic vocals
     - **Silence**: Best for sparse vocals
     - **Hybrid**: Best for complex material
   - Adjust duration ranges and threshold
   - Click "Generate Chops"
   - Chops saved in `output/chops/`

4. **MIDI Extraction**:
   - Navigate to "Phase 2: MIDI" tab
   - Upload melodic audio (vocals, instruments)
   - Click "Extract MIDI"
   - MIDI file saved in `output/midi/`

5. **Drum One-Shots**:
   - Navigate to "Phase 2: Drum One-Shots" tab
   - Upload drums stem (or drum loop)
   - Adjust duration ranges
   - Click "Extract One-Shots"
   - Individual hits saved in `output/drums/`

6. **Provide Feedback**:
   - Navigate to "Feedback" tab
   - Select feature and provide rating
   - Share your thoughts in comments
   - Click "Submit Feedback"
   - Feedback saved in `feedback/`

## Directory Structure

```text
NeuralWorkstation/
├── app.py             # Main application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── runs/              # Processing runs metadata
├── cache/             # Cached stem separations
├── config/            # Configuration files
├── checkpoint/        # Model checkpoints (if using AudioSep)
├── feedback/          # User feedback JSON files
└── output/            # All generated outputs
    ├── stems/         # Separated audio stems
    ├── loops/         # Extracted loops
    ├── chops/         # Vocal chops
    ├── midi/          # MIDI files
    └── drums/         # Drum one-shots
```

## Configuration

### Aperture Control Explained

The **Aperture** parameter (0.0 - 1.0) in loop generation controls how loops are ranked:

- **0.0 (Energy-focused)**: Prioritizes loud, energetic sections. Best for drops, chorus sections.
- **0.5 (Balanced)**: Equal weighting of energy and tonal content. Good general purpose.
- **1.0 (Spectral-focused)**: Prioritizes harmonic/melodic content. Best for intros, ambient sections.

This innovative control allows you to find different types of loops from the same audio source.

### Model Selection

- **htdemucs**: Best general-purpose model, fast and accurate
- **htdemucs_ft**: Fine-tuned version, slightly better quality
- **htdemucs_6s**: 6-stem separation (adds piano and guitar)
- **mdx_extra**: Higher quality, slower processing
- **mdx_extra_q**: Highest quality, slowest processing

## Tips & Best Practices

### Audio Quality

- Use high-quality source audio (WAV, FLAC preferred over MP3)
- Sample rate: 44.1kHz or 48kHz recommended
- Avoid heavily compressed or low-bitrate files

### Loop Generation Tips

- For 4-bar loops at 120 BPM: use 8-second duration
- For 2-bar loops at 120 BPM: use 4-second duration
- Experiment with Aperture to find different loop types
- Process stems individually for genre-specific loops

### Vocal Chops

- Use the separated vocals stem for cleanest results
- **Onset** mode works best for rap and rhythmic vocals
- **Silence** mode works best for sung, sustained vocals
- **Hybrid** mode for mixed vocal styles

### Drum One-Shots

- Process the separated drums stem for cleanest hits
- Lower max_duration for tighter one-shots
- Use extracted hits in your drum racks/samplers

## Troubleshooting

### Common Issues

#### "Could not get API info" or "No API found" Error

This is a known Gradio 5.x compatibility issue. **Fix:** The latest version includes `ssr_mode=False` in the launch configuration. If you still see this error, ensure you're using the latest version or see [TROUBLESHOOTING.md](TROUBLESHOOTING.md#api--gradio-issues) for manual fix instructions.

#### FFmpeg Not Found

Ensure FFmpeg is installed and in your PATH:

```bash
# Check if installed
ffmpeg -version

# Install if needed (Ubuntu/Debian)
sudo apt-get install ffmpeg
```

#### Memory Issues

- Process shorter audio files (under 5 minutes recommended)
- Use lighter Demucs models (htdemucs instead of mdx_extra_q)
- Close other applications to free RAM

### More Help

For comprehensive troubleshooting, see [**TROUBLESHOOTING.md**](TROUBLESHOOTING.md) which covers:

- API and Gradio issues
- Installation problems
- Audio processing issues
- Deployment issues
- Performance optimization

For information about optional features:

- [**OPTIONAL_FEATURES_GUIDE.md**](OPTIONAL_FEATURES_GUIDE.md) - Quick start guide for enabling AudioSep and other features
- [**OPTIONAL_FEATURES_IMPLEMENTATION.md**](OPTIONAL_FEATURES_IMPLEMENTATION.md) - Technical implementation details

## Advanced Usage

### Batch Processing (Advanced)

For processing multiple files, you can import and use the functions directly:

```python
from app import separate_stems_demucs, extract_loops

# Process multiple files
audio_files = ['track1.wav', 'track2.wav', 'track3.wav']

for audio_file in audio_files:
    # Separate stems
    stems = separate_stems_demucs(audio_file, model='htdemucs')
    
    # Extract loops from each stem
    for stem_name, stem_path in stems.items():
        loops = extract_loops(stem_path, loop_duration=4.0, aperture=0.5)
        print(f"Extracted {len(loops)} loops from {stem_name}")
```

### Custom Configuration

Save custom configurations:

```python
from app import Config

config = {
    'default_model': 'htdemucs_ft',
    'loop_duration': 8.0,
    'aperture': 0.7,
}

Config.save_config(config, 'my_config')
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

MIT License - see LICENSE file for details

## Credits

- **Demucs**: Meta AI Research
- **basic_pitch**: Spotify Research
- **AudioSep**: AudioSep Team
- **Gradio**: Gradio Team
- **FFmpeg**: FFmpeg Project

## Development

### Testing

Run the test suite:

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test categories
pytest tests/unit/ -v          # Unit tests only
pytest tests/integration/ -v   # Integration tests only
```

### REST API (Development)

Start the API server:

```bash
# Install API dependencies
pip install -r requirements-api.txt

# Start the server
python api.py

# Or with uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000
```

Access API documentation at `http://localhost:8000/docs`

Use the Python client:

```python
from api_client_example import ForgeAPIClient

client = ForgeAPIClient()
result = client.extract_loops("audio.wav", num_loops=5)
```

### Performance Optimization

Run performance optimizations:

```bash
python performance.py
```

This will:

- Clean expired cache files
- Manage cache size limits
- Display resource statistics

### CI/CD

The project includes automated CI/CD with GitHub Actions:

- Automated testing on multiple Python versions
- Code linting and quality checks
- Security scanning
- Coverage reporting

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Support

For issues, questions, or feature requests:

1. Check existing issues on GitHub
2. Submit detailed bug reports with error messages
3. Use the in-app feedback system
4. Join our community discussions

---

**FORGE v1** - Built with ❤️ by the NeuralWorkstation Team

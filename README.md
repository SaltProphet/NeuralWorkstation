# 🎵 FORGE v1 - Neural Audio Workstation

A unified, comprehensive audio processing workstation combining the power of Night Pulse and FORGE into a single, intuitive Gradio interface.

**NEW: Autonomous Agent System** - AI-powered conversational agent with intent routing, memory management, and sales capabilities.

## Features

### 🤖 Autonomous Agent System (NEW)

An intelligent conversational agent with:
- **Intent Classification**: Gatekeeper agent routes queries (sales, support, general, greeting)
- **LLM Integration**: Powered by Groq API with fallback stubs for offline use
- **Memory Management**: Qdrant vector database for persistent conversation context
- **Agent Roles**: Specialized agents (Gatekeeper, Librarian, Closer, Ghost, Logger)
- **Gradio Demo**: Interactive chat UI with reasoning log visualization
- **Colab Ready**: One-click deployment to Google Colab with shareable links

See [Autonomous Agent Quick Start](#autonomous-agent-quick-start) below.

---

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

### Phase 3: Video Rendering
- **FFmpeg-Powered Visualization**: Create stunning visualizations
  - Multiple aspect ratios: 16:9, 4:3, 1:1, 9:16
  - Visualization types: waveform, spectrum, or combined
  - High-quality H.264 encoding
  - Perfect for social media and presentations

### Feedback System
- **User Feedback Collection**: Help improve FORGE
  - Rate individual features (1-5 stars)
  - Provide detailed comments
  - Optional email for follow-up
  - All feedback stored as timestamped JSON files

## Installation

### Prerequisites

1. **Python 3.8+**
2. **FFmpeg** (required for video rendering and audio processing)
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

4. **Optional: Configure Autonomous Agent System**:
   
   Create a `.env` file in the project root:
   ```bash
   # Groq API for LLM (get free key from https://console.groq.com)
   GROQ_API_KEY=your_groq_api_key_here
   
   # Qdrant Vector Database (for agent memory)
   # Option 1: Cloud Qdrant (recommended)
   QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
   QDRANT_API_KEY=your_qdrant_api_key
   
   # Option 2: Local Qdrant
   QDRANT_URL=http://localhost:6333
   # QDRANT_API_KEY not needed for local
   ```
   
   **Note**: The agent system works without these credentials by using deterministic stubs and in-memory storage.

## Usage

### Launch the Audio Workstation

```bash
python forgev1.py
```

The Gradio interface will launch at `http://localhost:7860` (or `http://0.0.0.0:7860`).

### Autonomous Agent Quick Start

#### CLI Demo
Run the command-line interface for the autonomous agent:

```bash
python gemini_agentic_system.py
```

**Features**:
- Interactive conversational agent
- Intent classification (sales, support, general, greeting)
- Works offline with deterministic stubs if no API key is set
- Shows reasoning log for each interaction

**Example interactions**:
```
💬 You: How much does this cost?
🤖 Agent: [Sales response with pricing]

💬 You: I need help with my account
🤖 Agent: [Support acknowledgment]

💬 You: Hello!
🤖 Agent: [Greeting with capabilities]
```

#### Gradio Chat Demo (Colab)
Run the interactive Gradio demo in Google Colab:

1. Open the demo file:
   ```bash
   python colab_gradio_demo.py
   ```

2. Copy the displayed Colab cell code

3. Paste into a Google Colab notebook and run

4. Access the shareable Gradio link

**Features**:
- Two-column interface: Chat + Reasoning Log
- Real-time intent classification visualization
- Works in any environment (Colab, local, cloud)
- Shareable public links via Gradio

#### Using Agent Roles

The system includes specialized agent roles defined in `agent_roles.py`:

- **Gatekeeper**: Intent classification and routing
- **Librarian**: Knowledge retrieval and summarization
- **Closer**: Sales and conversion optimization
- **Ghost**: Input sanitization and validation
- **Logger**: Event logging and monitoring

Import and use in your own scripts:

```python
from agent_roles import get_system_prompt, list_available_roles
from gemini_agentic_system import autonomous_agent_main

# List available roles
print(list_available_roles())

# Process user input
result = autonomous_agent_main("How much does your pro plan cost?")
print(result['response'])
print(result['reasoning_log'])
```

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

6. **Video Rendering**:
   - Navigate to "Phase 3: Video" tab
   - Upload audio file
   - Select aspect ratio (16:9 for YouTube, 1:1 for Instagram, 9:16 for TikTok)
   - Choose visualization type
   - Click "Render Video"
   - Video saved in `output/videos/`

7. **Provide Feedback**:
   - Navigate to "Feedback" tab
   - Select feature and provide rating
   - Share your thoughts in comments
   - Click "Submit Feedback"
   - Feedback saved in `feedback/`

## Directory Structure

```
NeuralWorkstation/
├── forgev1.py          # Main application
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
    ├── drums/         # Drum one-shots
    └── videos/        # Rendered videos
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

### Loop Generation
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

### Video Rendering
- Choose aspect ratio based on target platform:
  - YouTube: 16:9
  - Instagram Feed: 1:1
  - Instagram Stories/TikTok: 9:16
  - Professional: 16:9 or 4:3
- **Waveform**: Classic look, works for all genres
- **Spectrum**: Modern look, great for electronic music
- **Both**: Most informative, uses more screen space

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

### Batch Processing

For processing multiple files, you can import and use the functions directly:

```python
from forgev1 import separate_stems_demucs, extract_loops

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
from forgev1 import Config

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

**RedLight Non-Sale License** - Copyright (c) 2026 SaltProphet

This software is free for personal, educational, and non-commercial use. Commercial use, sale, or sublicensing requires explicit written permission from SaltProphet. See [LICENSE](LICENSE) file for full terms.

## Credits

- **Demucs**: Meta AI Research
- **basic_pitch**: Spotify Research
- **AudioSep**: AudioSep Team
- **Gradio**: Gradio Team
- **FFmpeg**: FFmpeg Project

## Support

For issues, questions, or feature requests:
1. Check existing issues on GitHub
2. Submit detailed bug reports with error messages
3. Use the in-app feedback system
4. Join our community discussions

---

**FORGE v1** - Built with ❤️ by the NeuralWorkstation Team

# Quick Start: Enabling Optional Features

## AudioSep (Advanced Stem Separation)

AudioSep is an optional feature that allows you to extract specific instruments using natural language queries like "bass guitar" or "snare drum".

### How to Enable

1. **Install AudioSep**:
   ```bash
   pip install audiosep
   ```

2. **Restart FORGE**:
   ```bash
   python forgev1.py
   ```

3. **Verify Installation**:
   - Open the app (http://localhost:7860)
   - Go to "Phase 1.5: AUDIOSEP" tab
   - You should see: "✅ AudioSep is available"

### Usage

1. **Upload Audio**: Upload an audio file or use output from Phase 1
2. **Enter Query**: Type what you want to extract (e.g., "bass guitar")
3. **Click Extract**: Wait for processing
4. **Get Result**: Listen to or download the extracted audio

### Example Queries

- `bass guitar` - Extract bass guitar parts
- `snare drum` - Extract snare drum hits
- `piano` - Extract piano melody
- `female vocals` - Extract female voice
- `saxophone` - Extract saxophone parts
- `acoustic guitar` - Extract acoustic guitar
- `kick drum` - Extract kick drum

### Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB+ RAM recommended
- **GPU**: Recommended but not required (CPU works, just slower)
- **Disk**: ~500MB for model checkpoints (downloaded on first use)

### Troubleshooting

**Issue**: Button says "⚠️ AUDIOSEP NOT INSTALLED"
- **Solution**: Run `pip install audiosep` and restart FORGE

**Issue**: "AudioSep not available" error
- **Solution**: Make sure audiosep is installed: `pip show audiosep`

**Issue**: Very slow processing
- **Solution**: AudioSep works best with GPU. On CPU, expect 30-60 seconds for short clips.

**Issue**: Out of memory error
- **Solution**: Try processing shorter audio clips (under 30 seconds)

## Future Optional Features

We're planning to add more optional features:
- Advanced MIDI extraction options
- Vocal harmony detection
- Automatic music transcription
- Beat detection and tempo analysis

Check back for updates!

## Support

For help with optional features:
1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. See [OPTIONAL_FEATURES_IMPLEMENTATION.md](OPTIONAL_FEATURES_IMPLEMENTATION.md) for technical details
3. Open an issue on GitHub

---

Made with ❤️ by the NeuralWorkstation Team

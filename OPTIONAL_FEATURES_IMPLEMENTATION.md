# AudioSep Optional Feature Implementation

## Overview

This document describes the implementation of optional features in FORGE v1, specifically focusing on the AudioSep integration. AudioSep is an advanced AI-powered audio separation tool that allows users to extract specific audio elements using natural language queries.

## Changes Made

### 1. Feature Detection System

Added a global feature detection system at startup (`app.py` lines 38-47):

```python
# ============================================================================
# OPTIONAL FEATURES DETECTION
# ============================================================================

# Check if AudioSep is available
try:
    import audiosep
    AUDIOSEP_AVAILABLE = True
except ImportError:
    AUDIOSEP_AVAILABLE = False
```

**Purpose**: Detects whether optional dependencies are installed at runtime, allowing the app to gracefully handle missing features without crashing.

### 2. Enhanced Backend Function

Improved `separate_stems_audiosep()` function with:

- **Better error handling**: Clear, actionable error messages
- **GPU auto-detection**: Automatically uses GPU if available, falls back to CPU
- **Progress reporting**: Detailed progress updates throughout the process
- **Informative failures**: Helpful troubleshooting tips when errors occur

Key improvements:
- Uses global `AUDIOSEP_AVAILABLE` flag for consistent checking
- Auto-detects GPU availability with `torch.cuda.is_available()`
- Provides detailed error messages with common solutions
- Better progress tracking through the separation process

### 3. Functional UI Integration

**Before**: Phase 1.5 tab existed but was just a placeholder that showed a warning message.

**After**: Fully functional UI that:
- **Detects availability**: Shows different UI based on whether AudioSep is installed
- **Provides clear instructions**: If not installed, shows installation command
- **Accepts audio input**: Users can upload audio files directly
- **Validates inputs**: Checks both audio file and query before processing
- **Shows results**: Displays extracted audio and status messages

#### UI Components Added:

1. **Availability Indicator**:
   - ✅ Green message when AudioSep is available
   - ⚠️ Orange warning with installation instructions when not available

2. **Audio Input Section** (1.5.1):
   - File upload component for audio input
   - Can accept output from Phase 1 or new uploads

3. **Query Input Section** (1.5.2):
   - Text input for natural language queries
   - Example queries provided
   - Default value: "bass guitar"

4. **Dynamic Button**:
   - Shows "⚡ EXTRACT" when AudioSep is available
   - Shows "⚠️ AUDIOSEP NOT INSTALLED" when not available
   - Button is disabled when AudioSep is not installed

5. **Output Section**:
   - Audio player for extracted result
   - Status textbox with detailed feedback

### 4. Comprehensive Error Handling

The `audiosep_wrapper()` function provides three levels of validation:

1. **Availability Check**: Verifies AudioSep is installed
2. **Audio Validation**: Ensures audio file is provided
3. **Query Validation**: Ensures query is not empty

Each validation provides specific, actionable error messages.

### 5. Gradio 6.x Compatibility

Fixed compatibility issues with Gradio 6.x:
- Removed `info` parameter from `gr.Audio()` and `gr.Textbox()`
- Moved informational text to labels or separate markdown blocks
- Ensures compatibility with latest Gradio versions

## Usage

### When AudioSep is NOT Installed (Default)

1. Navigate to "Phase 1.5: AUDIOSEP" tab
2. See warning message with installation instructions
3. Button is disabled with text "⚠️ AUDIOSEP NOT INSTALLED"
4. User knows exactly what to do to enable the feature

### When AudioSep IS Installed

1. Navigate to "Phase 1.5: AUDIOSEP" tab
2. See success message "✅ AudioSep is available"
3. Upload audio file or use output from Phase 1
4. Enter natural language query (e.g., "bass guitar", "snare drum", "piano")
5. Click "⚡ EXTRACT"
6. Wait for processing (progress shown)
7. Result appears in audio player below

## Example Queries

AudioSep supports natural language queries like:
- `bass guitar`
- `snare drum`
- `piano`
- `saxophone`
- `female vocals`
- `acoustic guitar`
- `violin`
- `kick drum`

## Technical Details

### Dependencies

- **Required**: numpy, scipy, librosa, soundfile, gradio
- **Optional**: audiosep (with its dependencies: PyTorch, specific models)

### Performance Notes

- AudioSep is memory-intensive (requires ~2-4GB RAM)
- GPU recommended for faster processing
- CPU mode works but is significantly slower
- First run downloads model checkpoints (~500MB)

### File Output

Extracted audio is saved to:
```
output/stems/{original_filename}_audiosep_{query}.wav
```

Example:
```
output/stems/song_audiosep_bass_guitar.wav
```

## Testing

All functionality has been tested:

1. ✅ AudioSep detection works correctly
2. ✅ Error handling provides clear messages
3. ✅ UI shows appropriate state based on availability
4. ✅ Backend function validates inputs properly
5. ✅ Installation instructions are clear and actionable
6. ✅ Gradio 6.x compatibility confirmed

## Future Enhancements

Possible future improvements:
1. Add more optional features (e.g., advanced MIDI extraction)
2. Support batch processing for AudioSep
3. Add preset queries (dropdown with common instruments)
4. Cache AudioSep models to speed up subsequent runs
5. Add quality/speed tradeoff options

## Benefits

This implementation provides:

1. **No Breaking Changes**: App works exactly the same if AudioSep is not installed
2. **Clear User Experience**: Users know immediately if a feature is available
3. **Easy Enablement**: One command to install optional features
4. **Graceful Degradation**: Missing features don't crash the app
5. **Extensibility**: Pattern can be reused for other optional features

## Conclusion

The optional features implementation successfully integrates AudioSep into FORGE v1 while maintaining backward compatibility and providing a clear path for users to enable advanced features. The approach can serve as a template for adding other optional features in the future.

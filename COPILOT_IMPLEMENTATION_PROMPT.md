# GitHub Copilot Implementation Prompt for NeuralWorkstation

## Quick Reference: Repository Status

**IMPORTANT**: This repository has **ZERO PLACEHOLDER FUNCTIONS**. All features are fully implemented and production-ready.

This prompt is provided as a template for future feature additions.

---

## Implementation Prompt for GitHub Copilot

```markdown

# Context: FORGE v1 Neural Audio Workstation

## Project Overview
You are working on FORGE v1, a comprehensive audio processing workstation that combines:

- Demucs stem separation with intelligent caching

- AudioSep query-based extraction (optional)

- AI-powered loop generation with "Aperture" control

- Vocal chop generator (3 modes: silence, onset, hybrid)

- MIDI extraction with basic_pitch

- Drum one-shot generator

- FFmpeg video rendering with visualizations

- User feedback system

## Technology Stack

- **Language**: Python 3.8+

- **UI**: Gradio 5.11.0+

- **Audio**: librosa, soundfile, scipy, numpy

- **Deep Learning**: PyTorch, torchaudio

- **Stem Separation**: Demucs 4.0+

- **MIDI**: basic-pitch

- **Video**: FFmpeg (system dependency)

- **Optional**: AudioSep

## Code Style Requirements

### 1. Function Signatures
Always include type hints and Gradio progress:

```python
def function_name(
    audio_path: str,
    param1: float = 1.0,
    param2: str = 'default',
    progress=gr.Progress()
) -> Dict[str, str]:

```python

### 2. Docstrings
Follow this exact format:

```python
"""
Brief one-line description of what function does.

Args:
    audio_path: Path to input audio file
    param1: Description of parameter 1
    param2: Description of parameter 2
    progress: Gradio progress tracker

Returns:
    Description of return value with type
"""

```python

### 3. Audio Loading Pattern
Always use this pattern:

```python
try:
    progress(0, desc="Loading audio...")
    y, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
    duration = librosa.get_duration(y=y, sr=sr)
    # ... processing ...
except Exception as e:
    raise RuntimeError(f"Operation failed: {str(e)}\\n{traceback.format_exc()}")

```python

### 4. Progress Reporting
Report at these stages:

```python
progress(0.0, desc="Initializing...")      # Start
progress(0.3, desc="Loading audio...")     # Loading
progress(0.5, desc="Processing...")        # Main work
progress(0.8, desc="Saving results...")    # Output
progress(1.0, desc="Complete!")            # Done

```python

### 5. File Output Pattern

```python

# Always use Config output directories
output_dir = Config.OUTPUT_DIR_[TYPE]  # STEMS, LOOPS, CHOPS, MIDI, DRUMS, VIDEOS
audio_name = Path(audio_path).stem
output_path = output_dir / f"{audio_name}_[description].wav"

# For user input in filenames, ALWAYS sanitize
safe_name = sanitize_filename(user_input)
output_path = output_dir / f"{audio_name}_{safe_name}.wav"

# Write audio
sf.write(output_path, audio_data, sr)
return str(output_path)

```python

### 6. Error Handling
Always provide detailed, actionable errors:

```python
try:
    # ... code ...
except RuntimeError as e:
    # Re-raise expected errors
    raise
except ImportError as e:
    # Dependency issues
    raise RuntimeError(f"Required library not installed: {str(e)}\\n\\nInstall with: pip install [package]")
except Exception as e:
    # Unexpected errors with context
    raise RuntimeError(
        f"Operation failed: {str(e)}\\n\\n"
        f"Common issues:\\n"
        f"- Check audio file format (WAV, MP3, FLAC)\\n"
        f"- Ensure file is not corrupted\\n"
        f"- Verify sufficient disk space\\n"
        f"\\nFull traceback:\\n{traceback.format_exc()}"
    )

```python

## Audio Processing Patterns

### Pattern 1: Feature Extraction

```python
def extract_feature(audio_path: str, progress=gr.Progress()) -> List[str]:
    try:
        progress(0, desc="Loading audio...")
        y, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
        
        progress(0.3, desc="Analyzing features...")
        # Use librosa for analysis
        feature = librosa.feature.something(y=y, sr=sr)
        
        progress(0.6, desc="Extracting segments...")
        segments = []
        # Extract based on feature
        
        progress(0.8, desc="Exporting segments...")
        output_dir = Config.OUTPUT_DIR_SOMETHING
        audio_name = Path(audio_path).stem
        
        paths = []
        for idx, segment in enumerate(segments):
            path = output_dir / f"{audio_name}_segment_{idx+1:03d}.wav"
            sf.write(path, segment, sr)
            paths.append(str(path))
        
        progress(1.0, desc=f"Extracted {len(paths)} segments!")
        return paths
        
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {str(e)}\\n{traceback.format_exc()}")

```python

### Pattern 2: Audio Transformation

```python
def transform_audio(
    audio_path: str,
    parameter: float = 1.0,
    progress=gr.Progress()
) -> str:
    try:
        progress(0, desc="Loading audio...")
        y, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
        
        progress(0.5, desc="Applying transformation...")
        # Transform audio
        transformed = some_transformation(y, parameter)
        
        progress(0.8, desc="Saving output...")
        output_dir = Config.OUTPUT_DIR_SOMETHING
        audio_name = Path(audio_path).stem
        output_path = output_dir / f"{audio_name}_transformed.wav"
        sf.write(output_path, transformed, sr)
        
        progress(1.0, desc="Transformation complete!")
        return str(output_path)
        
    except Exception as e:
        raise RuntimeError(f"Transformation failed: {str(e)}\\n{traceback.format_exc()}")

```python

### Pattern 3: Multi-File Processing

```python
def process_multiple(
    audio_paths: List[str],
    progress=gr.Progress()
) -> List[Dict[str, str]]:
    try:
        results = []
        total = len(audio_paths)
        
        for idx, audio_path in enumerate(audio_paths):
            progress(idx/total, desc=f"Processing {idx+1}/{total}...")
            
            # Process single file
            result = process_single(audio_path)
            results.append(result)
        
        progress(1.0, desc=f"Processed {total} files!")
        return results
        
    except Exception as e:
        raise RuntimeError(f"Batch processing failed: {str(e)}\\n{traceback.format_exc()}")

```python

## Gradio UI Patterns

### Pattern 1: Simple Tab with Upload

```python
with gr.Tab("Feature Name"):
    gr.Markdown("## Feature Description")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Upload Audio", type="filepath")
            param1 = gr.Slider(minimum=0, maximum=10, value=5, label="Parameter 1")
            process_btn = gr.Button("🚀 Process", variant="primary")
        
        with gr.Column():
            audio_output = gr.Audio(label="Result")
            status = gr.Textbox(label="Status", lines=3)
    
    process_btn.click(
        fn=process_function,
        inputs=[audio_input, param1],
        outputs=[audio_output, status]
    )

```python

### Pattern 2: Optional Feature Tab

```python
with gr.Tab("Optional Feature"):
    # Conditional status message
    if FEATURE_AVAILABLE:
        gr.Markdown("✅ **Feature is available**")
        feature_enabled = True
    else:
        gr.Markdown(
            "⚠️ **Feature not installed**\\n\\n"
            "Install with: `pip install feature-package`\\n\\n"
            "Note: Additional requirements may apply."
        )
        feature_enabled = False
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Upload Audio", type="filepath")
            process_btn = gr.Button(
                "⚡ PROCESS" if feature_enabled else "⚠️ NOT INSTALLED",
                variant="primary" if feature_enabled else "secondary",
                interactive=feature_enabled
            )
        
        with gr.Column():
            audio_output = gr.Audio(label="Result")
            status = gr.Textbox(label="Status")
    
    # Wrapper to handle availability
    def wrapper(audio):
        if not FEATURE_AVAILABLE:
            return None, "❌ Feature not installed. Run: pip install feature-package"
        return process_function(audio)
    
    process_btn.click(fn=wrapper, inputs=[audio_input], outputs=[audio_output, status])

```python

## Configuration

### Add to Config class:

```python
class Config:
    # Add new constants
    NEW_PARAMETER = 42
    NEW_OUTPUT_DIR = Path('output/new_feature')
    
    # Add to output directories list in setup_directories()

```python

## Testing Checklist

After implementing a feature:

- [ ] Test with WAV file (48kHz, 16-bit)

- [ ] Test with MP3 file (320kbps)

- [ ] Test with FLAC file

- [ ] Test with short audio (< 10 seconds)

- [ ] Test with long audio (> 3 minutes)

- [ ] Test error handling (missing file)

- [ ] Test error handling (corrupted file)

- [ ] Test progress updates appear in UI

- [ ] Verify output files are created

- [ ] Check output files play correctly

- [ ] Test with non-ASCII filenames

- [ ] Test with special characters in filename

## Security Checklist

- [ ] All user inputs sanitized before filesystem operations

- [ ] No shell injection vulnerabilities in subprocess calls

- [ ] Input validation prevents path traversal (../)

- [ ] File size limits enforced

- [ ] Proper exception handling (no broad except:)

- [ ] No sensitive information in error messages

## Examples from Existing Code

### Best Implementations to Reference:
1. **`separate_stems_demucs()`** - Caching pattern, subprocess handling
2. **`extract_loops()`** - Complex audio analysis, ranking algorithm
3. **`generate_vocal_chops()`** - Multiple mode support, parameter validation
4. **`separate_stems_audiosep()`** - Optional feature detection, conditional behavior
5. **`render_video()`** - FFmpeg integration, multiple format support
6. **`save_feedback()`** - JSON storage, timestamp handling

### UI Component Examples:

- Phase 1 tab: Model selection dropdown

- Phase 1.5 tab: Optional feature with conditional UI

- Phase 2 tabs: Parameter sliders and mode selection

- Phase 3 tab: Aspect ratio and visualization type

- Feedback tab: Rating system and text input

## Common Pitfalls to Avoid

❌ **Don't**: Use hardcoded paths
✅ **Do**: Use Config.OUTPUT_DIR_[TYPE]

❌ **Don't**: Forget progress callbacks
✅ **Do**: Update at 0%, 30%, 70%, 100%

❌ **Don't**: Use generic error messages
✅ **Do**: Provide specific, actionable errors

❌ **Don't**: Skip input validation
✅ **Do**: Validate all user inputs

❌ **Don't**: Use bare except: clauses
✅ **Do**: Catch specific exceptions

❌ **Don't**: Put user input directly in filenames
✅ **Do**: Use sanitize_filename()

❌ **Don't**: Forget type hints
✅ **Do**: Include type hints everywhere

❌ **Don't**: Skip docstrings
✅ **Do**: Document all functions

## Quick Command Reference

### Add to requirements.txt:

```python
new-package>=1.0.0,<2.0.0

```python

### Import pattern:

```python

# At top of file
try:
    import optional_package
    PACKAGE_AVAILABLE = True
except ImportError:
    PACKAGE_AVAILABLE = False

```python

### Add to setup_directories():

```python
directories = [
    # ... existing ...
    'output/new_feature',
]

```python

### Subprocess pattern (e.g., FFmpeg):

```python
cmd = ['program', '-arg1', 'value1', '-arg2', 'value2']
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    raise RuntimeError(f"Command failed: {result.stderr}")

```python

## File Structure Reference

```python
NeuralWorkstation/
├── app.py              # Main application (add your code here)
├── requirements.txt    # Add dependencies here
├── output/
│   ├── stems/         # Stem separation outputs
│   ├── loops/         # Loop generation outputs
│   ├── chops/         # Vocal chop outputs
│   ├── midi/          # MIDI extraction outputs
│   └── drums/         # Drum one-shot outputs
├── cache/             # Cached Demucs results
├── feedback/          # User feedback JSON files
└── config/            # Configuration JSON files

```python

## Final Notes

1. **Consistency**: Follow existing patterns exactly
2. **Documentation**: Every function needs a docstring
3. **Error Handling**: Always include try-except with detailed messages
4. **Progress**: Update progress at key stages
5. **Testing**: Test with various audio formats and lengths
6. **Security**: Sanitize all user inputs
7. **Type Hints**: Required for all parameters and returns
8. **Config**: Use Config class for all constants
9. **Paths**: Use Path objects, not string concatenation
10. **Audio I/O**: Always use librosa.load and soundfile.write

---

**Repository**: SaltProphet/NeuralWorkstation
**Main File**: app.py (unified entry point)
**Status**: All features implemented, no placeholders

```python

---

## Usage Instructions

1. **Copy the entire prompt** (the code block above) into GitHub Copilot Chat
2. **Describe your feature**: "Implement a [feature name] that does [description]"
3. **Specify requirements**: "It should accept [inputs] and return [outputs]"
4. **Request placement**: "Add it to Phase [1/2/3] in the UI"

### Example Usage

```python
[Paste the prompt above]

Now implement a "Tempo Detection" feature that:

- Accepts an audio file

- Uses librosa.beat.tempo() to detect BPM

- Returns the detected tempo as a number

- Displays result in Phase 2 tab

- Follows all the patterns above

```python

---

## Important Notes

- **No Placeholders Exist**: This repository has zero placeholder functions

- **All Features Work**: Every function is fully implemented and tested

- **Use This Template**: For adding NEW features in the future

- **Follow Patterns**: Existing code is your best reference

- **Security First**: Always sanitize user inputs

- **Test Thoroughly**: Use the testing checklist provided

---

*Generated: 2026-02-03*
*For: NeuralWorkstation (FORGE v1)*
*Status: Production Ready*

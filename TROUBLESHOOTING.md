# 🔧 FORGE v1 Troubleshooting Guide

This guide covers common issues and their solutions for FORGE v1 Neural Audio Workstation.

## Table of Contents

1. [API / Gradio Issues](#api--gradio-issues)
2. [Installation Issues](#installation-issues)
3. [Audio Processing Issues](#audio-processing-issues)
4. [Deployment Issues](#deployment-issues)
5. [Performance Issues](#performance-issues)

---

## API / Gradio Issues

### "Could not get API info" or "No API found" Error

**Symptoms:**
- Application fails to load in the browser
- Error message: "Could not get API info"
- Error message: "No API found"
- Gradio interface shows connection errors

**Root Cause:**
This is a known compatibility issue with Gradio 5.x where Server-Side Rendering (SSR) mode can cause API initialization problems. The Gradio client cannot properly fetch the API schema when SSR is enabled.

**Solution:**
Add `ssr_mode=False` to the `app.launch()` configuration:

```python
app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    show_error=True,
    ssr_mode=False  # ← Add this line
)
```

**Status:** ✅ **FIXED** in the latest version
- Main `forgev1.py` already includes this fix
- Hugging Face deployment files (`huggingface/forgev1.py`) now include this fix
- If you're still experiencing this issue, ensure you're using the latest version

**Manual Fix (if needed):**
1. Open `forgev1.py` or `huggingface/forgev1.py`
2. Find the `app.launch()` call (around line 1462)
3. Add `ssr_mode=False` as shown above
4. Restart the application

**Related Links:**
- [Gradio 5.x Migration Guide](https://www.gradio.app/guides/gradio-5-migration-guide)
- [Gradio SSR Mode Documentation](https://www.gradio.app/docs/gradio/blocks#blocks-launch)

---

## Installation Issues

### FFmpeg Not Found

**Symptoms:**
- Error: "FFmpeg not found"
- Video rendering fails
- Some audio processing operations fail

**Solution:**

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add to PATH:
   - Open System Properties → Environment Variables
   - Edit PATH variable
   - Add FFmpeg bin folder (e.g., `C:\ffmpeg\bin`)
   - Restart terminal/command prompt

**Verify Installation:**
```bash
ffmpeg -version
```

### Module Not Found Errors

**Symptoms:**
- `ModuleNotFoundError: No module named 'librosa'`
- Other import errors

**Solution:**
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# If still failing, try with binary preference
pip install --prefer-binary -r requirements.txt

# For specific module issues
pip install --upgrade librosa  # or the specific module
```

### NumPy Build Timeout (Hugging Face Spaces)

**Symptoms:**
- Deployment hangs during numpy installation
- Build timeout after 10+ minutes
- "Building wheel for numpy" message

**Solution:**
This is already fixed in `requirements.txt` by pinning numpy version:
```
numpy==1.26.4  # Ensures prebuilt wheel is used
```

**If still experiencing issues:**
- Ensure using latest `requirements.txt`
- Check that `--prefer-binary` flag is used during installation
- See `DEPLOYMENT_TIMEOUT_FIX.md` for detailed information

---

## Audio Processing Issues

### Out of Memory Errors

**Symptoms:**
- Application crashes during processing
- "MemoryError" or "Out of memory"
- System becomes unresponsive

**Solutions:**

1. **Process shorter audio clips:**
   - Split long audio files into 3-5 minute segments
   - Process segments individually

2. **Use lighter Demucs models:**
   - Use `htdemucs` instead of `mdx_extra_q`
   - Smaller models use less memory

3. **Increase system resources:**
   - Close other applications
   - Enable swap memory (Linux):
     ```bash
     sudo fallocate -l 4G /swapfile
     sudo chmod 600 /swapfile
     sudo mkswap /swapfile
     sudo swapon /swapfile
     ```

4. **For cloud deployments:**
   - Upgrade to instances with more RAM
   - Use GPU instances with more VRAM

### Slow Processing

**Symptoms:**
- Operations take a long time to complete
- Stem separation takes 10+ minutes

**Solutions:**

1. **Use GPU acceleration:**
   - Ensure PyTorch CUDA is installed for GPU support
   - Use GPU instances on cloud platforms

2. **Enable caching:**
   - Caching is enabled by default
   - Second processing of same file should be instant
   - Check that `cache/` directory has write permissions

3. **Use faster models:**
   - `htdemucs` is faster than `mdx_extra_q`
   - Trade-off: faster models may have lower quality

4. **Check system resources:**
   ```bash
   # Monitor CPU/RAM usage
   htop  # Linux/macOS
   # or
   top   # All platforms
   ```

### Poor Quality Results

**Symptoms:**
- Stems have artifacts
- MIDI extraction is inaccurate
- Loops don't sound good

**Solutions:**

1. **For stem separation:**
   - Use higher quality models (`mdx_extra_q` or `htdemucs_ft`)
   - Ensure input audio is high quality (not heavily compressed)

2. **For MIDI extraction:**
   - Works best on monophonic, melodic content
   - Piano and guitar extract better than full mixes
   - Use isolated stems for better results

3. **For loop extraction:**
   - Adjust "Aperture" control to balance energy/spectral features
   - Try different loop durations
   - Extract more loops and choose the best ones

### File Format Issues

**Symptoms:**
- "Unsupported audio format"
- Files won't upload or process

**Supported Formats:**
- WAV (recommended)
- MP3
- FLAC
- OGG
- M4A

**Solution:**
Convert unsupported formats using FFmpeg:
```bash
ffmpeg -i input.xxx -ar 44100 output.wav
```

---

## Deployment Issues

### Port Already in Use

**Symptoms:**
- "Address already in use"
- "Port 7860 is already allocated"

**Solution:**

**Option 1: Change the port**
Edit `forgev1.py`:
```python
app.launch(
    server_name="0.0.0.0",
    server_port=7861,  # Change to different port
    ...
)
```

**Option 2: Kill existing process**
```bash
# Linux/macOS
lsof -ti:7860 | xargs kill -9

# Windows
netstat -ano | findstr :7860
taskkill /PID <PID> /F
```

### Hugging Face Spaces Build Failures

**Symptoms:**
- Build timeout
- Dependencies fail to install
- Application won't start

**Solutions:**

1. **Use optimized deployment files:**
   - Use files from `huggingface/` folder
   - These are pre-configured for HF Spaces

2. **Check build logs:**
   - Look for specific error messages
   - Common issues: FFmpeg missing (it's pre-installed on HF), wrong Python version

3. **Verify configuration:**
   - Ensure `app.py` is the entry point
   - Check that `requirements.txt` has version constraints
   - Verify SDK version in README frontmatter

4. **Hardware recommendations:**
   - Use CPU Upgrade or GPU T4 for better performance
   - Free tier may be too slow for processing

**See also:** `HUGGINGFACE_DEPLOYMENT.md`

### Connection Issues / Timeouts

**Symptoms:**
- "Connection refused"
- Application times out
- Cannot access interface

**Solutions:**

1. **Check firewall settings:**
   ```bash
   # Allow port 7860
   sudo ufw allow 7860
   ```

2. **Verify server is running:**
   - Check console for "Running on: http://0.0.0.0:7860"
   - Ensure no errors in startup logs

3. **Use correct URL:**
   - Local: `http://localhost:7860`
   - Network: `http://<your-ip>:7860`
   - Cloud: Use platform-provided URL

4. **Check deployment platform status:**
   - Hugging Face Spaces: Check space status page
   - Other platforms: Check service health dashboards

---

## Performance Issues

### High CPU Usage

**Causes:**
- Stem separation is CPU-intensive
- Multiple simultaneous operations

**Solutions:**
- Normal during processing
- Use GPU instances to offload work
- Process one file at a time
- Close other applications

### High Memory Usage

**Causes:**
- Large audio files
- Multiple models loaded

**Solutions:**
- Process shorter clips
- Restart application between large jobs
- Clear cache if needed:
  ```bash
  rm -rf cache/*
  ```

### Disk Space Issues

**Symptoms:**
- "No space left on device"
- Write errors

**Solutions:**

1. **Clean up output directories:**
   ```bash
   rm -rf output/*
   rm -rf cache/*
   rm -rf runs/*
   ```

2. **Check disk usage:**
   ```bash
   df -h
   du -sh output/ cache/
   ```

3. **Move processed files:**
   - Download outputs regularly
   - Delete old files you don't need

---

## Getting Help

If you're still experiencing issues:

1. **Check existing issues:**
   - [GitHub Issues](https://github.com/SaltProphet/NeuralWorkstation/issues)
   - Search for similar problems

2. **Open a new issue:**
   - Provide detailed description
   - Include error messages
   - Specify your platform (OS, Python version)
   - Include steps to reproduce

3. **Include diagnostic information:**
   ```bash
   python --version
   pip list | grep -E "gradio|torch|demucs|librosa"
   ffmpeg -version
   ```

4. **Community resources:**
   - [Gradio Documentation](https://www.gradio.app/docs/)
   - [Demucs GitHub](https://github.com/facebookresearch/demucs)
   - [Hugging Face Forums](https://discuss.huggingface.co/)

---

## Additional Resources

- **Deployment Guide:** `DEPLOYMENT.md`
- **Hugging Face Setup:** `HUGGINGFACE_DEPLOYMENT.md`
- **Timeout Fixes:** `DEPLOYMENT_TIMEOUT_FIX.md`
- **Main Documentation:** `README.md`

---

**Last Updated:** January 2026

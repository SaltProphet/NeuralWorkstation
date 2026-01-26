# Deploying FORGE v1 to Hugging Face Spaces

## Quick Start

1. **Create a new Space** on [Hugging Face Spaces](https://huggingface.co/spaces)
   - Select **Gradio** as the SDK
   - Choose Python 3.10 or higher

2. **Upload Files**
   - Upload `app.py` (main entry point)
   - Upload `forgev1.py` (core application)
   - Upload `requirements.txt` (dependencies)
   - Upload `README.md` (documentation)

3. **Configure Space**
   - Set the Space to use **Gradio** SDK
   - Ensure FFmpeg is available (it's pre-installed in Spaces)

4. **Build and Deploy**
   - Hugging Face will automatically build and deploy your Space
   - Access your app at: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

## Space Configuration

Create a `README.md` in your Space with the following header:

```yaml
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
---
```

## Requirements

All dependencies in `requirements.txt` should work out of the box on Hugging Face Spaces.

### Optional Dependencies

- **AudioSep**: Not included by default due to size. Can be added if needed.
- **GPU**: Recommended for faster processing, especially for Demucs stem separation

## System Dependencies

Hugging Face Spaces comes with:
- ✅ FFmpeg (pre-installed)
- ✅ Python 3.10+
- ✅ CUDA (on GPU-enabled Spaces)

## Performance Tips

1. **Use GPU Space** for faster Demucs processing
2. **Enable persistent storage** to cache model weights
3. **Set appropriate memory limits** for large audio files
4. **Consider using Space secrets** for any API keys (if you add external services)

## Troubleshooting

### Build Timeout / numpy Building from Source

If the deployment times out during numpy installation:
- The repository has been updated to use pinned numpy versions with prebuilt wheels
- Ensure you're using Python 3.10 or 3.11 (HF Spaces default)
- The Dockerfile now upgrades pip first to ensure binary packages are used
- If using direct requirements.txt (without Docker), pip should prefer binary packages

### Out of Memory
- Use a GPU Space with more RAM
- Process shorter audio clips
- Disable AudioSep if enabled

### Slow Processing
- Upgrade to a GPU Space
- Use lighter Demucs models (htdemucs vs mdx_extra)
- Enable caching for repeated operations

### Build Failures
- Check that all dependencies in `requirements.txt` are compatible
- Ensure `app.py` is set as the main file
- Verify Python version compatibility (3.10+)

## Example Space Setup

```bash
# Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Copy files
cp path/to/NeuralWorkstation/app.py .
cp path/to/NeuralWorkstation/forgev1.py .
cp path/to/NeuralWorkstation/requirements.txt .

# Add README header for Spaces
cat > README.md << 'EOF'
---
title: FORGE v1
emoji: 🎵
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 5.11.0
app_file: app.py
pinned: false
license: mit
---

# FORGE v1 - Neural Audio Workstation

[Your documentation here...]
EOF

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

## Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio on Spaces Guide](https://huggingface.co/docs/hub/spaces-sdks-gradio)
- [Custom Docker Containers on Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)

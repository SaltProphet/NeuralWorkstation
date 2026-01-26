# Deploying to Hugging Face Spaces

This folder contains all the files needed to deploy FORGE v1 to Hugging Face Spaces.

## Quick Deployment Steps

### Option 1: Direct Upload (Recommended for Beginners)

1. **Create a new Space** on [Hugging Face Spaces](https://huggingface.co/new-space)
   - Choose a name for your Space (e.g., "forge-audio-workstation")
   - Select **Gradio** as the SDK
   - Choose **Public** or **Private** visibility
   - Click "Create Space"

2. **Upload Files**
   - Click "Files and versions" tab
   - Click "Add file" → "Upload files"
   - Upload ALL files from this folder:
     - `README.md` (includes HF Spaces metadata)
     - `app.py` (entry point)
     - `forgev1.py` (main application)
     - `requirements.txt` (dependencies)
     - `LICENSE` (MIT license)
     - `.gitignore` (exclude generated files)

3. **Wait for Build**
   - Hugging Face will automatically build your Space
   - This may take 5-10 minutes on first deployment
   - Check the "Logs" tab to monitor progress

4. **Access Your App**
   - Once built, your app will be available at:
     `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

### Option 2: Git-Based Deployment (Recommended for Advanced Users)

```bash
# Clone your Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Copy all files from this folder
cp /path/to/NeuralWorkstation/huggingface/* .

# Commit and push
git add .
git commit -m "Initial deployment of FORGE v1"
git push
```

## Files in This Folder

- **README.md**: Documentation with HF Spaces metadata header
- **app.py**: Entry point that launches the application
- **forgev1.py**: Main application code (all features)
- **requirements.txt**: Python dependencies (optimized to avoid build timeouts)
- **LICENSE**: MIT License
- **.gitignore**: Excludes runtime-generated files from git

## Configuration

The `README.md` includes the following HF Spaces configuration:

```yaml
title: FORGE v1 - Neural Audio Workstation
emoji: 🎵
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
python_version: "3.10"
```

### Upgrading to GPU (Recommended)

For faster processing, upgrade your Space to use a GPU:

1. Go to your Space settings
2. Under "Hardware", select a GPU tier (e.g., T4 small)
3. Note: GPU Spaces may require payment

## Troubleshooting

### Build Timeout

If the build times out during dependency installation:
- The `requirements.txt` has been optimized to use prebuilt wheels
- Ensure numpy is pinned to version 1.26.4
- The build should complete in under 10 minutes

### Out of Memory

If you see OOM errors:
- Upgrade to a GPU Space with more RAM
- Process shorter audio clips
- Use lighter Demucs models (htdemucs instead of mdx_extra)

### Slow Processing

- Upgrade to a GPU-enabled Space for 5-10x speedup
- Enable caching (done automatically)
- Use lighter models for faster results

## Testing Your Deployment

1. Upload a short audio file (30 seconds - 1 minute)
2. Try Phase 1: Stem separation with "htdemucs" model
3. Try Phase 2: Generate loops or vocal chops
4. Try Phase 3: Create a video visualization

## Performance Expectations

- **CPU Space**: 
  - Stem separation: 2-5 minutes per minute of audio
  - Other operations: 10-30 seconds
  
- **GPU Space**: 
  - Stem separation: 20-60 seconds per minute of audio
  - Other operations: 5-15 seconds

## Support

- **Issues**: [GitHub Issues](https://github.com/SaltProphet/NeuralWorkstation/issues)
- **Documentation**: See main repository README
- **HF Spaces Docs**: [spaces.huggingface.co/docs](https://huggingface.co/docs/hub/spaces)

## License

MIT License - Free to use, modify, and distribute

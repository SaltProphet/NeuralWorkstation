# HF Spaces Deployment Fixes

## Recent Updates (2026-01-26)

### Gradio 6.x Compatibility Fix

**Problem**: UI was broken on Hugging Face Spaces due to Gradio 6.x incompatibility.

**Root Causes**:
1. Version constraint prevented Gradio 6.x: `gradio>=5.11.0,<6.0.0`
2. Gradio 6.0 moved `theme` and `css` parameters from `Blocks()` to `launch()`
3. CSS variable was not accessible in the correct scope

**Solutions**:
1. Updated requirements.txt to allow Gradio 6.x: `gradio>=5.11.0`
2. Moved `theme` and `css` to `launch()` method
3. Made `CUSTOM_CSS` a module-level constant for proper scoping
4. Standardized all documentation to reference `sdk_version: 6.0.0`

**Result**: ✅ UI now renders correctly on HF Spaces with Gradio 6.x

---

## What Was Changed

This folder contains optimized files for Hugging Face Spaces deployment that address the build timeout issue.

## The Problem

When deploying to HF Spaces, the build was hanging during numpy installation:
```
Downloading numpy-1.26.4.tar.gz (15.8 MB)
Installing build dependencies: started
Preparing metadata (pyproject.toml): still running...
```

This timeout occurred because pip was downloading numpy as a source tarball (.tar.gz) instead of a prebuilt wheel (.whl), requiring compilation from source which takes too long.

## The Solution

### 1. Pinned numpy Version
```python
# requirements.txt
numpy==1.26.4  # Exact version with guaranteed wheel availability
```

### 2. Added Version Constraints
All dependencies now have upper version bounds to ensure compatibility:
```python
scipy>=1.7.0,<1.12.0
librosa>=0.10.0,<0.11.0
gradio>=5.11.0  # Upgraded to fix security vulnerabilities and support Gradio 6.x
# etc.
```

### 3. Optimized Installation Order
Dependencies are ordered to install numpy first, then packages that depend on it.

### 4. HF Spaces Configuration
The README.md includes proper frontmatter:
```yaml
---
sdk: gradio
sdk_version: 6.0.0
python_version: "3.10"
---
```

## Why This Works

1. **Exact numpy version**: Ensures pip finds the correct prebuilt wheel
2. **Version constraints**: Prevents pip from attempting to resolve to incompatible versions
3. **Python 3.10**: This Python version has excellent wheel support across all dependencies
4. **Modern pip**: HF Spaces uses recent pip which properly handles binary packages

## Testing

To verify the fix works locally (with Python 3.10):
```bash
python3.10 -m venv test_env
source test_env/bin/activate
pip install --prefer-binary -r requirements.txt
```

This should complete in under 2-3 minutes without building any packages from source.

## Expected Build Time

- **Before fix**: Timeout after 10+ minutes (building numpy from source)
- **After fix**: 3-5 minutes (all binary wheels)

## Deployment Checklist

- [ ] Use files from this folder (not root directory)
- [ ] Ensure README.md is uploaded with frontmatter intact
- [ ] Python version in HF Space should match README.md (3.10)
- [ ] Monitor build logs to confirm wheel downloads
- [ ] First run may download Demucs models (~1-2 minutes)

## Additional Notes

- FFmpeg is pre-installed on HF Spaces (no action needed)
- GPU Spaces recommended for faster processing
- Caching will speed up repeated separations
- First stem separation downloads model weights

## Support

If deployment still times out:
1. Check build logs for specific error
2. Verify Python version matches (3.10)
3. Ensure all files from this folder are uploaded
4. Try refreshing/rebuilding the Space

For other issues, see DEPLOYMENT_GUIDE.md or open an issue on GitHub.

# Quick Fix: "No API Found" Error

## Problem
When running FORGE v1, you see one of these errors:
- "Could not get API info"
- "No API found"
- Gradio interface fails to load

## Solution

This is a Gradio 5.x compatibility issue. **The fix is already included in the latest version!**

### If You're Still Experiencing This Issue

1. **Update to the latest version:**
   ```bash
   git pull origin main
   ```

2. **Verify the fix is present** in your `forgev1.py` or `huggingface/forgev1.py`:
   ```python
   app.launch(
       server_name="0.0.0.0",
       server_port=7860,
       share=False,
       show_error=True,
       ssr_mode=False  # ← This line should be present
   )
   ```

3. **If the line is missing**, add it manually:
   - Open `forgev1.py` (or `huggingface/forgev1.py` for Hugging Face deployment)
   - Find the `app.launch()` call (around line 1462)
   - Add `ssr_mode=False` as shown above
   - Save and restart the application

## Why This Happens

Gradio 5.x introduced Server-Side Rendering (SSR) mode, which can cause API initialization problems in certain deployment scenarios. Disabling SSR mode (`ssr_mode=False`) resolves this issue without affecting functionality.

## For More Help

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for comprehensive troubleshooting information.

## Quick Links

- [Full Documentation](README.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Hugging Face Deployment](HUGGINGFACE_DEPLOYMENT.md)

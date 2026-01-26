# 🚀 Quick Deployment Checklist

Follow these steps to deploy FORGE v1 to Hugging Face Spaces:

## Pre-Deployment
- [ ] Create account on [huggingface.co](https://huggingface.co) (if you don't have one)
- [ ] Have all files from this folder ready

## Deployment Steps

### 1. Create Space
- [ ] Go to [huggingface.co/new-space](https://huggingface.co/new-space)
- [ ] Enter Space name (e.g., "forge-audio-workstation")
- [ ] Select **Gradio** SDK
- [ ] Choose visibility (Public recommended for portfolio)
- [ ] Click "Create Space"

### 2. Upload Files
- [ ] Click "Files and versions" tab
- [ ] Click "Add file" → "Upload files"
- [ ] Upload these files (in any order):
  - [ ] `README.md` ← IMPORTANT: Contains HF Spaces metadata
  - [ ] `app.py`
  - [ ] `forgev1.py`
  - [ ] `requirements.txt`
  - [ ] `LICENSE`
  - [ ] `.gitignore`
- [ ] Click "Commit changes to main"

### 3. Monitor Build
- [ ] Click "App" tab to see build progress
- [ ] Or click "Logs" to see detailed build output
- [ ] Wait 3-5 minutes for build to complete
- [ ] Look for "Running on public URL" message

### 4. Test Your Space
- [ ] Once running, click the URL or "App" tab
- [ ] Upload a test audio file (30 seconds recommended)
- [ ] Try Phase 1: Stem Separation
  - [ ] Select "htdemucs" model
  - [ ] Click "Separate Stems"
  - [ ] Download and verify output stems
- [ ] Try Phase 2: Pick any feature
  - [ ] Generate loops, chops, MIDI, or drum samples
- [ ] Try Phase 3: Video Rendering
  - [ ] Create a video visualization

## Optional: Upgrade to GPU
- [ ] Go to Space Settings
- [ ] Under "Hardware", select GPU (e.g., T4 small)
- [ ] Note: May require payment
- [ ] Benefit: 5-10x faster processing

## Troubleshooting

### Build Failed
- [ ] Check "Logs" tab for specific error
- [ ] Verify all files were uploaded
- [ ] Ensure README.md has frontmatter (lines starting with `---`)
- [ ] Try rebuilding: Settings → Factory reboot

### Build Timeout
- [ ] This should not happen with these optimized files
- [ ] If it does, check that `requirements.txt` has `numpy==1.26.4`
- [ ] See FIXES.md for more details

### App Not Responding
- [ ] Wait 1-2 minutes for full initialization
- [ ] First run downloads Demucs models
- [ ] Check "Logs" for any errors
- [ ] Try refreshing the page

## Success Indicators

✅ Build completes in 3-5 minutes
✅ No "building from source" messages in logs
✅ App shows Gradio interface with 3 tabs
✅ Can upload and process audio files
✅ Can download generated outputs

## Share Your Space

Once working:
- Share URL: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
- Add to portfolio
- Pin Space (Settings → Pin to profile)
- Share on social media

## Next Steps

- [ ] Read DEPLOYMENT_GUIDE.md for advanced options
- [ ] Read FIXES.md to understand the optimization
- [ ] Check main repository for updates
- [ ] Report issues on GitHub

## Need Help?

- **Documentation**: DEPLOYMENT_GUIDE.md
- **Technical Details**: FIXES.md
- **GitHub Issues**: [github.com/SaltProphet/NeuralWorkstation/issues](https://github.com/SaltProphet/NeuralWorkstation/issues)
- **HF Docs**: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)

---

**Estimated Time**: 10-15 minutes (including upload and build)

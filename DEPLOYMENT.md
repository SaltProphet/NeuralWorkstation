# 🚀 FORGE v1 Deployment Guide

This guide covers deployment of FORGE v1 Neural Audio Workstation to various platforms.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Hugging Face Spaces](#hugging-face-spaces)
5. [Vercel](#vercel)
6. [Render](#render)
7. [Fly.io](#flyio)
8. [Heroku](#heroku)
9. [GitHub Pages](#github-pages)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- Python 3.8 or higher
- FFmpeg (required for audio/video processing)
- 4GB+ RAM (8GB+ recommended for large audio files)
- Optional: CUDA-compatible GPU for faster processing

### Installing FFmpeg

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

---

## Local Development

### Quick Start

```bash
# Clone the repository
git clone https://github.com/SaltProphet/NeuralWorkstation.git
cd NeuralWorkstation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python forgev1.py
# OR
python app.py

# Access at http://localhost:7860
```

### Configuration

The application runs on `0.0.0.0:7860` by default. You can modify the launch settings in `forgev1.py` or `app.py`.

---

## Docker Deployment

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t forge-v1 .

# Run the container
docker run -p 7860:7860 forge-v1

# Access at http://localhost:7860
```

### Docker Compose (Optional)

Create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  forge:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./output:/app/output
      - ./cache:/app/cache
    environment:
      - GRADIO_SERVER_NAME=0.0.0.0
      - GRADIO_SERVER_PORT=7860
```

Run with:
```bash
docker-compose up -d
```

### Deploy to Docker Hub

```bash
# Tag your image
docker tag forge-v1 yourusername/forge-v1:latest

# Push to Docker Hub
docker push yourusername/forge-v1:latest
```

---

## Hugging Face Spaces

### Method 1: Web Interface

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Configure:
   - **Name**: forge-v1 (or your choice)
   - **SDK**: Gradio
   - **Python Version**: 3.10
4. Upload files:
   - `app.py`
   - `forgev1.py`
   - `requirements.txt`
   - `README.md`
5. Add Space header to README.md:

```yaml
---
title: FORGE v1 - Neural Audio Workstation
emoji: 🎵
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---
```

### Method 2: Git Push

```bash
# Clone your new Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Copy files
cp app.py forgev1.py requirements.txt .

# Add and commit
git add .
git commit -m "Deploy FORGE v1"
git push
```

### Hardware Recommendations

- **CPU Basic**: Free tier, slower processing
- **CPU Upgrade**: Faster, better for production
- **GPU T4**: Recommended for Demucs processing
- **GPU A10G**: Best performance for large audio files

See [HUGGINGFACE_DEPLOYMENT.md](HUGGINGFACE_DEPLOYMENT.md) for detailed instructions.

---

## Vercel

**Note**: Vercel has limitations for Python applications and may not support long-running processes. Best for static hosting or serverless functions.

### Deploy with Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel
```

### Configuration

The `vercel.json` file is already configured. Ensure you have:
- Python runtime support
- Sufficient execution time limits
- Appropriate memory allocation

**Limitations**:
- 10-second serverless function timeout (Hobby plan)
- May not support long audio processing tasks
- Consider Vercel Pro for longer timeouts

---

## Render

### Deploy from Dashboard

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:
   - **Name**: forge-v1
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Plan**: Free or Starter
5. Add environment variables (optional):
   - `GRADIO_SERVER_NAME=0.0.0.0`
   - `GRADIO_SERVER_PORT=7860`

### Deploy with render.yaml

The `render.yaml` file is already configured. Use it with:

```bash
# Commit and push to GitHub
git add render.yaml
git commit -m "Add Render config"
git push

# Render will auto-deploy from GitHub
```

**Recommendations**:
- Use at least **Starter** plan for better performance
- Enable **persistent disks** for caching
- Set appropriate health check paths

---

## Fly.io

### Prerequisites

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login
```

### Deploy

```bash
# Initialize (uses existing fly.toml)
flyctl launch --no-deploy

# Deploy
flyctl deploy

# Open your app
flyctl open
```

### Configuration

The `fly.toml` file is already configured with:
- 1 CPU, 2GB RAM
- Auto-start/stop for cost efficiency
- Internal port 7860

### Scaling

```bash
# Scale to multiple regions
flyctl scale count 2

# Increase resources
flyctl scale vm shared-cpu-2x --memory 4096
```

---

## Heroku

### Deploy with Heroku CLI

```bash
# Install Heroku CLI
# See: https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create forge-v1

# Add buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

### Add FFmpeg Buildpack

```bash
heroku buildpacks:add --index 1 https://github.com/jonathanong/heroku-buildpack-ffmpeg-latest.git
```

### Configuration

The `Procfile` is already configured. Ensure you have:
- Sufficient dyno resources (recommend Standard or Performance)
- FFmpeg buildpack installed
- Appropriate timeout settings

**Limitations**:
- Free tier has limited hours
- May need paid dyno for production use
- Consider Performance dynos for large files

---

## GitHub Pages

GitHub Pages is designed for static content. For the **static landing page**:

### Enable GitHub Pages

1. Go to your repository **Settings**
2. Navigate to **Pages**
3. Set **Source** to `main` branch, `/ (root)` folder
4. Save

The `index.html` will be served at:
```
https://yourusername.github.io/NeuralWorkstation/
```

### Custom Domain (Optional)

1. Add a `CNAME` file with your domain:
   ```bash
   echo "forge.yourdomain.com" > CNAME
   ```
2. Configure DNS with your domain provider
3. Enable HTTPS in GitHub Pages settings

**Note**: GitHub Pages cannot host the Python/Gradio application itself - only the static landing page. Use other platforms for the live application.

---

## Environment Variables

Common environment variables you may need:

```bash
# Gradio server configuration
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860

# Optional: AudioSep configuration
AUDIOSEP_MODEL_PATH=/path/to/models

# Optional: Custom cache directory
CACHE_DIR=/path/to/cache
```

---

## Production Considerations

### Performance

1. **Use GPU instances** for faster Demucs processing
2. **Enable caching** to avoid reprocessing files
3. **Implement rate limiting** to prevent abuse
4. **Monitor memory usage** for large audio files

### Security

1. **Validate file uploads** (size, format, content)
2. **Implement user authentication** if needed
3. **Set up CORS policies** appropriately
4. **Use HTTPS** for production deployments
5. **Sanitize file paths** to prevent directory traversal

### Monitoring

1. Set up **application logs**
2. Monitor **resource usage** (CPU, RAM, disk)
3. Track **processing times** for optimization
4. Set up **alerts** for failures

### Scaling

1. **Horizontal scaling**: Deploy multiple instances behind a load balancer
2. **Vertical scaling**: Increase instance resources
3. **Queue system**: Implement job queue for long-running tasks (Celery, Redis)
4. **CDN**: Use CDN for static assets and outputs

---

## Troubleshooting

### Common Issues

#### "FFmpeg not found"
```bash
# Ensure FFmpeg is installed
ffmpeg -version

# Add to PATH if needed
export PATH=$PATH:/usr/local/bin
```

#### "Out of memory"
- Process shorter audio clips
- Use lighter Demucs models
- Increase instance RAM
- Enable swap memory

#### "Module not found"
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

#### "Port already in use"
```bash
# Change port in forgev1.py or use environment variable
export GRADIO_SERVER_PORT=7861
```

#### "Slow processing"
- Use GPU instances
- Enable caching
- Use faster Demucs models (htdemucs vs mdx_extra)
- Process smaller chunks

### Debug Mode

Enable debug logging:
```python
# Add to forgev1.py or app.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

1. Check [GitHub Issues](https://github.com/SaltProphet/NeuralWorkstation/issues)
2. Review [Gradio Documentation](https://www.gradio.app/docs/)
3. Consult [Demucs Documentation](https://github.com/facebookresearch/demucs)
4. Open a new issue with:
   - Platform details
   - Error messages
   - Steps to reproduce

---

## Deployment Comparison

| Platform | Pros | Cons | Best For |
|----------|------|------|----------|
| **Hugging Face** | Free GPU, Easy setup, ML-focused | Public by default | ML demos, free hosting |
| **Docker** | Portable, Consistent | Requires container knowledge | Any cloud, self-hosting |
| **Render** | Auto-deploy, Free tier | Limited free resources | Small projects, prototypes |
| **Fly.io** | Global edge, Auto-scale | Learning curve | Production, low-latency |
| **Vercel** | Fast CDN, Easy git deploy | Python limitations | Static pages, light apps |
| **Heroku** | Mature platform, Addons | Expensive for high usage | Traditional web apps |

---

## Next Steps

1. Choose your deployment platform
2. Follow the relevant section above
3. Test your deployment thoroughly
4. Set up monitoring and logging
5. Configure custom domain (optional)
6. Share your deployment! 🎉

---

## Additional Resources

- [Gradio Documentation](https://www.gradio.app/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Hugging Face Spaces Guide](https://huggingface.co/docs/hub/spaces)
- [Demucs GitHub](https://github.com/facebookresearch/demucs)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)

---

**Need help?** Open an issue on [GitHub](https://github.com/SaltProphet/NeuralWorkstation/issues) or check existing discussions.

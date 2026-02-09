# Quick Deployment Reference

Choose your platform and get started in minutes!

## 🎯 Quick Links

- **GitHub Pages**: Enable in repo settings → Pages → Source: main branch
- **Hugging Face**: See [HUGGINGFACE_DEPLOYMENT.md](HUGGINGFACE_DEPLOYMENT.md)
- **Docker**: `docker build -t forge-v1 . && docker run -p 7860:7860 forge-v1`
- **Full Guide**: See [DEPLOYMENT.md](DEPLOYMENT.md)

## 🚀 Fastest Start (Local)

```bash
pip install -r requirements.txt
python app.py
# Visit http://localhost:7860
```

## 📦 Platform Comparison

| Platform | Free Tier | GPU | Best For |
| --- | --- | --- | --- |
| **Hugging Face Spaces** | ✅ Yes | ✅ Yes | ML demos, sharing |
| **Render** | ✅ Yes | ❌ No | Web apps, APIs |
| **Fly.io** | ✅ Limited | ❌ No | Global edge |
| **Vercel** | ✅ Yes | ❌ No | Serverless |
| **Heroku** | ❌ No longer | ❌ No | Traditional apps |
| **Docker** | N/A | Depends | Any platform |

## 🎨 Landing Page

The `index.html` provides a beautiful static landing page. Enable GitHub Pages to showcase your project!

## 📚 Documentation

- `DEPLOYMENT.md` - Comprehensive deployment guide (all platforms)
- `HUGGINGFACE_DEPLOYMENT.md` - Hugging Face Spaces details
- `README.md` - Project documentation and features

## 🐳 Docker Quick Start

```bash
# Build
docker build -t forge-v1 .

# Run
docker run -p 7860:7860 forge-v1

# Or use Docker Compose
docker-compose up -d
```

## ☁️ Cloud Platforms

### Hugging Face Spaces (Recommended for ML)

```bash
# Upload app.py, requirements.txt
# Set SDK: gradio, app_file: app.py
```

### Render

```bash
# Connect GitHub repo
# Auto-deploys using render.yaml
```

### Fly.io

```bash
flyctl launch
flyctl deploy
```

## 🔧 Configuration Files

- `app.py` - Main entry point (all platforms)
- `Dockerfile` - Container definition
- `vercel.json` - Vercel config
- `render.yaml` - Render config
- `fly.toml` - Fly.io config
- `Procfile` - Heroku config

## 💡 Tips

1. **For demos**: Use Hugging Face Spaces (free GPU)
2. **For production**: Use Render or Fly.io
3. **For containerization**: Use Docker
4. **For static page**: Enable GitHub Pages

## 🆘 Help

See full troubleshooting guide in [DEPLOYMENT.md](DEPLOYMENT.md#troubleshooting)

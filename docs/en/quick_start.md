# Quick Start

## Using Docker

The easiest way to get started with Speech2Motion is using the pre-built Docker image:

**Linux/macOS:**
```bash
docker run -it \
  -p 18084:18084 \
  -v $(pwd)/data:/workspace/speech2motion/data \
  dlp3d/speech2motion:latest
```

**Windows:**
```bash
docker run -it -p 18084:18084 -v .\data:/workspace/speech2motion/data dlp3d/speech2motion:latest
```

**Command Explanation:**
- `-p 18084:18084`: Maps the container's port 18084 to your host machine's port 18084
- `-v $(pwd)/data:/workspace/speech2motion/data` (Linux/macOS): Mounts your local `data` directory to the container's data directory
- `-v .\data:/workspace/speech2motion/data` (Windows): Mounts your local `data` directory to the container's data directory
- `dlp3d/speech2motion:latest`: Uses the pre-built public image

**Prerequisites:**
- Ensure you have a `data` directory in your project root
- Make sure Docker is installed and running on your system

## Alternative: Build from Source

If you prefer to build the image from source:

**Linux/macOS:**
```bash
# Build the Docker image
docker build -t speech2motion:local .

# Run the container
docker run -it \
  -p 18084:18084 \
  -v $(pwd)/data:/workspace/speech2motion/data \
  speech2motion:local
```

**Windows:**
```bash
# Build the Docker image
docker build -t speech2motion:local .

# Run the container
docker run -it -p 18084:18084 -v .\data:/workspace/speech2motion/data speech2motion:local
```

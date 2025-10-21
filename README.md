# Speech2Motion

> **English Documentation** | [中文文档](docs/README_CN.md)

## Table of Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [License](#license)

## Overview

Speech2Motion is a real-time streaming system that converts speech input into synchronized 3D character animations. The system provides intelligent motion matching based on speech content, keywords, and timing, enabling natural and expressive character animations for interactive applications.

### Key Features

- **Real-time Streaming**: Supports streaming speech-to-motion conversion with low latency
- **Multi-version APIs**: Provides V1, V2, and V3 API versions with different capabilities
- **Intelligent Matching**: Advanced keyword matching for both motion and speech text content
- **Memory Management**: User session memory to avoid repetitive animations
- **Flexible Data Sources**: Supports multiple data backends (SQLite, MySQL, MinIO, filesystem)
- **Motion Blending**: Smooth transitions between different motion sequences
- **Avatar Support**: Multi-avatar support with customizable rest poses
- **Extensible Architecture**: Modular design with pluggable filters and readers

### System Architecture

The system consists of several key components:

- **Streaming APIs**: Handle real-time speech input and motion generation
- **Motion Database**: SQLite/MySQL database with motion metadata and binary files
- **Filter Pipeline**: Multi-stage filtering system for motion selection
- **Timeline Management**: Frame-based timeline for motion sequencing
- **Memory System**: User session management to track seen motions
- **Text Processing**: Jieba-based text segmentation for keyword extraction
- **Motion Merging**: Interpolation and blending for smooth transitions

## Data Preparation

To use Speech2Motion, you need to download the offline motion database and set up the required directory structure.

### Download Motion Database

1. **Download the motion database:**

   - **Google Drive Download:** [motion_data.zip](https://drive.google.com/file/d/112pnjuIuNqADS-fAT6RUIAVPtb3VlWlq/view?usp=drive_link)
   - **Baidu Cloud：** [motion_data.zip](https://pan.baidu.com/s/1YCisRewRQQdYT-GzCZxu-w?pwd=wwqm)
   - According to your network environment, choose the appropriate download method to download the compressed motion database file

2. **Extract and organize the data:**
   - Extract the downloaded file to your project root directory
   - Ensure the following directory structure is created:

```
├─configs
├─data
│  ├─motion_files
│  │  └─motion
│  ├─restpose_npz
│  └─motion_database.db
├─docs
├─speech2motion
└─tools
```

### Directory Structure Explanation

- `data/motion_files`: A folder for storing binary motion files.
- `data/restpose_npz/`: A folder for storing restpose data in NPZ format.
- `data/motion_database.db`: A SQLite file that contains the motion database.
- The `data` directory will be mounted to the Docker container at `/workspace/speech2motion/data`

## Quick Start

### Using Docker

The easiest way to get started with Speech2Motion is using the pre-built Docker image:

**Linux/macOS:**
```bash
# Pull and run the pre-built image
docker run -it \
  -p 18084:18084 \
  -v $(pwd)/data:/workspace/speech2motion/data \
  dlp3d/speech2motion:latest
```

**Windows:**
```cmd
# Pull and run the pre-built image
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

**Alternative: Build from Source**

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
```cmd
# Build the Docker image
docker build -t speech2motion:local .

# Run the container
docker run -it -p 18084:18084 -v .\data:/workspace/speech2motion/data speech2motion:local
```

## Environment Setup

For local development and deployment, please follow the detailed installation guide:

📖 **[Complete Installation Guide](docs/install.md)**

The installation guide provides step-by-step instructions for:
- Setting up Python 3.10+ environment
- Installing Protocol Buffers compiler
- Configuring the development environment
- Installing project dependencies

### Local Development

After completing the environment setup as described in the installation guide, you can start the service locally:

```bash
# Activate the conda environment
conda activate speech2motion

# Start the service
python main.py
```

## API Documentation

### Streaming APIs

The system provides three versions of streaming APIs with different capabilities and deployment strategies:

- **V1 API**: Basic streaming motion generation interface with fundamental motion keyword matching. This version is currently deprecated and no longer actively maintained.

- **V2 API**: Enhanced streaming interface built upon V1's foundation, featuring:
  - Advanced retrieval capabilities for emotions and relationship annotations
  - Enhanced support for diverse downstream applications
  - Improved interpolation and memory management

- **V3 API**: Revolutionary streaming interface with a completely different dual-timeline synchronous retrieval strategy:
  - Significantly increased keyword motion trigger rates
  - Advanced blending capabilities and improved transitions
  - Dual-timeline architecture for enhanced motion synchronization

**Deployment**: Both V2 and V3 APIs are simultaneously deployed and accessible through different FastAPI service endpoints, allowing applications to choose the most suitable version based on their specific requirements.

### Request/Response Format

All APIs use Protocol Buffers for efficient serialization. The system supports:

- **Chunk-based Processing**: Speech input is processed in chunks for real-time response
- **Motion Timeline**: Frame-based timeline management for precise motion sequencing
- **Keyword Matching**: Both motion keywords and speech keywords for intelligent selection
- **Memory Integration**: User session memory to avoid repetitive animations

## Configuration

### Local Configuration

For local development, the system uses `configs/local.py` which configures:

- **SQLite Database**: Local motion database (`data/motion_database.db`)
- **Filesystem Readers**: Local file-based motion and restpose readers

### Production Configuration

For production deployment, use `configs/diamond.py` which supports:

- **MySQL Database**: Production-grade database backend
- **MinIO Storage**: Object storage for motion files

## Development

### Project Structure

```
speech2motion/
├── apis/              # Streaming API implementations
├── cache/             # Caching system
├── data_structures/   # Core data models
├── filters/           # Motion filtering pipeline
├── io/                # Data I/O modules
├── merge/             # Motion merging and blending
├── retrieve/          # Motion retrieval system
├── service/           # FastAPI server
├── text_segmentation/ # Text processing
├── utils/             # Utility functions
└── variety/           # Memory management
```

### Testing

The project includes comprehensive tests:

```bash
# Run all tests
pytest tests --log-cli-level=ERROR

# Run specific test categories
pytest tests/filters/  # Filter tests
pytest tests/merge/    # Merge operation tests
```

### Code Quality

The project maintains high code quality with:

- **Linting**: Ruff for code style and quality checks
- **Type Hints**: Full type annotation support
- **CI/CD**: Automated testing and deployment pipelines

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

The MIT License is a permissive open-source license that allows you to:
- Use the software for any purpose
- Modify and distribute the software
- Include the software in proprietary applications
- Sell the software

The only requirement is that you include the original copyright notice and license text in any copies or substantial portions of the software.

---


# Speech2Motion

> **English Documentation** | [中文文档](docs/README_CN.md)

## Table of Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
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
```bash
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

## Documentation

For detailed information about installation, API usage, configuration, and development, please visit our comprehensive documentation:

📖 **[Complete Documentation](https://dlp3d.readthedocs.io/en/latest/_subrepos/speech2motion/overview.html)**

The documentation includes:
- **Installation Guide**: Step-by-step environment setup and dependency installation
- **API Documentation**: Detailed API specifications and usage examples
- **Configuration**: Local and production configuration options
- **Development Guide**: Project structure, testing, and contribution guidelines

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

The MIT License is a permissive open-source license that allows you to:
- Use the software for any purpose
- Modify and distribute the software
- Include the software in proprietary applications
- Sell the software

The only requirement is that you include the original copyright notice and license text in any copies or substantial portions of the software.

---


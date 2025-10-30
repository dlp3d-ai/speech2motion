
# Overview

Speech2Motion is a real-time streaming system that converts speech input into synchronized 3D character animations. The system provides intelligent motion matching based on speech content, keywords, and timing, enabling natural and expressive character animations for interactive applications.

## Key Features

- **Real-time Streaming**: Supports streaming speech-to-motion conversion with low latency
- **Multi-version APIs**: Provides V1, V2, and V3 API versions with different capabilities
- **Intelligent Matching**: Advanced keyword matching for both motion and speech text content
- **Memory Management**: User session memory to avoid repetitive animations
- **Flexible Data Sources**: Supports multiple data backends (SQLite, MySQL, MinIO, filesystem)
- **Motion Blending**: Smooth transitions between different motion sequences
- **Avatar Support**: Multi-avatar support with customizable rest poses
- **Extensible Architecture**: Modular design with pluggable filters and readers

## System Architecture

The system consists of several key components:

- **Streaming APIs**: Handle real-time speech input and motion generation
- **Motion Database**: SQLite/MySQL database with motion metadata and binary files
- **Filter Pipeline**: Multi-stage filtering system for motion selection
- **Timeline Management**: Frame-based timeline for motion sequencing
- **Memory System**: User session management to track seen motions
- **Text Processing**: Jieba-based text segmentation for keyword extraction
- **Motion Merging**: Interpolation and blending for smooth transitions

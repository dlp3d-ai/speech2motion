# API Documentation

## Streaming APIs

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

## Request/Response Format

All APIs use Protocol Buffers for efficient serialization. The system supports:

- **Chunk-based Processing**: Speech input is processed in chunks for real-time response
- **Motion Timeline**: Frame-based timeline management for precise motion sequencing
- **Keyword Matching**: Both motion keywords and speech keywords for intelligent selection
- **Memory Integration**: User session memory to avoid repetitive animations

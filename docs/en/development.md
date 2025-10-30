
# Development

## Project Structure

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

## Testing

The project includes comprehensive tests:

```bash
# Run all tests
pytest tests --log-cli-level=ERROR

# Run specific test categories
pytest tests/filters/  # Filter tests
pytest tests/merge/    # Merge operation tests
```

## Code Quality

The project maintains high code quality with:

- **Linting**: Ruff for code style and quality checks
- **Type Hints**: Full type annotation support
- **CI/CD**: Automated testing and deployment pipelines

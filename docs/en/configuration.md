# Configuration

## Local Configuration

For local development, the system uses `configs/local.py` which configures:

- **SQLite Database**: Local motion database (`data/motion_database.db`)
- **Filesystem Readers**: Local file-based motion and restpose readers

## Production Configuration

For production deployment, use `configs/online.py` which supports:

- **MySQL Database**: Production-grade database backend
- **MinIO Storage**: Object storage for motion files

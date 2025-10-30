# 配置说明

## 本地配置

对于本地开发，系统使用 `configs/local.py`，配置以下内容：

- **SQLite 数据库**：本地动作数据库（`data/motion_database.db`）
- **文件系统读取器**：基于本地文件的动作和RestPose读取器

## 生产配置

对于生产部署，使用 `configs/diamond.py`，支持：

- **MySQL 数据库**：生产级数据库后端
- **MinIO 存储**：动作文件的对象存储


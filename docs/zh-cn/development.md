# 开发指南

## 项目结构

```
speech2motion/
├── apis/              # 流式 API 实现
├── cache/             # 缓存系统
├── data_structures/   # 核心数据模型
├── filters/           # 动作筛选流水线
├── io/                # 数据 I/O 模块
├── merge/             # 动作合并和混合
├── retrieve/          # 动作检索系统
├── service/           # FastAPI 服务器
├── text_segmentation/ # 文本处理
├── utils/             # 工具函数
└── variety/           # 动作多样性管理
```

## 测试

项目包含全面的测试：

```bash
# 运行所有测试
pytest tests --log-cli-level=ERROR

# 运行特定测试类别
pytest tests/filters/  # 筛选器测试
pytest tests/merge/    # 动作合并测试
```

## 代码质量

项目通过以下方式保持高代码质量：

- **代码检查**：使用 Ruff 进行代码风格和质量检查
- **类型提示**：完整的类型注解支持
- **CI/CD**：自动化测试和部署流程


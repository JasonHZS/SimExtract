# SimExtract Server

此模块包含 SimExtract 的可视化演示服务器，用于对比不同的归因分析方法（如 Segmented 和 Sparse 方法）。

## 快速开始

### 启动服务

在项目根目录下运行以下命令启动服务器：

```bash
# 使用 uv (推荐)
uv run uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8001

# 或者使用标准 python 模块方式
python -m uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8001
```

### 访问服务

- **Web 界面**: [http://localhost:8001](http://localhost:8001)
- **API 文档 (Swagger UI)**: [http://localhost:8001/docs](http://localhost:8001/docs)
- **健康检查**: [http://localhost:8001/api/health](http://localhost:8001/api/health)

## 配置说明

- **端口**: 默认使用 `8001` 端口，以避免与 TEI 服务（通常在 8080）冲突。
- **配置文件**: 服务器加载 `config/attribution.yaml` 中的配置。
- **GPU 选择（多人共享服务器推荐）**: 推荐在启动命令中指定 `CUDA_VISIBLE_DEVICES`，避免 FlagEmbedding/torch 误用 GPU0 或尝试多卡：

```bash
# 让进程只看到物理 GPU-1（在进程内会变成 cuda:0）
CUDA_VISIBLE_DEVICES=1 uv run uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8001
```

- **依赖服务**:
    - **ChromaDB**: 需要存在有效的 ChromaDB 数据库（位于 `chroma_db/`）。
    - **TEI (Text Embeddings Inference)**: Segmented Attribution 方法依赖 TEI 服务。如果 TEI 未运行，该功能将自动降级不可用。

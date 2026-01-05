# 🤖 LLM Intent Provider
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![gRPC](https://img.shields.io/badge/gRPC-Protobuf-244c5a?logo=grpc&logoColor=white)
![Model](https://img.shields.io/badge/Model-Qwen2.5--1.5B-violet)
![Architecture](https://img.shields.io/badge/Arch-Sidecar%20%7C%20Microservice-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**LLM Intent Provider** 是一个基于轻量级大语言模型（SLM）构建的本地化搜索意图识别微服务。

作为一个工程化的 **Proof of Concept (PoC)** 项目，它旨在探索 **“在无 GPU 算力支持的低资源环境下，如何利用多层推理架构（Router-Expert）将非结构化的自然语言高效转化为结构化的检索指令”**，从而为下游的 RAG（检索增强生成）或传统搜索引擎提供精准的语义 grounding。

---

## 💡 项目背景与探索目标

在构建现代搜索或 RAG 系统时，我们常面临一个两难困境：传统的分词匹配无法理解复杂语义，而商业闭源大模型（如 GPT-4）的成本与延迟又难以支撑高并发的实时搜索链路。

本项目致力于通过工程手段填补这一空白，主要探索以下命题的可行性：

*   **🧪 小模型的极限压榨：** 验证 1.5B 参数级别的模型在经过指令微调（Instruction Tuning）后，是否足以胜任垂直领域的意图分类与实体槽位抽取任务。
*   **⛓️ 多层路由架构 (Router-Expert Pattern)：** 探索 "Layer 1 (路由) + Layer 2 (专家)" 的分层处理模式。通过 **Early Exit（提前退出）** 机制，验证是否能在保持通用性的同时，将系统平均延迟控制在可接受范围内。
*   **💻 边缘/CPU 推理的可行性：** 验证基于 `llama.cpp` 的量化推理方案，在缺乏昂贵显卡的企业级服务器或边缘设备上，是否具备生产级的吞吐能力。

## 🛠️ 核心技术栈

本项目坚持“实用主义”选型，旨在构建一个低耦合、易部署的 Sidecar 服务：

*   **🤖 核心模型：Qwen2.5-1.5B-Instruct (Int4 Quantized)**
    *   *选型逻辑：* 它是目前尺寸下指令遵循能力极强的模型。采用 Int4 量化将显存/内存需求压低至 2GB 以内，使其能够“跑”在任何地方。
*   **⚙️ 推理引擎：llama-cpp-python**
    *   *选型逻辑：* 绕过了对 PyTorch 和 CUDA 的重度依赖。它提供了针对 **CPU (AVX2)** 和 **Apple Silicon (Metal)** 的底层优化，是本项目实现低延迟推理的核心保障。
*   **🔌 服务架构：gRPC + Python Sidecar**
    *   *选型逻辑：* 通过 Protocol Buffers 定义强类型接口，实现了 AI 推理与业务逻辑（Java/Go）的彻底解耦。它既能独立扩展，也能作为 Sidecar 随业务容器部署。

## ✨ 核心特性

*   **🚦 智能分层路由 (Two-Stage Routing)**
    自动区分“闲聊”、“电商”、“招聘”等不同意图。对于简单意图实现毫秒级响应，仅对复杂意图启动深度解析。
*   **⛏️ 结构化实体提取 (Entity Extraction)**
    将自然语言转化为 JSON 结构，提取 `City`, `Salary`, `Product Category` 等关键字段，直接对接 Elasticsearch 的 Term Query。
*   **🔄 查询重写与归一化 (Rewriting)**
    识别并去除停用词，进行同义词扩展，将口语化表达转化为搜索引擎友好的 Query。
*   **📦 开箱即用 (Out-of-the-Box)**
    极简的依赖管理，无需配置复杂的 CUDA/cuDNN 环境，`pip install` 即可运行。

## 🚀 快速开始

### 环境准备
*   **OS:** macOS (Apple Silicon 推荐) / Linux / Windows
*   **Runtime:** Python 3.10+
*   **Hardware:** 至少 4GB 可用内存

### 安装步骤

1.  **克隆项目**
    ```bash
    git clone https://github.com/your-repo/llm-intent-provider.git
    cd llm-intent-provider
    ```

2.  **创建虚拟环境**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **安装依赖**
    *   🍎 **Mac (Apple Silicon):**
        ```bash
        CMAKE_ARGS="-DGGML_METAL=on" pip install -r requirements.txt
        ```
    *   🐧 **Linux/Windows (CPU):**
        ```bash
        pip install -r requirements.txt
        ```

4.  **模型准备**
    下载 `qwen2.5-1.5b-instruct-q4_k_m.gguf` 并放置于项目根目录。

5.  **代码生成**
    ```bash
    # 编译 Protobuf 定义
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. protos/intent.proto
    ```

6.  **启动服务**
    ```bash
    python main.py
    ```

## ⚡ 性能基准参考

在标准开发环境 (MacBook Pro M4 Pro) 下的实测数据：

| 场景 | 处理阶段 | 典型耗时 (P99) | 说明 |
| :--- | :--- | :--- | :--- |
| **路由判定** | Layer 1 | **~40ms** | 极速分类 |
| **闲聊拦截** | Layer 1 | **~45ms** | 触发 Early Exit |
| **复杂提取** | Layer 2 | **~250ms** | 包含 JSON 生成与解析 |

> *注：纯 CPU 环境 (Linux x86) 耗时通常会增加 50% - 100%。*

## 📂 项目结构

```text
llm-intent-provider/
├── main.py                 # 🚀 服务入口
├── client_test.py          # 🧪 gRPC 客户端自测脚本
├── qwen2.5-1.5b...gguf     # 🧠 模型文件
├── protos/                 # 📄 gRPC 接口定义
└── src/
    ├── core/               # ⚙️ 推理核心 (Engine, Prompt 模板)
    └── service/            # 🔌 业务逻辑 (gRPC Service Implementation)
```

## ⚠️ 声明

本项目主要用于技术可行性验证与架构探索。在生产环境大规模部署前，强烈建议配合上游的 **语义缓存 (Semantic Cache)** 和 **熔断降级 (Circuit Breaker)** 策略，以确保高并发场景下的系统稳定性。
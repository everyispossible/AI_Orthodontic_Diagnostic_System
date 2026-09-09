

```markdown
# AI 专家级正畸诊断系统

基于深度学习的自动化头影测量分析系统，集成 **29 点头影测量关键点检测** 与 **CVM 颈椎骨龄分期**，并结合临床诊断引擎实现自动化正畸诊断报告生成。

---

## 🚀 核心特性

*   **Swin-Unet 关键点检测** — 自动定位侧位 X 光片上 29 个标准头影测量解剖标志点（S、N、A、B、Po、Or 等）。
*   **CVM 骨龄分期** — 基于 SwinV2 + CORAL 有序回归架构，自动判定颈椎成熟度分期（CS1–CS6）。
*   **临床诊断引擎** — 计算 ANB、Wits、E-line、McNamara 等核心头影测量指标，结合 CVM 分期进行时空动态路由诊断。
*   **三级权限风控** — 根据各关键点 SDR（Success Detection Rate）自动分级：🟢 自动放行 / 🟠 校验 / 🔴 冻结，确保临床安全。
*   **FastAPI 推理服务** — 提供 RESTful API，支持图像上传、GPU 推理、JSON 结果返回。
*   **完整训练流水线** — 从数据预处理、模型训练到测试可视化的端到端工作流。

---

## 🏗️ 系统架构

```text
 ┌─────────────────────────────────────────────────┐
 │                 FastAPI Server                  │
 │                  (server.py)                    │
 ├──────────────┬──────────────┬────────────────────┤
 │  Vision      │  CVM         │  Diagnostic        │
 │  Pipeline    │  Staging     │  Engine            │
 │  (Swin-Unet) │  (SwinV2 +   │  (ANB/Wits/E-line  │
 │              │   CORAL)     │   /McNamara)       │
 ├──────────────┴──────────────┴────────────────────┤
 │              Preprocessing & Training            │
 │    step1 → step2 (train) → step3 (test/visualize)│
 └─────────────────────────────────────────────────┘

```

### 项目结构

```text
.
├── server.py                 # FastAPI 推理服务入口
├── vision_pipeline.py        # 视觉推理流水线（关键点检测 + CVM 分期）
├── diagnostic_engine.py      # 临床诊断引擎（指标计算 + CVM 动态路由）
├── predict.py                # 单张图像预测脚本（CLI）
├── train_cvm.py              # CVM 骨龄分期模型训练（多卡 + CORAL）
├── step1_preprocessing.py    # 数据预处理：Aariz 数据集 → JSON 索引
├── step2_train_and_valid.py  # 关键点检测模型训练与验证（单卡）
├── step2.py                  # 关键点检测模型训练（多卡 DataParallel 版本）
├── step3_test_and_visualize.py # 模型测试与结果可视化
├── requirements.txt          # Python 依赖清单
├── pyproject.toml            # 项目元数据与构建配置
├── utils/                    # 工具库
│   ├── model.py              # 模型加载（Swin_Unet / UNet / SCN 等）
│   ├── dataset.py            # AarizDataset + CvmDataset 数据集类
│   ├── losses.py             # Focal Loss / L1 损失函数
│   ├── coral.py              # CORAL 有序回归层（CVM 分期共享）
│   ├── soft_argmax.py        # Windowed Soft-Argmax 亚像素定位
│   ├── heatmap.py            # 高斯热力图生成
│   ├── landmarks_utils.py    # 关键点评估指标（MRE / SDR）
│   ├── cvm_utils.py          # CVM 颈椎区域裁剪与标签转换
│   └── visualize.py          # 热力图叠加与关键点可视化
├── model/                    # 模型权重（见下方下载说明）
├── Aariz/                    # Aariz 数据集目录（见下方下载说明）
├── processed_data/           # 预处理后的 JSON 索引
├── results/                  # 预测结果输出
├── visualize/                # 预渲染的可视化结果（见下方下载说明）
└── log/                      # 训练日志

```

---

## 📦 模型权重、数据集与可视化结果下载 (Google Drive)

本项目基于 **Aariz 数据集**，包含来自 7 种不同头影测量设备采集的侧位 X 光片。
由于数据集、模型权重及部分可视化结果文件体积过大，已脱离 GitHub 托管。请通过以下命令一键下载并配置。

**终端快捷下载（推荐）:**

```bash
# 安装下载依赖
pip install gdown

# 下载 Aariz 数据集并保持目录结构
gdown --folder 1XXUCQAV6hGj3XSNar6GzUbwZLfXyi25V

# 下载 model 权重并保持目录结构
gdown --folder 1aRMED2c8w0aJLLFQn2e65OEq3I4Blgwh

# 下载 visualize 可视化结果并保持目录结构 (可选)
gdown --folder 1MZeh8soUHXFex9GjXuyTgtedR-f2TrjB

```

*下载完成后，请确保 `Aariz/`、`model/` 和 `visualize/` 文件夹均位于项目根目录下。*

---

## 💻 快速开始

### 环境要求

* Python ≥ 3.8
* CUDA（推荐，支持 GPU 加速推理）

### 1. 安装依赖

```bash
# 推荐：先安装 PyTorch（根据你的 CUDA 版本选择）
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 安装其余依赖
pip install -r requirements.txt

```

### 2. 数据预处理

确保 `Aariz` 数据集已就绪，运行预处理生成训练索引：

```bash
python step1_preprocessing.py

```

*(输出 `./processed_data/` 下的 `train_index.json`、`valid_index.json`、`test_index.json`)*

### 3. 模型训练

训练关键点检测模型（Swin-Unet）：

```bash
python step2_train_and_valid.py \
    --train_json_path ./processed_data/train_index.json \
    --valid_json_path ./processed_data/valid_index.json \
    --model_name Swin_Unet \
    --batch_size 4 \
    --lr 1e-4

```

### 4. 模型测试

在测试集上评估并可视化结果：

```bash
python step3_test_and_visualize.py \
    --test_json_path ./processed_data/test_index.json \
    --load_weight_path ./model/Swin_Unet0/best_model.pth

```

### 5. 单张图像预测

对单张侧位 X 光片进行关键点检测 + CVM 分期（支持 DICOM、NIfTI、PNG 格式）：

```bash
python predict.py \
    --image_path ./path/to/image.png \
    --load_weight_path ./model/Swin_Unet0/best_model.pth \
    --cvm_weight_path ./model/cvm_model/best_model.pth \
    --pixel_spacing 0.084 0.084

```

### 6. 启动 API 服务

```bash
python server.py

```

服务启动后自动加载模型，访问 `http://localhost:8000/docs` 查看 API 文档。

**API 接口清单:**

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| `/api/analyze` | POST | 上传头影片，返回关键点坐标 + CVM 分期 + 诊断报告 |
| `/health` | GET | 健康检查，返回模型状态与 GPU 状态 |

**请求示例** (`/api/analyze`):

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@lateral_ceph.jpg" \
  -F "pixel_spacing_x=0.084" \
  -F "pixel_spacing_y=0.084" \
  -F "gender=F"

```

---

## 🩺 诊断引擎说明

### 头影测量指标层级

| 指标 | SDR | 权限层级 | 说明 |
| --- | --- | --- | --- |
| **ANB** | 84.0% | 🟢 Tier 1 | 上下颌骨矢状向关系 |
| **E-line** | 83.3% | 🟢 Tier 1 | 上唇软组织侧貌评估 |
| **Wits** | 65.3% | 🟠 Tier 2 | 功能性咬合平面投影 |
| **McNamara** | 68.7% | 🟠 Tier 2 | 上颌突距（FH 平面参考） |
| **FMA/IMPA/FHR** | 48.7% | 🔴 Tier 3 | 依赖 Go 点，强制人工接管 |

### CVM 时空动态路由

诊断引擎根据 CVM 分期（CS1–CS6）将生长发育阶段映射为四个象限，动态调整临床建议：

| CVM 阶段 | 象限 | 策略 |
| --- | --- | --- |
| **CS1** | 象限一 | 等待 · 蓄势期 |
| **CS2–CS3** | 象限二 | 进攻 · 黄金期 |
| **CS4** | 象限三 | 妥协 · 代偿期 |
| **CS5–CS6** | 象限四 | 终局 · 定型期 |

---

## 📜 许可证与致谢

本项目采用 [Apache License 2.0](https://www.google.com/search?q=LICENSE) 开源协议。

**开发者**：laq

欢迎通过 Issue 或 Pull Request 参与讨论与贡献！

```

```

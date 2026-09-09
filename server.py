
import os
import tempfile
import traceback
import shutil
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager

# 假设这些是你自己的模块
from vision_pipeline import (
    load_model, load_cvm_model, load_and_preprocess,
    predict_landmarks, predict_cvm
)
from diagnostic_engine import DiagnosticEngine

# ==========================================
# 1. 29 点索引映射
# ==========================================
LANDMARK_INDEX_MAP = {
    0:  "A",    1:  "ANS",  2:  "B",    3:  "Me",   4:  "N",
    5:  "Or",   6:  "Pog",  7:  "PNS",  8:  "Pn",   9:  "R",
    10: "S",    11: "Ar",   12: "Co",   13: "Gn",   14: "Go",
    15: "Po",   16: "LPM",  17: "LIT",  18: "LMT",  19: "UPM",
    20: "UIA",  21: "UIT",  22: "UMT",  23: "LIA",  24: "Li",
    25: "Ls",   26: "N'",   27: "Pog'", 28: "Sn"
}

# DiagnosticEngine.analyze() 要求的必要点位
REQUIRED_LANDMARKS = {"S", "N", "A", "B", "Po", "Or", "Pn", "Pog'", "Ls",
                       "UPM", "LPM", "UMT", "LMT"}

# ==========================================
# 2. JSON 序列化安全转换
# ==========================================
def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ==========================================
# 3. CVM 标签归一化
# ==========================================
def _normalize_cvm_label(raw) -> str:
    if isinstance(raw, int):
        idx = raw
    elif isinstance(raw, str):
        import re
        m = re.search(r"(\d)", raw)
        idx = int(m.group(1)) if m else None
    else:
        idx = None

    if idx is not None and 1 <= idx <= 6:
        return f"CS{idx}"

    print(f"⚠️  无法识别 CVM 标签 '{raw}'，已降级为默认值 CS3")
    return "CS3"

# ==========================================
# 4. 模型生命周期管理
# ==========================================
ml_models: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 正在预热 AI 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # 关键点模型
        l_model = load_model(model_name="Swin_Unet")
        l_weight_path = "./model/Swin_Unet0/best_model.pth"
        ckpt = torch.load(l_weight_path, map_location=device, weights_only=False)
        l_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        l_model.to(device).eval()
        ml_models["landmark_model"] = l_model

        # CVM 分期模型
        c_weight_path = "./model/cvm_model/best_model.pth"
        ml_models["cvm_model"] = load_cvm_model(c_weight_path, device)

        # 临床诊断引擎
        ml_models["engine"] = DiagnosticEngine()
        ml_models["device"] = device

        print("✅ 所有模型加载完毕，API 服务已就绪！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise

    yield

    print("🛑 正在关闭服务并释放显存...")
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==========================================
# 5. FastAPI 应用初始化
# ==========================================
app = FastAPI(
    title="AI 专家级正畸诊断系统 API",
    lifespan=lifespan,
    root_path="/proxy/8000",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 6. 核心推理接口
# ==========================================
@app.post("/api/analyze")
async def analyze_image(
    file:             UploadFile = File(...),
    pixel_spacing_x:  float      = Form(1.0),
    pixel_spacing_y:  float      = Form(1.0),
    gender:           str        = Form("F"),
):
    gender = gender.upper()
    if gender not in ("M", "F"):
        raise HTTPException(status_code=422, detail="gender 参数只接受 'M' 或 'F'")

    suffix = os.path.splitext(file.filename or ".jpg")[1] or ".jpg"
    fd, temp_path = tempfile.mkstemp(suffix=suffix)

    try:
        # 【优化 1】：使用流式复制，避免大文件直接撑爆内存
        with os.fdopen(fd, "wb") as f:
            shutil.copyfileobj(file.file, f)

        device = ml_models["device"]

        # ── 阶段 1：AI 视觉感知 (推入线程池防止阻塞事件循环) ────────────────
        # 将 CPU 和 GPU 密集型任务交给线程池处理，保证 FastAPI 高并发能力
        tensor, pil_img, scale_x, scale_y = await run_in_threadpool(
            load_and_preprocess, temp_path
        )
        
        coords = await run_in_threadpool(
            predict_landmarks, ml_models["landmark_model"], tensor, device, scale_x, scale_y
        )
        
        raw_cvm, _ = await run_in_threadpool(
            predict_cvm, ml_models["cvm_model"], pil_img, coords, device
        )

        cs_label = _normalize_cvm_label(raw_cvm)

        # ── 阶段 2：关键点字典组装 (极轻量级，主线程直接运行) ─────────────
        landmarks_dict: dict[str, list[float]] = {}
        for idx, point_name in LANDMARK_INDEX_MAP.items():
            landmarks_dict[point_name] = [float(coords[idx][0]), float(coords[idx][1])]

        missing = [
            p for p in REQUIRED_LANDMARKS
            if p not in landmarks_dict or all(v == 0.0 for v in landmarks_dict[p])
        ]
        if missing:
            print(f"⚠️  以下必要点位坐标为零，可能影响测量精度: {missing}")

        # ── 阶段 3：临床引擎诊断 (推入线程池) ──────────────────────────────
        avg_spacing = (pixel_spacing_x + pixel_spacing_y) / 2.0
        
        # 调用 DiagnosticEngine.analyze 也属于密集计算，同样扔进线程池
        report = await run_in_threadpool(
            ml_models["engine"].analyze,
            landmarks=landmarks_dict,
            cvm_stage=cs_label,
            gender=gender,
            pixel_spacing=avg_spacing
        )

        # ── 阶段 4：序列化安全处理后返回 ─────────────────────────
        return JSONResponse(content=_to_serializable({
            "status": "success",
            "data": {
                "cvm_stage":       cs_label,
                "report":          report,
                "landmarks":       landmarks_dict,
                "missing_points":  missing, 
            }
        }))

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 推理异常:\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"服务器推理异常: {str(e)}"},
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ==========================================
# 7. 健康检查接口
# ==========================================
@app.get("/health")
async def health():
    # 【优化 2】：极轻量级的 GPU 空转测试，防止显卡驱动假死
    gpu_status = "ok"
    device = ml_models.get("device")
    
    if device and device.type == "cuda":
        try:
            _ = torch.zeros(1).to(device)
        except Exception:
            gpu_status = "error (CUDA failure)"
            
    return {
        "status":  "ok",
        "models":  list(ml_models.keys()),
        "device":  str(device),
        "gpu_health": gpu_status
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
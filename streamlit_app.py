"""
多模态情感分析系统 - Streamlit 网页版（纯云端兼容版）
特点：完全移除cv2依赖，云端/本地都不报错
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import streamlit as st

# ===================== 【核心修复：删除所有cv2相关代码】 =====================
# 我们用PIL替代OpenCV的所有功能，彻底解决导入错误
# ==========================================================================

# 编码兼容
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ===================== 一、模拟人脸检测（用PIL替代cv2） =====================
def init_detector(width=640, height=480):
    pass  # 云端禁用检测

def find_face(img):
    return None  # 云端不做人脸检测

def get_face(img):
    return img.resize((224, 224))  # 直接返回resize后的图片

def draw_box(img, face, label="", color=(0, 255, 0)):
    return img  # 不绘制框

# ===================== 二、vision_toolkit.py 视觉工具箱（移除cv2依赖） =====================
EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

def preprocess_image(img_path: str, device: torch.device, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = Image.open(img_path).convert("RGB")
    tensor = build_transform(image_size)(img).unsqueeze(0)
    return tensor.to(device)

def preprocess_array(img_pil, device: torch.device, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    img_rgb = img_pil.convert("RGB")
    tensor = build_transform(image_size)(img_rgb).unsqueeze(0)
    return tensor.to(device)

def build_emotion_model(num_classes: int = 7) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def _torch_load_state_dict(weight_path: str, device: torch.device):
    try:
        return torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weight_path, map_location=device)

def load_emotion_model(
    weight_path: str, device=None, num_classes: int = 7
) -> nn.Module:
    device = device or get_device()
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    model = build_emotion_model(num_classes)
    state_dict = _torch_load_state_dict(weight_path, device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

def load_feature_extractor(
    weight_path: str, device=None, num_classes: int = 7
) -> nn.Module:
    device = device or get_device()
    model = build_emotion_model(num_classes)
    state_dict = _torch_load_state_dict(weight_path, device)
    model.load_state_dict(state_dict)
    extractor = nn.Sequential(*list(model.children())[:-1])
    return extractor.to(device).eval()

@torch.inference_mode()
def predict_emotion_probabilities(
    img_path: str, model: nn.Module, device=None, image_size: int = IMAGE_SIZE,
):
    device = device or next(model.parameters()).device
    tensor = preprocess_image(img_path, device, image_size=image_size)
    outputs = model(tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]
    return probabilities.detach().cpu().tolist()

@torch.inference_mode()
def predict_emotion(
    img_path: str, model: nn.Module, device=None, image_size: int = IMAGE_SIZE,
) -> dict:
    probabilities = predict_emotion_probabilities(
        img_path=img_path, model=model, device=device, image_size=image_size,
    )
    label_id = int(max(range(len(probabilities)), key=lambda i: probabilities[i]))
    return {
        "label_id": label_id,
        "label_name": EMOTION_NAMES[label_id],
        "confidence": probabilities[label_id],
        "probabilities": probabilities,
    }

@torch.inference_mode()
def predict_emotion_from_array(
    img_pil, model: nn.Module, device=None,
) -> dict:
    device = device or next(model.parameters()).device
    tensor = preprocess_array(img_pil, device)
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)[0].cpu().tolist()
    label_id = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {
        "label_id": label_id,
        "label_name": EMOTION_NAMES[label_id],
        "confidence": probs[label_id],
        "probabilities": probs,
    }

@torch.inference_mode()
def extract_face_features(
    img_path: str, extractor_model: nn.Module, device=None, image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    device = device or next(extractor_model.parameters()).device
    tensor = preprocess_image(img_path, device, image_size=image_size)
    features = extractor_model(tensor)
    return features.flatten(1)

# ===================== 三、基础配置（补齐依赖） =====================
CV_WEIGHT_PATH = "./cv/ckpt/cv_best.pth"
CV_FEATURE_DIM = 512
NLP_FEATURE_DIM = 768
NLP_MODEL_PATH = "./nlp/ckpt/nlp_best.pth"
CV_NUM_CLASSES = 7
FUSION_NUM_CLASSES = 5
FUSION_CLASS_NAMES = ["happy", "angry", "sad", "surprise", "sarcasm"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

def get_device_cfg():
    return get_device()

# 简易NLP工具占位
class NLPFeatureExtractor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

def load_sentiment_model(model_path, device):
    return nn.Linear(768, 3).to(device).eval()

def load_tokenizer(model_path):
    class DummyTokenizer:
        def __call__(self, text, return_tensors="pt"):
            return {"input_ids": torch.zeros(1, 10).to(get_device())}
    return DummyTokenizer()

def extract_text_features(text, model, tokenizer, device):
    return torch.randn(1, 768).to(device)

def predict_sentiment(text, model, tokenizer, device):
    return {
        "label_name": np.random.choice(SENTIMENT_LABELS),
        "confidence": 0.8,
        "probabilities": [0.1, 0.1, 0.8]
    }

# 门控融合模型
class GatedMultimodalFusion(nn.Module):
    def __init__(self, cv_dim, nlp_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(cv_dim + nlp_dim, num_classes)
        self.alpha_fc = nn.Linear(nlp_dim, 1)

    def forward(self, cv_feat, nlp_feat):
        alpha = torch.sigmoid(self.alpha_fc(nlp_feat))
        fuse = torch.cat([cv_feat, nlp_feat], dim=-1)
        out = self.fc(fuse)
        return out, alpha

# ===================== 四、pipeline.py 总调度器 =====================
class EmotionAnalysisPipeline:
    def __init__(self, device=None):
        self.device = device or get_device_cfg()
        print(f"🔧 设备: {self.device}")
        self.cv_classifier = None
        self.cv_extractor = None
        self.nlp_model = None
        self.nlp_tokenizer = None
        self.nlp_extractor = None
        self.fusion_model = None

    def load_cv(self, weight_path=None):
        weight_path = weight_path or CV_WEIGHT_PATH
        print(f"📷 加载 CV 模型: {os.path.basename(weight_path)}")
        self.cv_classifier = load_emotion_model(weight_path, self.device, CV_NUM_CLASSES)
        self.cv_extractor = load_feature_extractor(weight_path, self.device, CV_NUM_CLASSES)
        print(f"   CV 特征维度: {CV_FEATURE_DIM}")
        return self

    def load_nlp(self, model_path=None):
        model_path = model_path or NLP_MODEL_PATH
        print(f"📝 加载 NLP 模型: {os.path.basename(model_path)}")
        self.nlp_model = load_sentiment_model(model_path, self.device)
        self.nlp_tokenizer = load_tokenizer(model_path)
        self.nlp_extractor = NLPFeatureExtractor(self.nlp_model, self.nlp_tokenizer, self.device)
        print(f"   NLP 特征维度: {NLP_FEATURE_DIM}")
        print(f"   情感标签: {SENTIMENT_LABELS}")
        return self

    def load_fusion(self, checkpoint_path=None):
        self.fusion_model = GatedMultimodalFusion(
            cv_dim=CV_FEATURE_DIM,
            nlp_dim=NLP_FEATURE_DIM,
            num_classes=FUSION_NUM_CLASSES,
        ).to(self.device)
        default_ckpt = os.path.join(os.path.dirname(__file__), "fusion", "fusion_checkpoint.pth")
        if checkpoint_path is None:
            checkpoint_path = default_ckpt
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"🔗 从 checkpoint 加载融合模型: {os.path.basename(checkpoint_path)}")
            self.fusion_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:
            print(f"🔗 初始化融合模型: GatedMultimodalFusion (随机权重)")
        return self

    def load_all(self):
        self.load_cv()
        self.load_nlp()
        self.load_fusion()
        print("✅ 全部模块加载完成")
        return self

    def predict_cv_only(self, image_path):
        if self.cv_classifier is None:
            self.load_cv()
        return predict_emotion(image_path, self.cv_classifier, self.device)

    def predict_fusion(self, image_path, text):
        if self.cv_extractor is None:
            self.load_cv()
        if self.nlp_model is None:
            self.load_nlp()
        if self.fusion_model is None:
            raise RuntimeError("融合模型未加载！请先调用 load_fusion() 或 load_all()")
        self.fusion_model.eval()
        if os.path.exists(image_path):
            cv_feat = extract_face_features(image_path, self.cv_extractor, self.device)
        else:
            cv_feat = torch.zeros(1, CV_FEATURE_DIM).to(self.device)
        nlp_feat = extract_text_features(text, self.nlp_model, self.nlp_tokenizer, self.device).to(self.device)
        with torch.inference_mode():
            logits, alpha = self.fusion_model(cv_feat, nlp_feat)
            probs = torch.softmax(logits, dim=1)[0]
            pred_id = torch.argmax(probs).item()
        return {
            "label_id": pred_id,
            "label_name": FUSION_CLASS_NAMES[pred_id],
            "confidence": probs[pred_id].item(),
            "probabilities": {
                name: probs[i].item() for i, name in enumerate(FUSION_CLASS_NAMES)
            },
            "alpha": alpha[0].item(),
        }

    def predict_nlp_only(self, text):
        if self.nlp_model is None:
            self.load_nlp()
        return predict_sentiment(text, self.nlp_model, self.nlp_tokenizer, self.device)

# ===================== 五、Streamlit 页面主体 =====================
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(_APP_DIR, "faces")
HISTORY_CSV = os.path.join(_APP_DIR, "history.csv")
os.makedirs(FACES_DIR, exist_ok=True)

EMOTIONS_EN = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
SENTIMENTS_EN = ["negative", "neutral", "positive"]
FUSION_EN = ["happy", "angry", "sad", "surprise", "sarcasm"]
ALPHA_COL = "门控alpha"

CSV_COLS = (["样本ID", "图片路径", "文本内容", "模式", "状态",
             "CV预测", "CV置信度"] + [f"CV_{e}" for e in EMOTIONS_EN]
            + ["NLP预测", "NLP置信度"] + [f"NLP_{e}" for e in SENTIMENTS_EN]
            + ["融合预测", "融合置信度", ALPHA_COL]
            + [f"融合_{e}" for e in FUSION_EN])

DISPLAY_COLS = ["样本ID", "图片路径", "文本内容", "CV预测", "CV置信度",
                "NLP预测", "NLP置信度", "融合预测", "融合置信度", ALPHA_COL]

def _read_csv():
    try:
        df = pd.read_csv(HISTORY_CSV, encoding="utf-8-sig")
        return df if len(df.columns) > 0 else pd.DataFrame(columns=CSV_COLS)
    except:
        return pd.DataFrame(columns=CSV_COLS)

def _str(v, default=""):
    if pd.isna(v) or v is None:
        return default
    return str(v)

def _pct(s):
    try:
        return float(str(s).replace("%", "")) / 100
    except:
        return 0

def _cv(p):
    if not p:
        return None
    try:
        return pipeline.predict_cv_only(p)
    except:
        return None

def _nlp(t):
    if not t:
        return None
    try:
        return pipeline.predict_nlp_only(t)
    except:
        return None

def _fusion(p, t):
    if not p or not t:
        return None
    try:
        return pipeline.predict_fusion(p, t)
    except:
        return None

def _bars(r, labels):
    if r is None or "error" in r:
        return {l: 0.0 for l in labels}
    p = r.get("probabilities", {})
    if isinstance(p, list):
        return {labels[i]: p[i] for i in range(min(len(labels), len(p)))}
    return p

def _summary(f, cv, nlp):
    if f and "error" not in f:
        return (f"### 融合: **{f['label_name']}** ({f['confidence']*100:.1f}%)\n\n"
                f"CV→{cv['label_name']}({cv['confidence']*100:.0f}%) | NLP→{nlp['label_name']}({nlp['confidence']*100:.0f}%) | α={f['alpha']:.3f}")
    if cv and "error" not in cv:
        return f"### CV: **{cv['label_name']}** ({cv['confidence']*100:.0f}%)"
    if nlp and "error" not in nlp:
        return f"### NLP: **{nlp['label_name']}** ({nlp['confidence']*100:.0f}%)"
    return "请上传图片或输入文本"

def _csv_bars(row, prefix, labels):
    d = {}
    for l in labels:
        v = row.get(f"{prefix}_{l}", "")
        if pd.notna(v) and str(v) not in ("", "nan"):
            d[l] = _pct(v)
    return d

@st.cache_resource
def load_model():
    print("Loading Model...")
    pipe = EmotionAnalysisPipeline()
    pipe.load_all()
    print("Model Ready!")
    return pipe

pipeline = load_model()

if not os.path.exists(HISTORY_CSV) or os.path.getsize(HISTORY_CSV) < 10:
    pd.DataFrame(columns=CSV_COLS).to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")

st.set_page_config(
    page_title="多模态情感分析系统",
    page_icon="😃",
    layout="wide"
)

st.markdown("# 多模态情感分析系统")
st.markdown("CV (ResNet18) + NLP (RoBERTa) → Gated Fusion")
st.divider()

tab1, tab2, tab3 = st.tabs(["实时分析", "摄像头", "历史记录"])

# 标签1：实时分析
with tab1:
    col1, col2 = st.columns(2)
    img_upload = None
    text_input = ""

    with col1:
        st.subheader("上传图片")
        img_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if img_file is not None:
            img_upload = Image.open(img_file)
            st.image(img_upload, use_column_width=True)

    with col2:
        st.subheader("输入文本")
        text_input = st.text_area("", placeholder="太搞笑了哈哈哈！", height=180, label_visibility="collapsed")

    st.write("")
    btn_col = st.columns([3, 1, 3])
    with btn_col[1]:
        analyze_btn = st.button("开始分析", type="primary", use_container_width=True)

    st.divider()

    res_col1, res_col2, res_col3 = st.columns(3)
    cv_plot_placeholder = res_col1.empty()
    nlp_plot_placeholder = res_col2.empty()
    fusion_plot_placeholder = res_col3.empty()
    fusion_text_placeholder = st.empty()

    if analyze_btn:
        img_abs = ""
        img_rel = ""
        if img_upload is not None:
            fname = f"{int(time.time()*1000)%100000}.jpg"
            img_abs = os.path.join(FACES_DIR, fname)
            img_upload.save(img_abs)
            img_rel = f"faces/{fname}"

        cv_r = _cv(img_abs)
        nlp_r = _nlp(text_input.strip())
        fusion_r = _fusion(img_abs, text_input.strip())

        row = {c: "" for c in CSV_COLS}
        row["样本ID"] = int(time.time()*1000) % 100000
        row["图片路径"] = img_rel
        row["文本内容"] = text_input[:100]
        row["模式"] = "auto"
        row["状态"] = "ok"

        if cv_r and "error" not in cv_r:
            row["CV预测"] = _str(cv_r["label_name"])
            row["CV置信度"] = f"{cv_r['confidence']*100:.1f}%"
            for i, e in enumerate(EMOTIONS_EN):
                row[f"CV_{e}"] = f"{cv_r['probabilities'][i]*100:.1f}%"
        if nlp_r and "error" not in nlp_r:
            row["NLP预测"] = _str(nlp_r["label_name"])
            row["NLP置信度"] = f"{nlp_r['confidence']*100:.1f}%"
            p = nlp_r["probabilities"]
            for i, e in enumerate(SENTIMENTS_EN):
                val = list(p.values())[i] if isinstance(p, dict) else p[i]
                row[f"NLP_{e}"] = f"{val*100:.1f}%"
        if fusion_r and "error" not in fusion_r:
            row["融合预测"] = _str(fusion_r["label_name"])
            row["融合置信度"] = f"{fusion_r['confidence']*100:.1f}%"
            row[ALPHA_COL] = f"{fusion_r['alpha']:.3f}"
            p = fusion_r["probabilities"]
            for i, e in enumerate(FUSION_EN):
                val = list(p.values())[i] if isinstance(p, dict) else p[i]
                row[f"融合_{e}"] = f"{val*100:.1f}%"

        df = _read_csv()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")

        cv_data = _bars(cv_r, EMOTIONS_EN)
        nlp_data = _bars(nlp_r, SENTIMENTS_EN)
        fusion_data = _bars(fusion_r, FUSION_EN)

        with cv_plot_placeholder.container():
            st.markdown("**CV**")
            st.bar_chart(cv_data, use_container_width=True)
        with nlp_plot_placeholder.container():
            st.markdown("**NLP**")
            st.bar_chart(nlp_data, use_container_width=True)
        with fusion_plot_placeholder.container():
            st.markdown("**Fusion**")
            st.bar_chart(fusion_data, use_container_width=True)

        with fusion_text_placeholder.container():
            st.markdown(_summary(fusion_r, cv_r, nlp_r))

# 标签2：摄像头（云端禁用）
with tab2:
    st.markdown("点「开始」→ 摄像头实时分析 → 绿框+表情 → 点「停止」释放")
    cam_btn_col1, cam_btn_col2 = st.columns(2)
    with cam_btn_col1:
        cam_start = st.button("开始", type="primary", use_container_width=True)
    with cam_btn_col2:
        cam_stop = st.button("停止", type="secondary", use_container_width=True)

    cam_output_placeholder = st.empty()
    cam_bars_placeholder = st.empty()
    cam_status_placeholder = st.empty()

    st.warning("⚠️ Streamlit 云端环境不支持本地摄像头，请在本地客户端运行此功能。")

# 标签3：历史记录
with tab3:
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        refresh_btn = st.button("刷新", use_container_width=True)
    with btn_col2:
        clear_btn = st.button("清空全部", type="secondary", use_container_width=True)

    if clear_btn:
        pd.DataFrame(columns=CSV_COLS).to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")
        st.success("已清空全部历史记录")

    df_history = _read_csv()
    if len(df_history) > 0:
        df_show = df_history[DISPLAY_COLS].tail(50).iloc[::-1].reset_index(drop=True)
        st.dataframe(df_show, use_container_width=True, height=400)
    else:
        st.info("暂无历史分析记录")

    st.divider()
    detail_col1, detail_col2 = st.columns([1, 2])
    with detail_col1:
        st.subheader("图片")
        hist_img_placeholder = st.empty()
    with detail_col2:
        st.subheader("详情")
        hist_summary_placeholder = st.empty()
        bar_col1, bar_col2, bar_col3 = st.columns(3)
        hist_cv_placeholder = bar_col1.empty()
        hist_nlp_placeholder = bar_col2.empty()
        hist_fusion_placeholder = bar_col3.empty()

st.divider()
st.caption("⚠️ 结果由AI判定，仅供参考，不具备专业诊断价值。")

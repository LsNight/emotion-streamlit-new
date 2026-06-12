"""
多模态情感分析系统 - Streamlit 网页版（复刻 Gradio 布局）
1:1 还原原界面结构、交互逻辑与视觉体验
"""
import os
import sys
import time
import cv2
import pandas as pd
import numpy as np
from PIL import Image

# 编码兼容
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 项目路径配置（沿用原有逻辑）
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

# 导入原有模型与工具
from main.pipeline import EmotionAnalysisPipeline

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_FD_DIR = os.path.join(_APP_DIR, "face_detect")
if _FD_DIR not in sys.path:
    sys.path.insert(0, _FD_DIR)

from face_detect import find_face, get_face, draw_box, init_detector
from cv.vision_toolkit import predict_emotion_from_array

# Streamlit 导入
import streamlit as st

# ===================== 全局常量 & 路径（完全沿用原代码） =====================
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

# ===================== 工具函数（原样复用） =====================
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

def _camera_backends():
    if sys.platform.startswith("win"):
        return [getattr(cv2, "CAP_DSHOW", None), getattr(cv2, "CAP_MSMF", None), getattr(cv2, "CAP_ANY", None)]
    if sys.platform == "darwin":
        return [getattr(cv2, "CAP_AVFOUNDATION", None), getattr(cv2, "CAP_ANY", None)]
    return [getattr(cv2, "CAP_V4L2", None), getattr(cv2, "CAP_ANY", None)]

def _open_camera(index=0):
    for backend in _camera_backends():
        try:
            cam = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
            if cam.isOpened():
                return cam
            cam.release()
        except Exception:
            continue
    return None

# ===================== 加载模型（Streamlit 缓存加速） =====================
@st.cache_resource
def load_model():
    print("Loading Model...")
    pipe = EmotionAnalysisPipeline()
    pipe.load_all()
    print("Model Ready!")
    return pipe

pipeline = load_model()

# 初始化历史CSV
if not os.path.exists(HISTORY_CSV) or os.path.getsize(HISTORY_CSV) < 10:
    pd.DataFrame(columns=CSV_COLS).to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")

# ===================== Streamlit 页面主体（复刻 Gradio 布局） =====================
st.set_page_config(
    page_title="多模态情感分析系统",
    page_icon="😃",
    layout="wide"
)

# 复刻原标题样式
st.markdown("# 多模态情感分析系统")
st.markdown("CV (ResNet18) + NLP (RoBERTa) → Gated Fusion")
st.divider()

# 复刻原标签页结构
tab1, tab2, tab3 = st.tabs(["实时分析", "摄像头", "历史记录"])

# ========== 标签1：实时分析（复刻原布局） ==========
with tab1:
    # 复刻原左右分栏布局
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

    # 复刻原居中按钮
    st.write("")
    btn_col = st.columns([3, 1, 3])
    with btn_col[1]:
        analyze_btn = st.button("开始分析", type="primary", use_container_width=True)

    st.divider()

    # 复刻原结果三列布局
    res_col1, res_col2, res_col3 = st.columns(3)
    cv_plot_placeholder = res_col1.empty()
    nlp_plot_placeholder = res_col2.empty()
    fusion_plot_placeholder = res_col3.empty()
    fusion_text_placeholder = st.empty()

    if analyze_btn:
        img_abs = ""
        img_rel = ""
        # 保存上传图片到本地 faces 文件夹
        if img_upload is not None:
            fname = f"{int(time.time()*1000)%100000}.jpg"
            img_abs = os.path.join(FACES_DIR, fname)
            img_upload.save(img_abs)
            img_rel = f"faces/{fname}"

        # 执行分析
        cv_r = _cv(img_abs)
        nlp_r = _nlp(text_input.strip())
        fusion_r = _fusion(img_abs, text_input.strip())

        # 写入历史CSV
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

        # 复刻原三列结果展示
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

# ========== 标签2：摄像头（复刻原控制逻辑） ==========
with tab2:
    st.markdown("点「开始」→ 摄像头实时分析 → 绿框+表情 → 点「停止」释放")
    # 复刻原按钮布局
    cam_btn_col1, cam_btn_col2 = st.columns(2)
    with cam_btn_col1:
        cam_start = st.button("开始", type="primary", use_container_width=True)
    with cam_btn_col2:
        cam_stop = st.button("停止", type="secondary", use_container_width=True)

    # 复刻原画面和状态布局
    cam_output_placeholder = st.empty()
    cam_bars_placeholder = st.empty()
    cam_status_placeholder = st.empty()

    # 全局摄像头状态
    if "cam_active" not in st.session_state:
        st.session_state.cam_active = False
    if "cam" not in st.session_state:
        st.session_state.cam = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_infer" not in st.session_state:
        st.session_state.last_infer = 0

    if cam_start:
        st.session_state.cam_active = True
        if st.session_state.cam is not None:
            st.session_state.cam.release()
        st.session_state.cam = _open_camera(0)
        st.session_state.last_result = None
        st.session_state.last_infer = 0

    if cam_stop:
        st.session_state.cam_active = False
        if st.session_state.cam is not None:
            st.session_state.cam.release()
            st.session_state.cam = None
        cam_output_placeholder.empty()
        cam_bars_placeholder.empty()
        cam_status_placeholder.markdown("已停止")

    if st.session_state.cam_active and st.session_state.cam is not None and st.session_state.cam.isOpened():
        ret, frame = st.session_state.cam.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.time()
            if now - st.session_state.last_infer > 2.0:
                st.session_state.last_infer = now
                init_detector(w, h)
                face = find_face(frame)
                if face is not None:
                    face_crop = get_face(frame)
                    r = predict_emotion_from_array(face_crop, pipeline.cv_classifier)
                    label = f"{r['label_name']} {r['confidence']*100:.0f}%"
                    st.session_state.last_result = (face, label, r)
                else:
                    st.session_state.last_result = None

            bars = {e:0.0 for e in EMOTIONS_EN}
            status = "未检测到人脸"
            if st.session_state.last_result is not None:
                face, label, r = st.session_state.last_result
                frame = draw_box(frame, face, label, (0, 255, 0))
                status = f"**{label}**"
                bars = {EMOTIONS_EN[i]: r['probabilities'][i] for i in range(7)}

            # 复刻原画面尺寸
            display = cv2.resize(frame, (640, 360))
            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            cam_output_placeholder.image(display_rgb, caption="实时画面", use_column_width=True)
            cam_status_placeholder.markdown(status)
            with cam_bars_placeholder.container():
                st.markdown("**全部表情概率**")
                st.bar_chart(bars, use_container_width=True)

# ========== 标签3：历史记录（复刻原表格与详情） ==========
with tab3:
    # 复刻原按钮布局
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

    # 复刻原详情展示区
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

# 底部免责声明
st.divider()
st.caption("⚠️ 结果由AI判定，仅供参考，不具备专业诊断价值。")
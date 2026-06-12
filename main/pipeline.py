"""
全局调度器 —— 整合 CV + NLP + Fusion 三模块，对外暴露统一接口。

用法:
  # 单条推理
  python -m main.pipeline --mode predict --image faces/happy_01.png --text "太搞笑了哈哈哈"

  # 交互式测试
  python -m main.pipeline --mode demo
"""
import os
import sys

# Windows 控制台默认 GBK，emoji 会报错
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

# 确保项目根目录在 sys.path 中，以支持顶层模块导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (
    CV_WEIGHT_PATH,
    CV_FEATURE_DIM,
    NLP_FEATURE_DIM,
    NLP_MODEL_PATH,
    CV_NUM_CLASSES,
    FUSION_NUM_CLASSES,
    FUSION_CLASS_NAMES,
    get_device,
)
from cv.vision_toolkit import (
    load_emotion_model,
    load_feature_extractor,
    extract_face_features,
    predict_emotion,
    EMOTION_NAMES,
)
from fusion.gated_fusion import GatedMultimodalFusion
from nlp.nlp_toolkit import (
    load_sentiment_model,
    load_tokenizer,
    extract_text_features,
    NLPFeatureExtractor,
    SENTIMENT_LABELS,
)


class EmotionAnalysisPipeline:
    """
    情感分析总调度器
    ┌──────────────────────────────────────────┐
    │  CV (ResNet18)  │  NLP (占位→BERT)       │
    │  512d 特征       │  768d 特征              │
    │       └─────────┬─────────┘               │
    │           Gated Fusion                   │
    │         门控注意力融合                     │
    │        分类器 → 喜/怒/哀                   │
    └──────────────────────────────────────────┘
    """

    def __init__(self, device=None):
        self.device = device or get_device()
        print(f"🔧 设备: {self.device}")

        # ---- 各模块（延迟加载） ----
        self.cv_classifier = None     # 完整 CV 分类器（7 分类预测用）
        self.cv_extractor = None      # CV 特征提取器（512d）
        self.nlp_model = None         # NLP 情感分类模型
        self.nlp_tokenizer = None     # NLP 分词器
        self.nlp_extractor = None     # NLP 特征提取器
        self.fusion_model = None      # 多模态融合模型

    # ===================== 加载 =====================

    def load_cv(self, weight_path=None):
        """加载 CV 模块：分类器 + 特征提取器"""
        weight_path = weight_path or CV_WEIGHT_PATH
        print(f"📷 加载 CV 模型: {os.path.basename(weight_path)}")
        self.cv_classifier = load_emotion_model(weight_path, self.device, CV_NUM_CLASSES)
        self.cv_extractor = load_feature_extractor(weight_path, self.device, CV_NUM_CLASSES)
        print(f"   CV 特征维度: {CV_FEATURE_DIM}")
        return self

    def load_nlp(self, model_path=None):
        """加载 NLP 模块：RoBERTa 情感分类模型 + 分词器 + 特征提取封箱"""
        model_path = model_path or NLP_MODEL_PATH
        print(f"📝 加载 NLP 模型: {os.path.basename(model_path)}")
        self.nlp_model = load_sentiment_model(model_path, self.device)
        self.nlp_tokenizer = load_tokenizer(model_path)
        self.nlp_extractor = NLPFeatureExtractor(self.nlp_model, self.nlp_tokenizer, self.device)
        print(f"   NLP 特征维度: {NLP_FEATURE_DIM}")
        print(f"   情感标签: {SENTIMENT_LABELS}")
        return self

    def load_fusion(self, checkpoint_path=None):
        """加载/初始化融合模型"""
        self.fusion_model = GatedMultimodalFusion(
            cv_dim=CV_FEATURE_DIM,
            nlp_dim=NLP_FEATURE_DIM,
            num_classes=FUSION_NUM_CLASSES,
        ).to(self.device)

        # 默认找 fusion_checkpoint.pth
        if checkpoint_path is None:
            default_ckpt = os.path.join(os.path.dirname(__file__), "..", "fusion", "fusion_checkpoint.pth")
            if os.path.exists(default_ckpt):
                checkpoint_path = default_ckpt

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"🔗 从 checkpoint 加载融合模型: {os.path.basename(checkpoint_path)}")
            self.fusion_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:
            print(f"🔗 初始化融合模型: GatedMultimodalFusion (随机权重)")
            print(f"   输入: [{CV_FEATURE_DIM}d CV] + [{NLP_FEATURE_DIM}d NLP]")
            print(f"   输出: {FUSION_NUM_CLASSES} 类 {FUSION_CLASS_NAMES}")
        return self

    def load_all(self):
        """一键加载全部模块"""
        self.load_cv()
        self.load_nlp()
        self.load_fusion()
        print("✅ 全部模块加载完成")
        return self

    # ===================== 推理 =====================

    def predict_cv_only(self, image_path):
        """纯 CV 表情预测（7 分类）"""
        if self.cv_classifier is None:
            self.load_cv()
        return predict_emotion(image_path, self.cv_classifier, self.device)

    def predict_fusion(self, image_path, text):
        """
        多模态融合预测（3 分类: 喜/怒/哀）
        1. CV 提取器 → 512d 特征
        2. NLP 提取器 → 768d 特征
        3. 融合模型 → 分类结果
        """
        if self.cv_extractor is None:
            self.load_cv()
        if self.nlp_model is None:
            self.load_nlp()
        if self.fusion_model is None:
            raise RuntimeError("融合模型未加载！请先调用 load_fusion() 或 load_all()")

        self.fusion_model.eval()

        # CV 特征
        if os.path.exists(image_path):
            cv_feat = extract_face_features(image_path, self.cv_extractor, self.device)
        else:
            print(f"⚠️  图片不存在: {image_path}，使用零向量")
            cv_feat = torch.zeros(1, CV_FEATURE_DIM)

        # NLP 特征
        nlp_feat = extract_text_features(text, self.nlp_model, self.nlp_tokenizer, self.device).to(self.device)

        # 融合预测
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
            "alpha": alpha[0].item(),  # NLP 权重
        }

    def predict_nlp_only(self, text):
        """纯 NLP 文本情感预测（3 分类: 负面/中立/正面）"""
        if self.nlp_model is None:
            self.load_nlp()
        from nlp.nlp_toolkit import predict_sentiment
        return predict_sentiment(text, self.nlp_model, self.nlp_tokenizer, self.device)

    # ===================== 演示 =====================

    def demo(self):
        """交互式演示：CV 预测 + 融合预测"""
        self.load_all()

        print(f"\n{'='*50}")
        print(f"🎭 情感分析系统 Demo")
        print(f"{'='*50}")
        print(f"输入格式: <图片路径> <文本>")
        print(f"示例: faces/happy_01.png 这个主播太有才了！")
        print(f"输入 'quit' 退出\n")

        while True:
            try:
                user_input = input("👉 ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    print("👋 再见！")
                    break
                if not user_input:
                    continue

                parts = user_input.split(maxsplit=1)
                img_path = parts[0]
                text = parts[1] if len(parts) > 1 else ""

                # 1) CV 纯视觉预测
                cv_result = self.predict_cv_only(img_path)
                print(f"\n  📷 CV 纯视觉预测:")
                print(f"     表情: {cv_result['label_name']} (置信度: {cv_result['confidence']*100:.1f}%)")
                print(f"     概率分布: ", end="")
                for name, p in zip(EMOTION_NAMES, cv_result["probabilities"]):
                    bar = "█" * int(p * 20)
                    print(f"\n       {name:9s} {p*100:5.1f}% {bar}", end="")

                # 2) 多模态融合预测
                try:
                    fusion_result = self.predict_fusion(img_path, text)
                    print(f"\n\n  🔗 多模态融合预测:")
                    print(f"     情感: {fusion_result['label_name']} (置信度: {fusion_result['confidence']*100:.1f}%)")
                    print(f"     门控系数 α (NLP权重): {fusion_result['alpha']:.3f}")
                    print(f"     概率分布:")
                    for name, p in fusion_result["probabilities"].items():
                        bar = "█" * int(p * 20)
                        print(f"       {name:6s} {p*100:5.1f}% {bar}")
                except RuntimeError as e:
                    print(f"\n  ⚠️  融合预测失败: {e}")

                print()

            except FileNotFoundError as e:
                print(f"  ❌ {e}")
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break


# ========================= 入口 =========================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="情感分析系统总调度器")
    parser.add_argument("--mode", type=str, default="demo",
                        choices=["predict", "demo"],
                        help="运行模式: predict / demo")
    parser.add_argument("--image", type=str, default=None, help="图片路径（predict 模式）")
    parser.add_argument("--text", type=str, default="", help="文本内容（predict 模式）")
    parser.add_argument("--checkpoint", type=str, default=None, help="融合模型 checkpoint 路径")
    args = parser.parse_args()

    pipeline = EmotionAnalysisPipeline()

    if args.mode == "predict":
        if not args.image:
            print("❌ predict 模式需要 --image 参数")
            sys.exit(1)
        pipeline.load_all()
        if args.checkpoint:
            pipeline.load_fusion(args.checkpoint)
        result = pipeline.predict_fusion(args.image, args.text)
        print(f"\n 预测结果:")
        print(f"   情感: {result['label_name']}")
        print(f"   置信度: {result['confidence']*100:.1f}%")
        print(f"   门控 α: {result['alpha']:.3f}")
        print(f"   概率: {result['probabilities']}")

    elif args.mode == "demo":
        pipeline.demo()


if __name__ == "__main__":
    main()

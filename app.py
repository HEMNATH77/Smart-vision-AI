import os
import time
import json
from typing import Dict, Any, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers  # type: ignore
from ultralytics import YOLO

# Keras application imports
from keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess  # type: ignore
from keras.applications.efficientnet import EfficientNetB0, preprocess_input as effnet_preprocess  # type: ignore
from pathlib import Path

# ------------------------------------------------------------
# GLOBAL CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UI Enhancement: Custom CSS for Enterprise Look ---
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
        color: #31333F;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    
    /* Custom Cards */
    .css-card {
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Compact Header Styling (From original) */
    .block-container {
        padding-top: 2rem !important;
    }
    .center-text {
        text-align: center !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Progress Bar Color */
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ---- Compact Header ----
st.markdown("""
<div style='text-align: center; padding-bottom: 20px;'>
    <h1 style='color: #1E88E5; margin-bottom: 0;'>🤖 SmartVision AI</h1>
    <h4 style='color: #555; margin-top: 5px;'>Intelligent Multi-Class Object Recognition System</h4>
    <p style='color: #888; font-size: 0.9em;'>End-to-end computer vision pipeline on a COCO subset of 25 everyday object classes</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Resolve repository root relative to this file (streamlit_app/app.py)
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent  # repo/
SAVED_MODELS_DIR = REPO_ROOT / "saved_models"
YOLO_RUNS_DIR = REPO_ROOT / "yolo_runs"
SMARTVISION_METRICS_DIR = REPO_ROOT / "smartvision_metrics"
SMARTVISION_DATASET_DIR = REPO_ROOT / "smartvision_dataset"

# Then turn constants into Path objects / strings
YOLO_WEIGHTS_PATH = str(
    YOLO_RUNS_DIR / "smartvision_yolov8s" / "weights" / "best.pt"
)

CLASSIFIER_MODEL_CONFIGS = {
    "VGG16": {
        "type": "vgg16",
        "path": str(SAVED_MODELS_DIR / "vgg16_v2_stage2_best.h5"),
    },
    "ResNet50": {
        "type": "resnet50",
        "path": str(SAVED_MODELS_DIR / "resnet50_v2_stage2_best.weights.h5"),
    },
    "MobileNetV2": {
        "type": "mobilenetv2",
        "path": str(SAVED_MODELS_DIR / "mobilenetv2_v2_stage2_best.weights.h5"),
    },
    "EfficientNetB0": {
        "type": "efficientnetb0",
        "path": str(SAVED_MODELS_DIR / "efficientnetb0_stage2_best.weights.h5"),
    },
}

CLASS_METRIC_PATHS = {
    "VGG16": str(SMARTVISION_METRICS_DIR / "vgg16_v2_stage2" / "metrics.json"),
    "ResNet50": str(SMARTVISION_METRICS_DIR / "resnet50_v2_stage2" / "metrics.json"),
    "MobileNetV2": str(SMARTVISION_METRICS_DIR / "mobilenetv2_v2" / "metrics.json"),
    "EfficientNetB0": str(SMARTVISION_METRICS_DIR / "efficientnetb0" / "metrics.json"),
}

YOLO_METRICS_JSON = str(YOLO_RUNS_DIR / "smartvision_yolov8s_alltrain3" /"validation_all_20251206_210906" /"validation_metrics_all.json")
BASE_DIR = str(SMARTVISION_DATASET_DIR)
CLASS_DIR = str(SMARTVISION_DATASET_DIR / "classification")
DET_DIR = str(SMARTVISION_DATASET_DIR / "detection")

IMG_SIZE = (224, 224)
NUM_CLASSES = 25

CLASS_NAMES = [
    "airplane", "bed", "bench", "bicycle", "bird", "bottle", "bowl",
    "bus", "cake", "car", "cat", "chair", "couch", "cow", "cup", "dog",
    "elephant", "horse", "motorcycle", "person", "pizza", "potted plant",
    "stop sign", "traffic light", "truck"
]
assert len(CLASS_NAMES) == NUM_CLASSES


# ------------------------------------------------------------
# BUILDERS – MATCH TRAINING ARCHITECTURES (UNCHANGED)
# ------------------------------------------------------------

# ---------- VGG16 v2 ----------
def build_vgg16_model_v2():
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_layer")

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.2),
            layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.2)),
            layers.Lambda(lambda x: tf.image.random_saturation(x, 0.8, 1.2)),
        ],
        name="data_augmentation",
    )

    x = data_augmentation(inputs)

    x = layers.Lambda(
        lambda z: vgg16_preprocess(tf.cast(z, tf.float32)),
        name="vgg16_preprocess",
    )(x)

    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )

    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(base_model.output)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.5, name="dropout_0_5")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="VGG16_smartvision_v2")
    return model


# ---------- ResNet50 v2 ----------
def build_resnet50_model_v2():
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_layer")

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.15),
            layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.15)),
            layers.Lambda(lambda x: tf.image.random_saturation(x, 0.85, 1.15)),
        ],
        name="data_augmentation",
    )

    x = data_augmentation(inputs)

    x = layers.Lambda(
        keras.applications.resnet50.preprocess_input,
        name="resnet50_preprocess",
    )(x)

    base_model = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )

    x = base_model(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.BatchNormalization(name="head_batchnorm")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    x = layers.Dense(256, activation="relu", name="head_dense")(x)
    x = layers.BatchNormalization(name="head_batchnorm_2")(x)
    x = layers.Dropout(0.5, name="head_dropout_2")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ResNet50_smartvision_v2")
    return model


# ---------- MobileNetV2 v2 ----------
def build_mobilenetv2_model_v2():
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_layer")

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),  # ~±15°
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.15),
            layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.15)),
            layers.Lambda(lambda x: tf.image.random_saturation(x, 0.85, 1.15)),
        ],
        name="data_augmentation",
    )

    x = data_augmentation(inputs)

    x = layers.Lambda(
        keras.applications.mobilenet_v2.preprocess_input,
        name="mobilenetv2_preprocess",
    )(x)

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )

    x = base_model(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)

    x = layers.BatchNormalization(name="head_batchnorm_1")(x)
    x = layers.Dropout(0.4, name="head_dropout_1")(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="head_dense_1",
    )(x)

    x = layers.BatchNormalization(name="head_batchnorm_2")(x)
    x = layers.Dropout(0.5, name="head_dropout_2")(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="MobileNetV2_smartvision_v2",
    )
    return model


# ---------- EfficientNetB0 ----------
def bright_jitter(x):
    x_f32 = tf.cast(x, tf.float32)
    x_f32 = tf.image.random_brightness(x_f32, max_delta=0.25)
    return tf.cast(x_f32, x.dtype)

def sat_jitter(x):
    x_f32 = tf.cast(x, tf.float32)
    x_f32 = tf.image.random_saturation(x_f32, lower=0.7, upper=1.3)
    return tf.cast(x_f32, x.dtype)

def build_efficientnetb0_model():
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_layer")

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.3),
            layers.RandomTranslation(0.1, 0.1),
            layers.Lambda(bright_jitter),
            layers.Lambda(sat_jitter),
        ],
        name="advanced_data_augmentation",
    )

    x = data_augmentation(inputs)

    x = layers.Lambda(
        lambda z: effnet_preprocess(tf.cast(z, tf.float32)),
        name="effnet_preprocess",
    )(x)

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet"
    )
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn_1")(x)
    x = layers.Dense(256, activation="relu", name="head_dense_1")(x)
    x = layers.BatchNormalization(name="head_bn_2")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = keras.Model(inputs, outputs, name="EfficientNetB0_smartvision")
    return model


# ------------------------------------------------------------
# CACHED MODEL LOADERS (UNCHANGED)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_yolo_model() -> YOLO:
    if os.path.exists(YOLO_WEIGHTS_PATH):
        model = YOLO(YOLO_WEIGHTS_PATH)
    else:
        st.warning(f"⚠️ Custom YOLO weights not found at {YOLO_WEIGHTS_PATH}. Using default YOLOv8s model.")
        model = YOLO("yolov8s.pt")  # Download and use default YOLOv8s
    return model


@st.cache_resource(show_spinner=True)
def load_classification_models() -> Dict[str, keras.Model]:
    models: Dict[str, keras.Model] = {}

    for name, cfg in CLASSIFIER_MODEL_CONFIGS.items():
        model_type = cfg["type"]
        path = cfg["path"]

        # 1) Build the architecture
        if model_type == "vgg16":
            model = build_vgg16_model_v2()
        elif model_type == "resnet50":
            model = build_resnet50_model_v2()
        elif model_type == "mobilenetv2":
            model = build_mobilenetv2_model_v2()
        elif model_type == "efficientnetb0":
            model = build_efficientnetb0_model()
        else:
            continue

        # 2) Try to load your training weights (if path is provided and file exists)
        if path is not None and os.path.exists(path):
            try:
                model.load_weights(path)
            except Exception as e:
                st.sidebar.warning(
                    f"⚠️ Could not fully load weights for {name} from {path}: {e}\n"
                    "   Using ImageNet-pretrained backbone + random head."
                )
        elif path is not None:
            st.sidebar.warning(
                f"⚠️ Weights file for {name} not found at {path}. "
                "Using ImageNet-pretrained backbone + random head."
            )
        # if path is None → silently use ImageNet + random head

        models[name] = model

    return models


# ------------------------------------------------------------
# IMAGE HELPERS
# ------------------------------------------------------------
def read_image_file(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file).convert("RGB")
    return image


def preprocess_for_classifier(pil_img: Image.Image) -> np.ndarray:
    img_resized = pil_img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


# ------------------------------------------------------------
# DRAW BOXES FOR DETECTION
# ------------------------------------------------------------
def draw_boxes_with_labels(
    pil_img: Image.Image,
    detections: List[Dict[str, Any]],
    font_path: str = None
) -> Image.Image:
    draw = ImageDraw.Draw(pil_img)

    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, 16)
    else:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        yolo_label = det["label"]
        conf_yolo = det["conf_yolo"]
        cls_label = det.get("cls_label")
        cls_conf = det.get("cls_conf")

        if cls_label is not None:
            text = f"{yolo_label} {conf_yolo:.2f} | CLS: {cls_label} {cls_conf:.2f}"
        else:
            text = f"{yolo_label} {conf_yolo:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        text_bg = [x1,
                   max(0, y1 - text_h - 2),
                   x1 + text_w + 4,
                   y1]
        draw.rectangle(text_bg, fill="black")
        draw.text((x1 + 2, max(0, y1 - text_h - 1)), text, fill="white", font=font)

    return pil_img


def run_yolo_with_optional_classifier(
    pil_img: Image.Image,
    yolo_model: YOLO,
    classifier_model: keras.Model = None,
    conf_threshold: float = 0.5
) -> Dict[str, Any]:
    """Run YOLO on a PIL image, optionally verify each box with classifier."""
    orig_w, orig_h = pil_img.size

    t0 = time.perf_counter()
    results = yolo_model.predict(
        pil_img,
        imgsz=640,
        conf=conf_threshold,
        device="cpu",  # change to "0" if GPU available
        verbose=False,
    )
    t1 = time.perf_counter()
    infer_time = t1 - t0

    res = results[0]
    boxes = res.boxes

    detections = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        conf_yolo = float(box.conf[0].item())
        label = res.names[cls_id]

        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))

        cls_label = None
        cls_conf = None
        if classifier_model is not None:
            crop = pil_img.crop((x1, y1, x2, y2))
            arr = preprocess_for_classifier(crop)
            probs = classifier_model.predict(arr, verbose=0)[0]
            idx = int(np.argmax(probs))
            cls_label = CLASS_NAMES[idx]
            cls_conf = float(probs[idx])

        detections.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "label": label,
                "conf_yolo": conf_yolo,
                "cls_label": cls_label,
                "cls_conf": cls_conf,
            }
        )

    annotated = pil_img.copy()
    annotated = draw_boxes_with_labels(annotated, detections)

    return {
        "annotated_image": annotated,
        "detections": detections,
        "yolo_inference_time_sec": infer_time,
    }


# ------------------------------------------------------------
# METRICS LOADING
# ------------------------------------------------------------
@st.cache_data
def load_classification_metrics() -> pd.DataFrame:
    rows = []
    for name, path in CLASS_METRIC_PATHS.items():
        if os.path.exists(path):
            with open(path, "r") as f:
                m = json.load(f)
            rows.append(
                {
                    "Model": name,
                    "Accuracy": m.get("accuracy", None),
                    "F1 (weighted)": m.get("f1_weighted", None),
                    "Top-5 Accuracy": m.get("top5_accuracy", None),
                    "Images/sec": m.get("images_per_second", None),
                    "Size (MB)": m.get("model_size_mb", None),
                }
            )
    df = pd.DataFrame(rows)
    return df


@st.cache_data
def load_yolo_metrics() -> Dict[str, Any]:
    if not os.path.exists(YOLO_METRICS_JSON):
        return {}
    with open(YOLO_METRICS_JSON, "r") as f:
        return json.load(f)


# ------------------------------------------------------------
# SIDEBAR NAVIGATION (UI ENHANCED)
# ------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
st.sidebar.title("Navigation")

PAGES = {
    "Home": "🏠 Home",
    "Classification": "🖼️ Image Classification",
    "Detection": "📦 Object Detection",
    "Performance": "📊 Model Performance",
    "Webcam": "📷 Webcam Detection",
    "About": "ℹ️ About",
}

page_selection = st.sidebar.radio("Go to", list(PAGES.values()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Classes", NUM_CLASSES)
col_s2.metric("Models", 5)

st.sidebar.info("Running on Local CPU/GPU\n\nv1.0.0")

# Reverse lookup for logic
page = page_selection

# ------------------------------------------------------------
# PAGE 1 – HOME
# ------------------------------------------------------------
if page == PAGES["Home"]:
    
    # --- Feature Grid Layout ---
    st.subheader("📌 Project Overview")
    
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        st.markdown("""
        <div class="css-card">
            <h3>🖼️ Classification</h3>
            <p>Multi-backbone CNNs (VGG16, ResNet50, etc.) for single object recognition.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_feat2:
        st.markdown("""
        <div class="css-card">
            <h3>📦 Detection</h3>
            <p>YOLOv8s + ResNet50 Pipeline for robust multi-object detection and verification.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_feat3:
        st.markdown("""
        <div class="css-card">
            <h3>📊 Analytics</h3>
            <p>Interactive dashboards for performance metrics, confusion matrices, and speed.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🏷️ Supported Classes (COCO Subset)")
    
    # Badge styling
    st.markdown("""
    <style>
    .badge {
        display: inline-block;
        padding: 6px 14px;
        margin: 4px;
        background-color: #E3F2FD;
        color: #1565C0;
        border: 1px solid #BBDEFB;
        border-radius: 16px;
        font-weight: 500;
        font-size: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

    classes = [c.title() for c in CLASS_NAMES]
    html = "".join([f"<span class='badge'>{c}</span>" for c in classes])
    st.markdown(html, unsafe_allow_html=True)

    st.divider()

    st.subheader("🕹️ Workflow Guide")
    with st.expander("Expand for User Guide", expanded=True):
        st.markdown("""
        1. **Start with Object Detection**: This is the most visual feature. Upload a scene (e.g., a street or a desk).
        2. **Verify with Classification**: If you have a close-up of a specific item, use the Classification tab to see how different architectures perceive it.
        3. **Analyze Performance**: Check the Metrics tab to understand trade-offs between speed (MobileNet) and accuracy (EfficientNet).
        """)

    # --- Sample Gallery ---
    st.subheader("🧪 Sample Outputs")
    sample_dir = "inference_outputs"
    if os.path.exists(sample_dir):
        imgs = [
            os.path.join(sample_dir, f)
            for f in os.listdir(sample_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        if imgs:
            # Use a scrollable container or simple columns
            cols = st.columns(min(4, len(imgs)))
            for i, img_path in enumerate(imgs[:4]):
                with cols[i]:
                    st.image(img_path, width='stretch')
                    st.caption(os.path.basename(img_path))
        else:
            st.info("No sample images found in `inference_outputs/` yet.")
    else:
        st.info("💡 Run inference to generate and save sample outputs here.")

# ------------------------------------------------------------
# PAGE 2 – IMAGE CLASSIFICATION
# ------------------------------------------------------------
elif page == PAGES["Classification"]:
    st.markdown("## 🖼️ Image Classification")
    st.markdown("Compare predictions across **4 different CNN architectures** for a single object.")

    # Two-column layout: Upload/Image on Left, Results on Right
    col_input, col_results = st.columns([1, 1.5], gap="large")

    with col_input:
        st.markdown("### 1. Upload Image")
        uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            pil_img = read_image_file(uploaded_file)
            st.image(pil_img, caption="Source Image", width='stretch')
            
            # Pre-calc array
            arr = preprocess_for_classifier(pil_img)

    with col_results:
        st.markdown("### 2. Analysis Results")
        
        if uploaded_file:
            with st.spinner("Running 4x Neural Networks..."):
                cls_models = load_classification_models()

            if not cls_models:
                st.error("❌ No classification models found.")
            else:
                # Use Tabs for cleaner view of each model
                tab_names = list(cls_models.keys())
                tabs = st.tabs(tab_names)

                for (model_name, model), tab in zip(cls_models.items(), tabs):
                    with tab:
                        # Logic
                        probs = model.predict(arr, verbose=0)[0]
                        top5_idx = probs.argsort()[-5:][::-1]
                        top5_labels = [CLASS_NAMES[i] for i in top5_idx]
                        top5_probs = [probs[i] for i in top5_idx]

                        # UI: Top Prediction Card
                        st.success(f"**Top Prediction:** {top5_labels[0].upper()}")
                        
                        # UI: Detailed Progress Bars
                        st.markdown("#### Confidence Scores")
                        for lbl, p in zip(top5_labels, top5_probs):
                            col_lbl, col_bar, col_val = st.columns([3, 5, 2])
                            with col_lbl:
                                st.write(f"**{lbl}**")
                            with col_bar:
                                st.progress(float(p))
                            with col_val:
                                st.caption(f"{p:.1%}")
        else:
            st.info("👈 Waiting for image upload...")

# ------------------------------------------------------------
# PAGE 3 – OBJECT DETECTION
# ------------------------------------------------------------
elif page == PAGES["Detection"]:
    st.markdown("## 📦 Object Detection")
    st.markdown("Powered by **YOLOv8** (Detection) + **ResNet50** (Verification).")

    # --- UI Enhancement: Configuration Expander ---
    with st.expander("🛠️ Model Configuration", expanded=False):
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            conf_th = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05, help="Minimum confidence to accept a detection.")
        with col_cfg2:
            use_classifier = st.toggle("Enable ResNet50 Verification", value=True, help="Cross-verify detected objects with a second classifier.")

    # --- Main Input Area ---
    st.markdown("### Upload Scene")
    
    # Form to prevent auto-reloading during upload
    with st.form("detection_form", clear_on_submit=False):
        uploaded_file = st.file_uploader("Select image...", type=["jpg", "jpeg", "png"])
        
        # Center the submit button
        col_sub1, col_sub2, col_sub3 = st.columns([1,2,1])
        with col_sub2:
            submitted = st.form_submit_button("🚀 Run Detection Pipeline", use_container_width=True)

    if submitted and uploaded_file is not None:
        pil_img = read_image_file(uploaded_file)
        
        # Create placeholders for visual feedback
        status_container = st.container()
        
        with st.spinner("🧠 Analyzing scene..."):
            # Load models
            yolo_model = load_yolo_model()
            
            classifier_model = None
            if use_classifier:
                try:
                    cls_models = load_classification_models()
                    classifier_model = cls_models.get("ResNet50")
                    if classifier_model is None:
                        # Fallback logic preserved
                        classifier_model = build_resnet50_model_v2()
                        path = CLASSIFIER_MODEL_CONFIGS["ResNet50"]["path"]
                        if os.path.exists(path):
                            classifier_model.load_weights(path)
                except Exception:
                    pass

            # Run inference
            result = run_yolo_with_optional_classifier(
                pil_img=pil_img,
                yolo_model=yolo_model,
                classifier_model=classifier_model,
                conf_threshold=conf_th,
            )

        # --- UI Enhancement: Results Dashboard ---
        
        # 1. Key Metrics Row
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        col_kpi1.metric("Objects Detected", len(result['detections']))
        col_kpi2.metric("Inference Time", f"{result['yolo_inference_time_sec']*1000:.1f} ms")
        col_kpi3.metric("Verification", "Active" if use_classifier else "Off")
        
        st.divider()

        # 2. Side-by-Side Images
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(pil_img, caption="Original Input", width='stretch')
        with col_img2:
            st.image(result["annotated_image"], caption="AI Analysis Result", width='stretch')

        # 3. Data Table
        st.subheader("📋 Detection Details")
        if result["detections"]:
            df_det = pd.DataFrame([
                {
                    "Object": det["label"],
                    "YOLO Conf": f"{det['conf_yolo']:.2%}",
                    "Verifier": det.get("cls_label", "N/A"),
                    "Verifier Conf": f"{det.get('cls_conf', 0):.2%}" if det.get('cls_conf') else "N/A",
                }
                for det in result["detections"]
            ])
            st.dataframe(df_det, width='stretch', hide_index=True)
        else:
            st.warning("No objects found matching the criteria.")

    elif submitted and uploaded_file is None:
        st.warning("⚠️ Please upload an image first.")

# ------------------------------------------------------------
# PAGE 4 – MODEL PERFORMANCE
# ------------------------------------------------------------
elif page == PAGES["Performance"]:
    st.markdown("## 📊 Model Performance Analytics")
    
    # --- UI Enhancement: Tabs for cleaner data viewing ---
    tab_cls, tab_yolo, tab_plots = st.tabs(["🧠 Classification Metrics", "📦 Detection Metrics", "📈 Confusion Matrices"])

    with tab_cls:
        st.markdown("### CNN Architectures Benchmark")
        df_cls = load_classification_metrics()
        
        if not df_cls.empty:
            # Highlight best models
            st.dataframe(
                df_cls.style.highlight_max(axis=0, subset=["Accuracy", "F1 (weighted)", "Images/sec"], color='#d4edda'),
                use_container_width=True
            )

            st.markdown("#### Visual Comparison")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.write("**Accuracy vs Model**")
                st.bar_chart(df_cls.set_index("Model")["Accuracy"], color="#4CAF50")
            
            with col_chart2:
                st.write("**Inference Speed (Img/Sec)**")
                st.bar_chart(df_cls.set_index("Model")["Images/sec"], color="#2196F3")
        else:
            st.info("Metrics file not found.")

    with tab_yolo:
        st.markdown("### YOLOv8 Validation Results")
        yolo_m = load_yolo_metrics()
        
        if yolo_m:
            overall = yolo_m.get('overall_metrics', {})
            speed = yolo_m.get('speed_metrics', {})
            
            # Card layout for metrics
            col_ym1, col_ym2, col_ym3, col_ym4 = st.columns(4)
            col_ym1.metric("mAP@0.5", f"{overall.get('mAP50', 0):.3f}", delta="Accuracy")
            col_ym2.metric("mAP@0.5:0.95", f"{overall.get('mAP50_95', 0):.3f}")
            col_ym3.metric("Recall", f"{overall.get('recall', 0):.3f}")
            col_ym4.metric("FPS", f"{speed.get('fps', 0):.1f}", delta="Speed")
        else:
            st.info("YOLO metrics not found.")

    with tab_plots:
        st.markdown("### Performance Plots")
        comp_dir = os.path.join("smartvision_metrics", "comparison_plots")
        if os.path.exists(comp_dir):
            imgs = sorted([f for f in os.listdir(comp_dir) if f.endswith(".png")])
            if imgs:
                # Grid layout for plots
                cols = st.columns(2)
                for i, img_file in enumerate(imgs):
                    with cols[i % 2]:
                        st.image(os.path.join(comp_dir, img_file), caption=img_file, width='stretch')
            else:
                st.info("No plots generated yet.")
        else:
            st.warning("Comparison folder missing.")

# ------------------------------------------------------------
# PAGE 5 – WEBCAM DETECTION
# ------------------------------------------------------------
elif page == PAGES["Webcam"]:
    st.markdown("## 📷 Live Snapshot Detection")
    st.caption("Capture a frame from your camera to run the pipeline.")

    col_cam, col_res = st.columns([1, 1], gap="medium")

    with col_cam:
        conf_th = st.slider("Sensitivity (Confidence)", 0.1, 0.9, 0.5)
        cam_image = st.camera_input("Take Snapshot")

    with col_res:
        if cam_image is not None:
            pil_img = Image.open(cam_image).convert("RGB")

            with st.spinner("Processing Snapshot..."):
                yolo_model = load_yolo_model()
                result = run_yolo_with_optional_classifier(
                    pil_img=pil_img,
                    yolo_model=yolo_model,
                    classifier_model=None, # Speed optimization
                    conf_threshold=conf_th,
                )
            
            st.image(result["annotated_image"], caption=f"Inference: {result['yolo_inference_time_sec']*1000:.1f}ms", width='stretch')
            
            if result["detections"]:
                with st.expander("See Detection List", expanded=True):
                    for det in result["detections"]:
                        st.markdown(f"**{det['label']}**: {det['conf_yolo']:.2%}")
            else:
                st.info("No objects detected.")

# ------------------------------------------------------------
# PAGE 6 – ABOUT
# ------------------------------------------------------------
elif page == PAGES["About"]:
    col_about1, col_about2 = st.columns([2, 1])
    
    with col_about1:
        st.markdown("## ℹ️ About SmartVision AI")
        st.markdown("""
        **SmartVision AI** is a demonstration of a hybrid Computer Vision pipeline. 
        It solves the problem of detecting and verifying objects in complex scenes by combining 
        the speed of **YOLO** with the feature richness of standard **CNN classifiers**.
        """)
        
        st.subheader("Architecture Stack")
        st.markdown("""
        * **Frontend:** Streamlit
        * **Detection:** Ultralytics YOLOv8 (Small)
        * **Classification:** TensorFlow/Keras (VGG16, ResNet50, MobileNetV2, EfficientNetB0)
        * **Data Processing:** NumPy, Pandas, Pillow
        """)

    with col_about2:
        st.markdown("### Data & Model Info")
        st.info("""
        **Dataset:** MS COCO Subset (25 Classes)
        
        **Classes:** Includes everyday items like Person, Car, Laptop, Phone, Cat, Dog, etc.
        
        **Deployment:** Optimized for CPU inference, expandable to GPU/TFLite.
        """)
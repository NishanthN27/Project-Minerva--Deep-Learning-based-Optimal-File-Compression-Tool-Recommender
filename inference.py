import os
import json
import math
import subprocess
import tempfile
import time
import shutil
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
try:
    import fitz

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# --- 1. CONFIGURATION AND LOADING ARTIFACTS ---
MODEL_DIR = "saved_models"
SCALER_PATH = os.path.join(MODEL_DIR, "my_scaler.gz")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
HYBRID_EXTRACTOR_PATH = os.path.join(MODEL_DIR, "tuned_feature_extractor.h5")

MODELS = {
    "Baseline MLP": os.path.join(MODEL_DIR, "tuned_baseline_mlp_model.h5"),
    "Robust MLP": os.path.join(MODEL_DIR, "tuned_robust_mlp_model.h5"),
    "Wide & Deep": os.path.join(MODEL_DIR, "tuned_wide_deep_model.h5"),
    "ResNet MLP": os.path.join(MODEL_DIR, "tuned_resnet_model.h5"),
    "DL-ML Hybrid": os.path.join(MODEL_DIR, "tuned_xgb_classifier.pkl"),
}

COMPRESSOR_PATHS = {
    "7zip": "7z",
    "zip": "zip",
    "winrar": "rar",
    "gzip": "gzip",
    "bzip2": "bzip2",
    "flac": "flac",
}

print("Loading all models and artifacts into INFERENCE_CACHE...")
INFERENCE_CACHE = {}
try:
    INFERENCE_CACHE["SCALER"] = joblib.load(SCALER_PATH)
    INFERENCE_CACHE["LABEL_ENCODER"] = joblib.load(LABEL_ENCODER_PATH)
    INFERENCE_CACHE["HYBRID_EXTRACTOR"] = tf.keras.models.load_model(
        HYBRID_EXTRACTOR_PATH
    )
    for name, path in MODELS.items():
        if ".h5" in path:
            INFERENCE_CACHE[name] = tf.keras.models.load_model(path)
        elif ".pkl" in path:
            INFERENCE_CACHE[name] = joblib.load(path)
    print("✅ All artifacts loaded successfully.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load a required model or artifact. Error: {e}")
    exit()

# --- 2. FEATURE EXTRACTION FUNCTIONS ---
def get_universal_features(file_path):
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return 0, 0.0, {i: 0.0 for i in range(256)}
        counts = Counter()
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                counts.update(chunk)
        entropy = (
            -sum((c / file_size) * math.log2(c / file_size) for c in counts.values())
            if file_size > 0
            else 0
        )
        byte_dist = {i: counts.get(i, 0) / file_size for i in range(256)}
        return file_size, entropy, byte_dist
    except Exception:
        return 0, 0.0, {i: 0.0 for i in range(256)}


def get_image_features(file_path):
    try:
        with Image.open(file_path) as img:
            return img.size[0], img.size[1], len(img.getbands()), 8
    except Exception:
        return 0, 0, 0, 0


def get_audio_features(file_path):
    if not PYDUB_AVAILABLE:
        return 0, 0, 0, 0
    try:
        audio = AudioSegment.from_wav(file_path)
        return (
            len(audio) / 1000.0,
            audio.frame_rate,
            audio.channels,
            audio.sample_width * 8,
        )
    except Exception:
        return 0, 0, 0, 0


def get_text_features(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return sum(len(l.strip()) for l in lines) / len(lines) if lines else 0
    except Exception:
        return 0


def get_pdf_features(file_path):
    if not PYMUPDF_AVAILABLE:
        return 0
    try:
        with fitz.open(file_path) as doc:
            return doc.page_count
    except Exception:
        return 0


def compile_feature_vector(file_path):
    ext = os.path.splitext(file_path)[1].lower().strip(".")
    fs, ent, bd = get_universal_features(file_path)
    all, pc, w, h, ci, bdi, ds, sr, ca, bda = (0,) * 10
    if ext in ["txt", "csv", "json"]:
        all = get_text_features(file_path)
    elif ext == "pdf":
        pc = get_pdf_features(file_path)
    elif ext in ["jpg", "jpeg", "png"]:
        w, h, ci, bdi = get_image_features(file_path)
    elif ext == "wav":
        ds, sr, ca, bda = get_audio_features(file_path)

    stat_features = [fs, ent, all, pc, w, h, ci, bdi, ds, sr, ca, bda]
    byte_features = [bd.get(i, 0) for i in range(256)]
    raw_features = np.array(stat_features + byte_features).reshape(1, -1)

    insights = {
        "File Type": ext.upper(),
        "File Size (KB)": f"{fs/1024:.2f}",
        "Shannon Entropy": f"{ent:.4f}",
    }
    if ds > 0:
        insights["Duration (s)"] = f"{ds:.2f}"
    if w > 0:
        insights["Dimensions"] = f"{w} x {h}"
    if pc > 0:
        insights["Page Count"] = pc
    return raw_features, insights


# --- 3. BENCHMARKING AND PREDICTION FUNCTIONS ---
def run_single_compression(tool, file_path):
    """Runs one compression tool, returns its ratio, time, and compressed file path."""
    start_time = time.time()
    original_size = os.path.getsize(file_path)
    if original_size == 0:
        return 1.0, 0.0, None

    tool_cmd_path = COMPRESSOR_PATHS.get(tool)
    executable = shutil.which(tool_cmd_path)
    if not executable:
        return 1.0, 0.0, None

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            base_name = os.path.basename(file_path)
            temp_input_path = os.path.join(temp_dir, base_name)
            shutil.copyfile(file_path, temp_input_path)

            output_path = ""
            cmd = []

            if tool == "7zip":
                output_path = os.path.join(temp_dir, "output.7z")
                cmd = [executable, "a", "-y", output_path, base_name]
            elif tool == "winrar":
                output_path = os.path.join(temp_dir, "output.rar")
                cmd = [executable, "a", "-m5", "-o+", output_path, base_name]
            elif tool == "zip":
                output_path = os.path.join(temp_dir, "output.zip")
                cmd = [executable, "-9", "-j", output_path, base_name]
            elif tool == "flac":
                output_path = os.path.join(temp_dir, "output.flac")
                cmd = [executable, "-8", "-f", "-o", output_path, base_name]

            if tool in ["7zip", "winrar", "zip", "flac"]:
                subprocess.run(
                    cmd,
                    cwd=temp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    timeout=60,
                )
            elif tool in ["gzip", "bzip2"]:
                if tool == "gzip":
                    output_path = f"{temp_input_path}.gz"
                if tool == "bzip2":
                    output_path = f"{temp_input_path}.bz2"
                cmd = [executable, "-9", temp_input_path]
                subprocess.run(
                    cmd,
                    cwd=temp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    timeout=60,
                )

            comp_size = os.path.getsize(output_path)
            ratio = original_size / comp_size if comp_size > 0 else 1.0

            final_download_path = f"compressed_output.{tool}"
            shutil.copyfile(output_path, final_download_path)

            return ratio, time.time() - start_time, final_download_path
        except Exception:
            return 1.0, time.time() - start_time, None


def get_prediction(file_path, model_name):
    """STAGE 1: Performs only the fast feature extraction and model prediction."""
    start_time = time.time()
    original_size = os.path.getsize(file_path)
    allowed_extensions = [
        ".txt",
        ".csv",
        ".json",
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".wav",
    ]
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in allowed_extensions:
        raise ValueError(f"Invalid file type '{ext}'.")
    if original_size > 50 * 1024 * 1024:
        raise ValueError("File is too large for this demo (max 50 MB).")

    raw_features, key_insights = compile_feature_vector(file_path)
    scaled_features = INFERENCE_CACHE["SCALER"].transform(raw_features)

    model = INFERENCE_CACHE[model_name]

    if model_name == "DL-ML Hybrid":
        extractor = INFERENCE_CACHE["HYBRID_EXTRACTOR"]
        feature_layer_model = tf.keras.Model(
            inputs=extractor.inputs, outputs=extractor.layers[-2].output
        )
        deep_features = feature_layer_model.predict(scaled_features, verbose=0)
        pred_enc = model.predict(deep_features)[0]
    else:
        wide_indices = list(range(12))
        if model_name == "Wide & Deep":
            pred_probs = model.predict(
                [scaled_features, scaled_features[:, wide_indices]], verbose=0
            )
        else:
            pred_probs = model.predict(scaled_features, verbose=0)
        pred_enc = np.argmax(pred_probs, axis=1)[0]

    recommended_tool = INFERENCE_CACHE["LABEL_ENCODER"].inverse_transform([pred_enc])[0]
    prediction_time = time.time() - start_time

    return recommended_tool, key_insights, prediction_time


def run_full_benchmark(file_path, recommended_tool):
    """STAGE 2: Performs the slow, full compression benchmark."""
    original_size = os.path.getsize(file_path)
    benchmark_results = {"ratios": {}, "sizes": {}}
    time_start_full = time.time()
    all_tools = ["7zip", "zip", "winrar", "gzip", "bzip2", "flac"]

    for tool in all_tools:
        ext = os.path.splitext(file_path)[1].lower().strip(".")
        if tool == "flac" and ext != ".wav":
            ratio, size = 0.0, original_size
        else:
            ratio, _, _ = run_single_compression(tool, file_path)
            size = original_size / ratio if ratio > 0 else original_size

        benchmark_results["ratios"][tool] = ratio
        benchmark_results["sizes"][tool] = size

    brute_force_time = time.time() - time_start_full

    df_bench = pd.DataFrame(
        {
            "Ratio": benchmark_results.get("ratios", {}),
            "Size (KB)": {
                k: v / 1024 for k, v in benchmark_results.get("sizes", {}).items()
            },
        }
    ).sort_values(by="Ratio", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4"] * len(df_bench)
    if recommended_tool in df_bench.index:
        idx = df_bench.index.get_loc(recommended_tool)
        colors[idx] = "#ff7f0e"
    ax.bar(df_bench.index, df_bench["Ratio"], color=colors)
    ax.set_title("Compression Ratio Comparison (Higher is Better)", fontsize=16)
    ax.set_ylabel("Compression Ratio", fontsize=12)
    ax.set_xlabel("Compression Tool", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    summary_report = f"Original File Size: {original_size/1024:.2f} KB\n\n{df_bench.to_string(float_format='%.2f')}"

    return fig, summary_report, brute_force_time

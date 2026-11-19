import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import base64
import hashlib
from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from io import BytesIO


# ==========================================
# 1. PAGE CONFIGURATION (must be first st call)
# ==========================================

st.set_page_config(
    page_title="Smart Waste AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==========================================
# 2. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_fine_tuned_model.keras"

DEFAULT_IMG_SIZE = (160, 160)
TEST_ACCURACY = 84.90
LOW_CONFIDENCE_THRESHOLD = 60.0  # below this, show a "retake photo" hint
MAX_FILE_MB = 10  # keep equal to maxUploadSize in .streamlit/config.toml

# How pixels are scaled before they go into the model:
#   "raw"          -> 0..255 (use if preprocessing layer is INSIDE the model)
#   "rescale"      -> 0..1   (use if you trained with Rescaling(1/255))
#   "mobilenet_v2" -> -1..1  (use if you trained with mobilenet_v2.preprocess_input)
# Keep this the same as training. Original code behaviour = "raw".
PREPROCESS_MODE = "raw"

# Order MUST match the training class order (alphabetical in TrashNet folders).
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CATEGORY_INFO = {
    "cardboard": {
        "emoji": "📦", "color": "#a8691d", "kind": "Recyclable",
        "tip": "Flatten the box, keep it dry and remove tape or labels.",
    },
    "glass": {
        "emoji": "🍾", "color": "#0f8f82", "kind": "Recyclable",
        "tip": "Rinse it out. Don't mix with ceramics, mirrors or bulbs.",
    },
    "metal": {
        "emoji": "🥫", "color": "#4f6272", "kind": "Recyclable",
        "tip": "Rinse cans and tins; crush them to save space.",
    },
    "paper": {
        "emoji": "📄", "color": "#2f6fb3", "kind": "Recyclable",
        "tip": "Keep it clean and dry. Greasy or wet paper can't be recycled.",
    },
    "plastic": {
        "emoji": "🧴", "color": "#7c4fd6", "kind": "Check the recycling code",
        "tip": "Look for the resin code, rinse it and remove caps.",
    },
    "trash": {
        "emoji": "🗑️", "color": "#6b7280", "kind": "General waste",
        "tip": "Not recyclable. Put it in the general waste bin.",
    },
}


# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');

/* Fonts: applied to text elements only, so Streamlit's icon font keeps working */
html, body, .stApp, .stMarkdown, .stMarkdown p, label, button, input, textarea {
    font-family: 'Inter', sans-serif;
}
span[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

.stApp { background-color: #f3f7f2; color: #1f3a2d; }
.stApp p, .stApp label, .stApp li { color: #1f3a2d; }

/* Top bar: match the page, hide Deploy / menu */
header[data-testid="stHeader"] { background: #f3f7f2; }
[data-testid="stToolbar"], [data-testid="stAppDeployButton"], .stDeployButton, #MainMenu, footer {
    display: none !important;
    visibility: hidden;
}

.block-container {
    max-width: 1150px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 { color: #123b2a !important; font-family: 'Bricolage Grotesque', sans-serif !important; }

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(120deg, #0d4f35 0%, #178a55 100%);
    border-radius: 22px;
    padding: 34px 38px;
    margin-bottom: 28px;
    display: flex; justify-content: space-between; align-items: center;
    gap: 24px; flex-wrap: wrap;
}
.hero-title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 42px; font-weight: 800; line-height: 1.1;
    color: #ffffff !important; margin: 0;
}
.hero-sub { font-size: 17px; color: #d5efe0 !important; margin-top: 8px; max-width: 520px; }
.hero-stats { display: flex; gap: 12px; flex-wrap: wrap; }
.hero-chip {
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 14px; padding: 12px 18px; min-width: 110px;
}
.hero-chip b { display: block; font-size: 22px; color: #ffffff !important; font-family: 'Bricolage Grotesque', sans-serif; }
.hero-chip span { font-size: 13px; color: #cfeadb !important; }

/* ---------- Section titles ---------- */
.section-title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 26px; font-weight: 700; color: #174b35 !important; margin-bottom: 4px;
}
.section-sub { color: #5d7266 !important; margin-bottom: 14px; }

/* ---------- Source switch (segmented pills) ---------- */
div[role="radiogroup"] { gap: 10px; }
div[role="radiogroup"] label {
    background: #ffffff; border: 1px solid #cfe0d4; border-radius: 999px;
    padding: 6px 20px; cursor: pointer; transition: background .15s, border-color .15s;
}
div[role="radiogroup"] label > div:first-child { display: none; }
div[role="radiogroup"] label p { font-weight: 600; margin: 0; }
div[role="radiogroup"] label:has(input:checked) { background: #16834b; border-color: #16834b; }
div[role="radiogroup"] label:has(input:checked) p { color: #ffffff !important; }

/* ---------- Uploader / camera ---------- */
div[data-testid="stFileUploader"] section {
    background: #ffffff; border: 2px dashed #8fb7a0; border-radius: 16px; padding: 22px;
}
div[data-testid="stFileUploader"] section:hover { border-color: #16834b; background: #fbfefc; }
div[data-testid="stFileUploader"] label,
div[data-testid="stCameraInput"] label { color: #294638 !important; font-weight: 600; }
div[data-testid="stFileUploader"] small { color: #6b7f73 !important; }
div[data-testid="stFileUploader"] button,
div[data-testid="stCameraInput"] button {
    background: #16834b; color: #ffffff !important; border: none; border-radius: 10px; font-weight: 600;
}
div[data-testid="stFileUploader"] button:hover,
div[data-testid="stCameraInput"] button:hover { background: #116b3d; }
div[data-testid="stFileUploader"] button span,
div[data-testid="stFileUploader"] button p { color: #ffffff !important; }

div[data-testid="stImage"] img { border-radius: 16px; border: 1px solid #dce8df; }

/* ---------- Bin strip (the signature element) ---------- */
.bins { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin: 6px 0 4px 0; }
.bin {
    background: #ffffff; border: 1px solid #dce8df; border-radius: 16px;
    padding: 14px 8px 12px; text-align: center; opacity: .62;
    transition: transform .25s ease, box-shadow .25s ease, opacity .25s ease;
}
.bin .e { font-size: 30px; line-height: 1.1; }
.bin .n { font-size: 14px; font-weight: 600; margin-top: 4px; color: #4b6355 !important; }
.bin.idle { opacity: 1; }
.bin.on {
    opacity: 1; transform: translateY(-6px);
    box-shadow: 0 10px 22px rgba(20, 60, 40, .18);
}
.bin.on .n { color: #ffffff !important; }

/* ---------- Result card ---------- */
.result-card {
    background: #ffffff; border: 1px solid #dce8df; border-left-width: 7px;
    border-radius: 18px; padding: 22px 24px;
    display: flex; gap: 20px; align-items: center; margin-bottom: 14px;
}
.result-emoji { font-size: 56px; line-height: 1; }
.result-label { font-size: 14px; color: #6b7f73 !important; }
.result-name {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 40px; font-weight: 800; line-height: 1.1;
}
.badge {
    display: inline-block; margin-top: 8px; padding: 4px 12px;
    border-radius: 999px; font-size: 13px; font-weight: 600;
    background: #e8f4ec; color: #16663f !important;
}
.tip-box {
    background: #eaf5ee; border-radius: 12px; padding: 14px 18px;
    color: #1d4a35 !important; margin: 10px 0 14px;
}
.tip-box b { color: #123b2a !important; }

/* ---------- Stat tiles ---------- */
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.stat { background: #ffffff; border: 1px solid #dce8df; border-radius: 14px; padding: 14px 18px; }
.stat .k { font-size: 14px; color: #6b7f73 !important; }
.stat .v {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 28px; font-weight: 800; color: #174b35 !important; line-height: 1.2;
}
.preview { width: 100%; border-radius: 16px; border: 1px solid #dce8df; display: block; }
.preview-name { color: #6b7f73 !important; font-size: 14px; margin-top: 6px; }

/* ---------- Probability bars ---------- */
.prob-row { margin-bottom: 12px; }
.prob-head { display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 4px; }
.prob-track { background: #e3ece6; border-radius: 999px; height: 10px; overflow: hidden; }
.prob-fill { height: 10px; border-radius: 999px; }

/* ---------- Steps (empty state) ---------- */
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 18px; }
.step { background: #ffffff; border: 1px solid #dce8df; border-radius: 16px; padding: 18px 20px; }
.step b { font-family: 'Bricolage Grotesque', sans-serif; font-size: 18px; color: #174b35 !important; }
.step p { margin: 6px 0 0; color: #5d7266 !important; font-size: 15px; }
.step .num {
    display: inline-flex; width: 28px; height: 28px; border-radius: 50%;
    background: #16834b; color: #ffffff !important; font-weight: 700;
    align-items: center; justify-content: center; margin-bottom: 10px; font-size: 14px;
}

/* ---------- History ---------- */
.hist { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
.hist-item {
    background: #ffffff; border: 1px solid #dce8df; border-radius: 999px;
    padding: 6px 14px; font-size: 14px; font-weight: 500;
}

/* ---------- Info cards ---------- */
.info-card { background: #ffffff; border: 1px solid #dce8df; border-radius: 14px; padding: 18px 20px; }
.info-card .k { font-size: 14px; color: #6b7f73 !important; }
.info-card .v {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 30px; font-weight: 800; color: #174b35 !important;
}
.spacer { height: 14px; }

.footer { text-align: center; color: #7d8f84 !important; margin-top: 50px; font-size: 14px; }

@media (max-width: 760px) {
    .hero { padding: 24px 20px; }
    .hero-title { font-size: 30px; }
    .bins { grid-template-columns: repeat(3, 1fr); }
    .steps { grid-template-columns: 1fr; }
    .result-card { flex-direction: column; align-items: flex-start; }
    .result-name { font-size: 30px; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource(show_spinner=False)
def load_waste_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception:
        # Fallback for models with Lambda layers (Keras 3 safe mode)
        return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)


def get_input_size(model):
    """Read H, W from the model so a size mismatch can't break prediction."""
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        height, width = shape[1], shape[2]
        if isinstance(height, int) and isinstance(width, int):
            return (width, height)
    except Exception:
        pass
    return DEFAULT_IMG_SIZE


# ==========================================
# 5. IMAGE HANDLING
# ==========================================

def open_image(image_bytes):
    """Open image, fix phone rotation (EXIF) and handle transparent PNGs."""
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)

    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        return background

    return image.convert("RGB")


def preprocess_image(image, size):
    image = image.resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)

    if PREPROCESS_MODE == "rescale":
        array = array / 255.0
    elif PREPROCESS_MODE == "mobilenet_v2":
        array = array / 127.5 - 1.0

    return np.expand_dims(array, axis=0)


# ==========================================
# 6. PREDICTION
# ==========================================

def predict_waste(model, image):
    processed = preprocess_image(image, get_input_size(model))
    predictions = model.predict(processed, verbose=0)
    probabilities = np.asarray(predictions[0], dtype=np.float32)

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {len(probabilities)} outputs but "
            f"{len(CLASS_NAMES)} categories are configured."
        )

    # If the model outputs logits instead of probabilities, convert them
    if probabilities.min() < 0 or not np.isclose(probabilities.sum(), 1.0, atol=1e-2):
        exp = np.exp(probabilities - probabilities.max())
        probabilities = exp / exp.sum()

    index = int(np.argmax(probabilities))
    return CLASS_NAMES[index], float(probabilities[index] * 100), probabilities


# ==========================================
# 7. UI HELPERS
# ==========================================

def render_probability_bars(probabilities):
    order = np.argsort(probabilities)[::-1]
    rows = []
    for rank, idx in enumerate(order):
        name = CLASS_NAMES[idx]
        info = CATEGORY_INFO[name]
        pct = float(probabilities[idx] * 100)
        opacity = "1" if rank == 0 else "0.55"
        rows.append(
            f'<div class="prob-row"><div class="prob-head">'
            f'<span>{info["emoji"]} {escape(name.capitalize())}</span><span>{pct:.2f}%</span></div>'
            f'<div class="prob-track"><div class="prob-fill" '
            f'style="width:{max(pct, 0.5):.2f}%;background:{info["color"]};opacity:{opacity}"></div>'
            f'</div></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_preview(image, name, max_side=900):
    """Show the image as an inline data-URI so it doesn't depend on Streamlit's media server."""
    preview = image.copy()
    preview.thumbnail((max_side, max_side))
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    st.markdown(
        f'<img class="preview" src="data:image/jpeg;base64,{encoded}" alt="Uploaded waste image">'
        f'<div class="preview-name">{escape(name)}</div>',
        unsafe_allow_html=True,
    )


def render_result(category, confidence, probabilities):
    info = CATEGORY_INFO[category]

    st.markdown(
        f'<div class="result-card" style="border-left-color:{info["color"]}">'
        f'<div class="result-emoji">{info["emoji"]}</div>'
        f'<div><div class="result-label">Predicted waste category</div>'
        f'<div class="result-name" style="color:{info["color"]} !important">{escape(category.capitalize())}</div>'
        f'<span class="badge">{escape(info["kind"])}</span></div></div>',
        unsafe_allow_html=True,
    )

    order = np.argsort(probabilities)[::-1]
    second = CLASS_NAMES[order[1]]
    second_pct = float(probabilities[order[1]] * 100)

    st.markdown(
        f'<div class="stats">'
        f'<div class="stat"><div class="k">Confidence</div><div class="v">{confidence:.2f}%</div></div>'
        f'<div class="stat"><div class="k">Second guess</div>'
        f'<div class="v">{escape(second.capitalize())} <span style="font-size:16px;font-weight:600">{second_pct:.1f}%</span></div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.progress(min(max(confidence / 100, 0.0), 1.0))

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            "The model isn't very sure about this one. Try a closer photo "
            "with good light and a plain background, with only one item in frame."
        )

    st.markdown(
        f'<div class="tip-box"><b>How to dispose of it:</b> {escape(info["tip"])}</div>',
        unsafe_allow_html=True,
    )


def render_bins(active=None):
    tiles = []
    for name in CLASS_NAMES:
        info = CATEGORY_INFO[name]
        if active is None:
            cls, style = "bin idle", ""
        elif name == active:
            cls, style = "bin on", f'style="background:{info["color"]};border-color:{info["color"]}"'
        else:
            cls, style = "bin", ""
        tiles.append(
            f'<div class="{cls}" {style}><div class="e">{info["emoji"]}</div>'
            f'<div class="n">{escape(name.capitalize())}</div></div>'
        )
    st.markdown(f'<div class="bins">{"".join(tiles)}</div>', unsafe_allow_html=True)


def render_steps():
    st.markdown(
        '<div class="steps">'
        '<div class="step"><div class="num">1</div><b>Add a photo</b>'
        '<p>Upload a file or take a picture of one item.</p></div>'
        '<div class="step"><div class="num">2</div><b>The model looks</b>'
        '<p>MobileNetV2 scores the image against 6 categories.</p></div>'
        '<div class="step"><div class="num">3</div><b>Sort it</b>'
        '<p>You get the right bin, the confidence and a disposal tip.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def add_history(image_hash, name, category, confidence):
    history = st.session_state.setdefault("history", [])
    if any(item["hash"] == image_hash for item in history):
        return
    history.insert(0, {"hash": image_hash, "name": name, "category": category, "confidence": confidence})
    del history[8:]


def render_history():
    history = st.session_state.get("history", [])
    if not history:
        return
    st.markdown('<div class="section-title" style="font-size:20px;margin-top:22px">Recent results</div>', unsafe_allow_html=True)
    items = "".join(
        f'<div class="hist-item">{CATEGORY_INFO[h["category"]]["emoji"]} {escape(h["category"].capitalize())} '
        f'· {h["confidence"]:.0f}% · {escape(str(h["name"])[:22])}</div>'
        for h in history
    )
    st.markdown(f'<div class="hist">{items}</div>', unsafe_allow_html=True)


# ==========================================
# 8. HEADER
# ==========================================

st.markdown(
    f"""
<div class="hero">
  <div>
    <div class="hero-title">♻️ Smart Waste AI</div>
    <div class="hero-sub">Take or upload a photo of an item and find out which waste category it belongs to.</div>
  </div>
  <div class="hero-stats">
    <div class="hero-chip"><b>{TEST_ACCURACY:.1f}%</b><span>Test accuracy</span></div>
    <div class="hero-chip"><b>{len(CLASS_NAMES)}</b><span>Categories</span></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 9. IMAGE INPUT
# ==========================================

st.markdown('<div class="section-title">Classify your waste</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Choose how to add the image. It is analysed as soon as it is added.</div>',
    unsafe_allow_html=True,
)

input_method = st.radio(
    "Image source",
    ["Upload image", "Use camera"],
    horizontal=True,
    label_visibility="collapsed",
)

if input_method == "Upload image":
    source_file = st.file_uploader(
        "Choose a waste image",
        type=["jpg", "jpeg", "png"],
        help=f"JPG, JPEG or PNG, up to {MAX_FILE_MB} MB.",
        key="waste_image_uploader",
    )
else:
    source_file = st.camera_input("Take a photo of the item", key="waste_camera")


# ==========================================
# 10. PREVIEW AND PREDICTION
# ==========================================

if source_file is not None:
    image_bytes = source_file.getvalue()
    file_name = getattr(source_file, "name", "camera_photo.jpg")
    display_name = "Camera photo" if file_name.startswith("camera") else file_name

    if len(image_bytes) > MAX_FILE_MB * 1024 * 1024:
        st.error(f"This file is larger than {MAX_FILE_MB} MB. Choose a smaller image.")
    else:
        try:
            uploaded_image = open_image(image_bytes)
        except (UnidentifiedImageError, OSError) as error:
            uploaded_image = None
            st.error(f"This file can't be read as an image. Try another JPG or PNG. ({error})")

        if uploaded_image is not None:
            image_hash = hashlib.md5(image_bytes).hexdigest()
            image_width, image_height = uploaded_image.size

            cached = st.session_state.get("prediction")
            if cached is None or cached.get("hash") != image_hash:
                cached = None
                try:
                    with st.spinner("Analysing your image..."):
                        model = load_waste_model()
                        category, confidence, probabilities = predict_waste(model, uploaded_image)
                    cached = {
                        "hash": image_hash,
                        "category": category,
                        "confidence": confidence,
                        "probabilities": probabilities.tolist(),
                    }
                    st.session_state["prediction"] = cached
                except FileNotFoundError as error:
                    st.session_state.pop("prediction", None)
                    st.error(f"{error}. Check that the model file is in the `models` folder.")
                except Exception as error:
                    st.session_state.pop("prediction", None)
                    st.error(f"Prediction failed: {error}")

            st.divider()
            left_col, right_col = st.columns([1, 1], gap="large")

            with left_col:
                st.subheader("Your image")
                render_preview(uploaded_image, display_name)
                st.caption(f"Size: {image_width} × {image_height} px")

                if cached is not None:
                    st.markdown('<div class="section-title" style="font-size:20px;margin-top:18px">All categories</div>', unsafe_allow_html=True)
                    render_probability_bars(np.asarray(cached["probabilities"], dtype=np.float32))

            with right_col:
                st.subheader("AI prediction")
                if cached is not None:
                    add_history(image_hash, display_name, cached["category"], cached["confidence"])
                    render_result(
                        cached["category"],
                        cached["confidence"],
                        np.asarray(cached["probabilities"], dtype=np.float32),
                    )

            if cached is not None:
                st.markdown('<div class="section-title" style="font-size:20px;margin-top:8px">Your item goes in</div>', unsafe_allow_html=True)
                render_bins(cached["category"])

    render_history()
else:
    render_bins(None)
    render_steps()
    st.session_state.pop("prediction", None)
    render_history()


# ==========================================
# 11. MODEL INFORMATION
# ==========================================

st.divider()
st.markdown('<div class="section-title">About the model</div>', unsafe_allow_html=True)
st.write("")

info_cols = st.columns(4)
info_items = [
    ("Test accuracy", f"{TEST_ACCURACY:.2f}%"),
    ("Waste categories", str(len(CLASS_NAMES))),
    ("Input size", f"{DEFAULT_IMG_SIZE[0]} × {DEFAULT_IMG_SIZE[1]}"),
    ("Architecture", "MobileNetV2"),
]
for col, (label, value) in zip(info_cols, info_items):
    with col:
        st.markdown(
            f'<div class="info-card"><div class="k">{label}</div><div class="v">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
st.caption(
    "Dataset: TrashNet. Accuracy is measured on the held-out test set. "
    "Categories: " + ", ".join(c.capitalize() for c in CLASS_NAMES) + "."
)


# ==========================================
# 12. FOOTER
# ==========================================

st.markdown('<div class="footer">Smart Waste AI | Deep Learning Project</div>', unsafe_allow_html=True)
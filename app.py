import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageEnhance
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --- Page Config ---
st.set_page_config(page_title="AI Image Forensic Lab", layout="wide")
st.title("🛡️ AI Image Forensic Lab")
st.markdown("Detecting Synthetic Media using Swin & ViT Ensemble Architectures.")

# --- Forensic Preprocessing Function ---
def get_ela(image, quality=90):
    """Performs Error Level Analysis to highlight digital artifacts."""
    resaved_file = "temp_resaved.jpg"
    image.save(resaved_file, "JPEG", quality=quality)
    resaved_image = Image.open(resaved_file)
    
    ela_image = ImageChops.difference(image, resaved_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

# --- Model Logic ---
@st.cache_resource # Caches models so they don't reload on every click
def load_models():
    model_path = "./models/detector_1"

    proc = AutoImageProcessor.from_pretrained(model_path)
    mod = AutoModelForImageClassification.from_pretrained(model_path)

    return [(proc, mod)]

# --- Main App ---
models = load_models()
uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(len(uploaded_files))
    
    for idx, file in enumerate(uploaded_files):
        with cols[idx]:
            img = Image.open(file).convert("RGB")
            st.image(img, caption=f"File: {file.name}", use_container_width=True)
            
            if st.button(f"Analyze {idx+1}"):
                with st.spinner("Running Forensic Scan..."):
                    scores = []
                    for proc, mod in models:
                        inputs = proc(images=img, return_tensors="pt")
                        with torch.no_grad():
                            outputs = mod(**inputs)
                            probs = F.softmax(outputs.logits, dim=-1)
                        
                        # Grab index 1 (usually 'Fake/Artificial')
                        scores.append(probs[0][1].item())
                    
                    final_score = sum(scores) / len(scores)
                    
                    # Display Results
                    if final_score > 0.5:
                        st.error(f"VERDICT: AI GENERATED ({final_score:.1%})")
                    else:
                        st.success(f"VERDICT: AUTHENTIC ({1-final_score:.1%})")
                    
                    # Show Forensic View
                    with st.expander("View Forensic Noise Analysis (ELA)"):
                        st.image(get_ela(img), caption="Error Level Analysis")
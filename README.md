# 🛡️ AI Image Forensic Lab

A high-precision forensic tool designed to distinguish between authentic photographs and AI-generated (synthetic) media. This application leverages an ensemble of state-of-the-art **Vision Transformers (ViT)** and **Swin Transformers** to detect subtle patterns invisible to the human eye.

## 🚀 Key Features
* **Multi-Model Ensemble:** Combines predictions from multiple deep learning architectures to reduce false negatives.
* **Error Level Analysis (ELA):** Includes a forensic preprocessing layer to highlight compression inconsistencies typical in AI generation.
* **Real-time Inference:** Built with Streamlit for a seamless, browser-based user experience.
* **Batch Processing:** Supports uploading and analyzing multiple images simultaneously.

## 🔬 How It Works
AI generators (like Stable Diffusion, Midjourney, and DALL-E) leave behind "digital fingerprints"—mathematical patterns in the frequency domain. 
1. **Transformer Analysis:** Our models analyze the global and local context of the image.
2. **Forensic Noise Check:** ELA highlights areas with different compression levels, often revealing where AI has "stitched" textures together.

## 🛠️ Tech Stack
* **Language:** Python 3.11+
* **ML Framework:** PyTorch & Hugging Face Transformers
* **UI Framework:** Streamlit
* **Forensics:** Pillow (PIL) for ELA implementation

## 📦 Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/deepfake-detector.git](https://github.com/your-username/deepfake-detector.git)
   cd deepfake-detector
# 🎨 StudioAI: Generative AI Image Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Diffusers-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white)
![Microsoft ElevAte](https://img.shields.io/badge/Microsoft_ElevAte-2025-0078D4?style=for-the-badge&logo=microsoft&logoColor=white)

## 📌 Overview
**StudioAI** is a fully functional Generative AI application built with Streamlit. This project demonstrates an end-to-end image generation pipeline utilizing the powerful **Stable Diffusion v1.5** model. 

Developed as a capstone project for the **Microsoft ElevAte Training Center 2025**, this application features both standard Text-to-Image generation and advanced Image-to-Image editing capabilities (Inpainting and seamless Outpainting/Zoom Out).

---

## ✨ Key Features
* **Text-to-Image Generation:** Generate high-quality images from descriptive text prompts.
* **Advanced Schedulers:** Support for multiple diffusion schedulers for fine-tuned generation.
* **Interactive Inpainting:** Draw masks directly on the web interface to edit, replace, or add specific objects to an existing image.
* **Seamless Outpainting (Zoom Out):** Expand the canvas of an image to reveal more of the surrounding environment using custom color-blocking and overlap masking techniques.
* **Cloud-Ready:** Clean architecture separated into logic and UI, fully optimized for deployment on Streamlit Community Cloud.

---

## 🚀 Live Demo & Preview
* **Live App:** [Insert your Streamlit App Link Here]
* **Video Demo:** Watch `video_demo_aplikasi_BFGAI.mp4` in this repository for a full walkthrough.

---

## 🛠️ Technology Stack
* **Frontend:** Streamlit, Streamlit Drawable Canvas
* **Machine Learning Framework:** PyTorch
* **Generative AI Models:** Hugging Face `diffusers` (`StableDiffusionPipeline`, `StableDiffusionInpaintPipeline`)
* **Image Processing:** Pillow (PIL), NumPy

---

## 💻 Installation & Local Setup

If you want to run this application locally on your machine, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/studio-ai-stable-diffusion.git](https://github.com/yourusername/studio-ai-stable-diffusion.git)
cd studio-ai-stable-diffusion

```

### 2. Install Dependencies

It is highly recommended to use a virtual environment.

```bash
pip install -r requirements.txt

```

### 3. Run the Application

```bash
streamlit run app.py

```

*Note: A GPU (CUDA-enabled) is highly recommended for faster inference times.*

---

## 📂 Project Structure

```text
📦 studio-ai-stable-diffusion
 ┣ 📜 app.py               # Main Streamlit application interface
 ┣ 📜 logic.py             # Core pipeline logic (Model loading, Txt2Img, Img2Img)
 ┣ 📜 requirements.txt     # Python dependencies
 ┣ 📜 README.md            # Project documentation
 ┣ 📓 Pipeline.ipynb  # Step-by-step experimental notebook
 ┗ 🎥 video_demo_aplikasi_BFGAI.mp4 # Application demonstration

```

---

## 👨‍💻 Author

**Erlangga Azhar** * Tech Enthusiast

* Connect with me on [LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/erlangga-azhar/)

---

*This project was developed as part of the Microsoft ElevAte Training Center 2025 curriculum.*


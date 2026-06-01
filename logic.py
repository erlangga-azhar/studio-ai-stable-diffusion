import torch
import gc
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler
)
from PIL import Image, ImageDraw, ImageFilter

# Memuat model Generative AI ke dalam memori.
# Mengimplementasikan 'Safe Mode' untuk mencegah Out of Memory (OOM) pada environment CPU (seperti Streamlit Community Cloud).
def load_models_cached():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models to {device}")

    # Safe Mode: Bypass pengunduhan model besar jika berjalan di instans CPU murni
    if device == "cpu":
        print("Environment CPU terdeteksi. Mengaktifkan Safe Mode (Bypass model load).")
        return "SAFE_MODE", "SAFE_MODE"

    # Memuat model normal untuk environment GPU
    weight_dtype = torch.float16
    pipe_txt2img = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True
    ).to(device)

    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", 
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True
    ).to(device)

    return pipe_txt2img, pipe_inpaint

# Membersihkan VRAM GPU untuk mencegah kebocoran memori antar iterasi generasi.
def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory Flushed!")

# Mengonfigurasi scheduler difusi berdasarkan input pengguna.
def set_scheduler(pipe, scheduler_name):
    if scheduler_name == "Euler A":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DPM++":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe

# Menghasilkan gambar dari teks (Text-to-Image).
def generate_image(pipe, prompt, neg_prompt, seed, steps, cfg, num_images=1, scheduler_name="Euler A"):
    
    # Penanganan untuk Safe Mode (Demo Portofolio)
    if pipe == "SAFE_MODE":
        img = Image.new('RGB', (512, 512), color=(30, 30, 30))
        d = ImageDraw.Draw(img)
        pesan = "[DEMO PORTFOLIO MODE]\n\nServer tidak memiliki instans GPU.\nSistem mencegah render untuk menghindari\nCrash (Out of Memory).\n\nLihat Video Demo di repositori\nuntuk hasil render asli."
        d.text((40, 200), pesan, fill=(255, 200, 0))
        return [img]
    
    pipe = set_scheduler(pipe, scheduler_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        num_images_per_prompt=num_images,
        generator=generator
    ).images
    
    return result

# Memodifikasi area gambar berdasarkan masking (Image-to-Image / Inpainting).
def run_inpainting(pipe, image, mask, prompt, strength):
    
    # Penanganan untuk Safe Mode
    if pipe == "SAFE_MODE":
        img = Image.new('RGB', (512, 512), color=(30, 30, 30))
        d = ImageDraw.Draw(img)
        pesan = "[DEMO PORTFOLIO MODE]\n\nFitur Inpainting dinonaktifkan\nkarena keterbatasan RAM Server."
        d.text((40, 220), pesan, fill=(255, 200, 0))
        return img

    # Format standarisasi mode warna dan penyesuaian resolusi mask
    if image.mode != "RGB": image = image.convert("RGB")
    if mask.mode != "L": mask = mask.convert("L")
    
    if image.size != mask.size:
        mask = mask.resize(image.size, resample=Image.NEAREST)

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        strength=strength
    ).images[0]
    
    return result

# Menyiapkan kanvas perluasan dan mask spasial untuk proses Outpainting (Zoom Out).
def prepare_outpainting(image, expand_pixels=128):
    w, h = image.size
    new_w = w + (expand_pixels * 2)
    new_h = h + (expand_pixels * 2)

    # Memastikan resolusi kanvas kelipatan 8 (standar model arsitektur difusi)
    new_w -= (new_w % 8)
    new_h -= (new_h % 8)

    # Mengisi area ekstensi kanvas dengan interpolasi blur dari gambar asli
    bg = image.resize((new_w, new_h), resample=Image.BICUBIC)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=50))

    canvas = bg.copy()
    paste_x = (new_w - w) // 2
    paste_y = (new_h - h) // 2
    canvas.paste(image, (paste_x, paste_y))

    # Membuat masking biner (Putih = Area difusi baru, Hitam = Area retensi gambar asli)
    mask = Image.new("L", (new_w, new_h), 255)
    inner_box = Image.new("L", (w, h), 0)
    
    mask.paste(inner_box, (paste_x, paste_y))

    return canvas, mask

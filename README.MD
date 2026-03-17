# Neural Style Transfer

### Link to stylised portfolio of images and videos: `https://drive.google.com/drive/folders/1J4Mb9XTQUdS7O_ybdBIqgrRz_U8AQ2Yj?usp=drive_link`

## Project Template 3.1: Final Project Code Instructions

The project uses a virtual environment created in WSL.

### Setup

Create the virtual environment:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

---

We already have a selected set of content and style images used to test the models.
See folder: `00_input_data`

---

# Classical NST Approaches

The classical NST methods were trained and used for stylization (inference) on an NVIDIA RTX 3060 GPU.

The classical NST implementations:

## 1. Baseline Neural Style Transfer (Optimisation-Based) – Gatys et al.

The implementation was taken from:

> F. Chollet, _Deep Learning with Python_, 2nd ed. Shelter Island, NY: Manning Publications Co., 2021, ch. 12.3, pp. 383–390.

- **Folder:** `1_gatys_nst`
- **Outputs:** `output_data/`

We experimented with different optimisers and content/style weights.

---

## 2. Real-Time Style Transfer (Single Style) – Johnson et al.

Implemented based on:

> J. Johnson, A. Alahi, and L. Fei-Fei,
> “Perceptual losses for real-time style transfer and superresolution,” ECCV 2016.

Video stylisation with optical flow based on:

> M. Ruder, A. Dosovitskiy, and T. Brox,
> “Artistic style transfer for videos,” 2016.

- **Folder:** `2_single_nst`
- **Models:** `models/` (3 trained styles)
- **Image outputs:** `output_data/images/`
- **Video outputs:** `output_data/videos/`

### Scripts to run from inside the respective folder

Training: `train.py`

Inference (image): `inference.py`

Inference (video with optical flow):`inference_video_opt_flow.py`

Inference (video without optical flow): `python3 inference_video.py`

```

### Notes

- Ensure paths are correctly set.
- Content data is expected to be included in:
  - `00_input_data/images`
  - `00_input_data/videos`

- Model and image names use **file stems (no extensions)**.

---

## 3. Multi-Style Transfer (AdaIN) – Huang et al.

Based on:

> X. Huang and S. Belongie,
> “Arbitrary style transfer in real-time with adaptive instance normalization,” 2017.
> https://arxiv.org/abs/1703.06868

- **Folder:** `3_adain_nst`
- **Model:** `models/`
- **Outputs:** `output_data/`

### Scripts to run from inside the respective folder

Training: `train.py`

Inference: `inference.py`

### Dataset Requirements

- MS COCO 2017
- WikiArt

Place datasets in: `00_input_data/`

# Diffusion-based NST Approaches

Since Stable Diffusion models are computationally intensive, we performed training and stylization in Google Colab using an NVIDIA Tesla T4 GPU.

- **Folder:** `4_stable_diffusion_nst`
- **Model LoRA:** `models/`
- **Outputs:** `output_data/`

## 1. NST with Stable Diffusion, IP-Adapter and ControlNet

Inference: `ip_adapter_controlnet.ipynb`

Make sure to load the content and style images and check the paths.

## 2. NST with Stable Diffusion and LoRA Fine-Tuning

### 2.1. Training
We have used a selected dataset of Van Gogh images.

Training: `lora_training.ipynb`

### 2.1. Inference without ControlNet
Use a trained model from `models` folder and selected content and style images.

Inference: `lora_inference_without_controlnet.ipynb`

### 2.2. Inference without ControlNet
Use a trained model from `models` folder and selected content and style images.

Inference: `lora_inference_controlnet.ipynb`
```

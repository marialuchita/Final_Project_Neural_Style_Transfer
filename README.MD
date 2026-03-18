# Neural Style Transfer

### Link to stylised portfolio of images and videos: [View Portfolio](https://drive.google.com/drive/folders/1J4Mb9XTQUdS7O_ybdBIqgrRz_U8AQ2Yj?usp=drive_link)

## Project Template 3.1: Final Project Code Instructions

The project uses a virtual environment created in WSL.

### Setup

Python used in the project is: `Python 3.12.3`

Create the virtual environment:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

Install the requirements. This is only for the classical NST implementations:

```bash
pip install -r requirements.txt
```

---

We already have a selected set of content and style images used to test the models.
See folder: `00_input_data`

---

# Classical NST Approaches

The classical NST methods were trained and used for stylization (inference) on my device with NVIDIA RTX 3060 GPU.

These are:

## 1. Baseline Neural Style Transfer (Optimisation-Based) – Gatys et al.

The implementation was taken from:

> F. Chollet, _Deep Learning with Python_, 2nd ed. Shelter Island, NY: Manning Publications Co., 2021, ch. 12.3, pp. 383–390.

- **Go to Folder:** `1_gatys_nst`
- **Visualisation Outputs are in the following subfolder:** `output_data/`

We experimented with different optimisers and content/style weights.

### Stylisation examples:

![Optimisation based NST](readme_visuals/results_gatys.png)

---

## 2. Real-Time Style Transfer (Single Style) – Johnson et al. for BOTH Images and Videos

Implemented based on:

> J. Johnson, A. Alahi, and L. Fei-Fei,
> “Perceptual losses for real-time style transfer and superresolution,” ECCV 2016.

Video stylisation with optical flow based on:

> M. Ruder, A. Dosovitskiy, and T. Brox,
> “Artistic style transfer for videos,” 2016.

- **Go to Folder:** `2_single_nst`
- **Trained Models are in the following subfolder:** `models/` (3 trained styles)
- **Image Visualisation Outputs are in the following subfolder:** `output_data/images/`
- **Video Visualisation Outputs are in the following subfolder:** `output_data/videos/`
  - with Optical Flow: `with_optical_flow/`
  - without Optical Flow: `without_optical_flow/`

### Scripts to run from inside the respective folder

Training: `train.py`

Inference (image): `inference.py`

Inference (video with optical flow):`inference_video_opt_flow.py`

Inference (video without optical flow): `inference_video.py`

### Notes

- Ensure paths are correctly set.
- Content data is expected to be included in:
  - `00_input_data/images`
  - `00_input_data/videos`

- Model and image names use **file stems (no extensions)** in the script.

### Stylisation examples:

![Single NST](readme_visuals/single_style_results_grid.png)

---

## 3. Multi-Style Transfer (AdaIN) – Huang et al.

Based on:

> X. Huang and S. Belongie,
> “Arbitrary style transfer in real-time with adaptive instance normalization,” 2017.

- **Go to Folder:** `3_adain_nst`
- **Trained Models:** `models/`
- **Visualisation Outputs:** `output_data/`

### Scripts to run from inside the respective folder

Training: `train.py`

Inference: `inference.py`

### Dataset Requirements

- MS COCO 2017
- WikiArt

Place datasets in: `00_input_data/`

### Stylisation examples:

![AdaIN NST](readme_visuals/adain0.8_results_grid.png)

---

# Diffusion-based NST Approaches

Since Stable Diffusion models are computationally intensive, we performed training and stylization in Google Colab using an NVIDIA Tesla T4 GPU.

- **Go to Folder:** `4_stable_diffusion_nst`
- **Trained Models LoRA:** `models/`
- **Visualisation Outputs:** `output_data/`

---

## 1. NST with Stable Diffusion, IP-Adapter and ControlNet

Inference: `ip_adapter_controlnet.ipynb`

Make sure to load the content and style images and check the paths.

### Stylisation examples:

#### NST with Stable Diffusion with IP-Adapter and ControlNet (single style)

![NST with Stable Diffusion with IP-Adapter and ControlNet (single style)](readme_visuals/stable_diff_1_1_6.png)

#### NST with Stable Diffusion with IP-Adapter and ControlNet (multiple styles)

![NST with Stable Diffusion with IP-Adapter and ControlNet (multiple styles)](readme_visuals/SDip0.5multiple.png)

---

## 2. NST with Stable Diffusion and LoRA Fine-Tuning

### 2.1. Training

We have used a selected dataset of Van Gogh images.

Training: `lora_train.ipynb`

### 2.1. Inference without ControlNet

Use a trained model from `models` folder and selected content images.

Inference: `lora_inference_without_controlnet.ipynb`

### Stylisation examples:

![NST with Stable Diffusion with LoRA fine-tuning without ControlNet](readme_visuals/lora_comparison_grid.png)

### 2.2. Inference with ControlNet

Use a trained model from `models` folder and selected content images.

Inference: `lora_inference_controlnet.ipynb`

### Stylisation examples:

![NST with Stable Diffusion with LoRA fine-tuning and ControlNet](readme_visuals/lora_comparison_grid2.png)

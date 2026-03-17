# Neural Style Transfer

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

### Scripts

Training:

```bash
python train.py
```

Inference (image):

```bash
python inference.py
```

Inference (video with optical flow):

```bash
python inference_video_opt_flow.py
```

Inference (video without optical flow):

```bash
python inference_video.py
```

### Notes

- Ensure paths are correctly set.
- Content data is expected in:
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

### Scripts

Training:

```bash
python train.py
```

Inference:

```bash
python inference.py
```

### Dataset Requirements

- MS COCO 2017
- WikiArt

Place datasets in:

```bash
00_input_data/
```

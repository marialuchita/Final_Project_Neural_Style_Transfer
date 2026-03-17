# Neural Style Transfer

## Project Template 3.1: Final Project Code Instructions

The project uses a virtual environment created in WSL.
To install venv run: python3 -m venv venv
To activate it run: source venv/bin/activate
Install the requirements using the following command: pip install -r requirements.txt

We already have a selected set of content and style images which we use to test the models. See folder: 00_input_data

# Classical NST approaches

Methods/Implementations:

1. **Baseline Neural Style Transfer (Optimisation-Based) by Gatys et al.**

The implementation for this was entirely taken from:
F. Chollet, Deep Learning with Python, 2nd ed. Shelter Island, NY: Manning Publications Co.,2021, ch. 12.3, pp. 383–390.

Relevant folder: 1_gatys_nst
Examples of stylised images are in the subfolder output_data

We have experimented here with other optimisers and content and style weights.

2. **Real-Time Style Transfer (Single Style) according to Johnson et al. for BOTH images and videos**

The implementation was done by me as per paper and architectures provided in:
J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-time style transfer and superresolution,” in Computer Vision – ECCV 2016, vol. 9906. Springer, 10 2016, pp. 694–711.

The implementation of optical flow for video stylisation using the single style transfer models trained was done by me as per approach provided by M. Ruder, A. Dosovitskiy, and T. Brox, “Artistic style transfer for videos,” Arxiv, 04 2016.

Relevant folder: 2_single_nst
Models trained are in the subfolder: models - There are 3 models trained for 3 different styles.
Examples of stylised images are in the subfolder output_data/images
Examples of stylised videos with and without optical flow are in the subfolder output_data/videos

For training use the script: train.py (we used the MS COCO 2017 dataset. Make sure to include it in the 00_input_data folder)
For inference (image) use the script: inference.py
For inference (video with optical flow) use the script: inference_video_opt_flow.py
For inference (video without optical flow) use the script: inference_video.py

NB: Make sure that the paths are correctly given. For content images and video images we currently use the paths pointing to the 00_input_data/images and 00_input_data/videos.
For the trained models, content images names we use only the stems without file ending.

3. **Multi-Style Transfer (AdaIN) according to Huang et al.**

The implementation was done by me as per paper and architectures provided in:
X. Huang and S. Belongie, “Arbitrary style transfer in real-time with adaptive instance normalization,” 2017. [Online]. Available: https://arxiv.org/abs/1703.06868

Relevant folder: 3_adain_nst
Model trained is in the subfolder: models.
Examples of stylised images are in the subfolder output_data.

For inference (image) use the script: inference.py
For training use the script: train.py (we used the MS COCO 2017 dataset and Wikiart dataset. Make sure to include it in the 00_input_data folder)

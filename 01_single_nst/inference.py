import os
import torch 
from transform_network import TransformNetwork
from datetime import datetime
from data_pipeline import process_image, save_as_image

CONTENT_IMAGE = "../00_input_data/images/01_content/puppy.jpg"
MODEL_PATH    = "models/20260204_125032_sunset/model_1_29570.pth"   # <-- update to your new saved name
OUTPUT_IMAGE  = "output_data/images"
USE_CUDA      = True
VGG_MEAN = (0.485, 0.456, 0.406)
VGG_STD = (0.229, 0.224, 0.225)

@torch.no_grad()
def main():
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = TransformNetwork().to(device).eval()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["transformer_state_dict"], strict=True)

    img = process_image(CONTENT_IMAGE, device)
    stylised_output = model(img)[0]  

 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"{OUTPUT_IMAGE}/{timestamp}.jpg"
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    save_as_image(stylised_output, out_dir)


if __name__ == "__main__":
    main()

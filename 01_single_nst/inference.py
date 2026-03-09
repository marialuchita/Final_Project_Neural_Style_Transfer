import os
import torch 
from transform_network import TransformNetwork
from data_pipeline import img_to_tensor, save_img, tensor_to_img, get_img, resize_img

STYLES = ["circles", "starry_night", "sunset"]
CONTENT_IMAGES_PATH = f"../00_input_data/images/01_content"
OUTPUT_IMAGE_FOLDER  = "output_data/images"
USE_CUDA = True
VGG_MEAN = (0.485, 0.456, 0.406)
VGG_STD = (0.229, 0.224, 0.225)

@torch.no_grad()
def main(content_img_names, style_id):
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_path = f"models/{style_id}.pth" 
    model = TransformNetwork().to(device).eval()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["transformer_state_dict"], strict=True)

    os.makedirs(os.path.dirname(OUTPUT_IMAGE_FOLDER), exist_ok=True)

    for img_name in content_img_names:
        img_path = os.path.join(CONTENT_IMAGES_PATH, f"{img_name}.jpg")  
        img = get_img(img_path)
        img_resized = resize_img(img=img)
        img_tensor = img_to_tensor(img_resized, device)

        stylised_output = model(img_tensor)[0]  
    
        out_dir = os.path.join(OUTPUT_IMAGE_FOLDER, f"{img_name}_{style_id}.jpg")
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        stylised_img = tensor_to_img(stylised_output)
        save_img(stylised_img, out_dir)


if __name__ == "__main__":
    style_id = STYLES[0] # 3 models trained. Pick the model here 0-2 ids
    content_img_names = [1, 2, 3, 4, 5, 6, 7, 8]
    main(content_img_names, style_id)
    print("Finished")

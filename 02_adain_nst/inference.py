import torch
from data_pipeline import get_img, img_to_tensor, tensor_to_img, save_img, resize_img
from style_transfer_network import StyleTransferNet
import os

STYLE_IMAGES_PATH = f"../00_input_data/images/02_style"
CONTENT_IMAGES_PATH = f"../00_input_data/images/01_content"
OUTPUT_IMAGE_FOLDER  = "output_data"

@torch.no_grad()
def stylise(content_img_names, style_id, model_path, alpha = 1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = torch.load(model_path, map_location=device)
    network = StyleTransferNet().to(device).eval()
    network.decoder.load_state_dict(model["state_dict"], strict=True)
    
    out_dir = os.path.join(OUTPUT_IMAGE_FOLDER, f"{style_id}_alpha{alpha}")
    os.makedirs(out_dir, exist_ok=True)

    style_img_path = os.path.join(STYLE_IMAGES_PATH, f"{style_id}.jpg")
    style_img = img_to_tensor(get_img(style_img_path), device)

    for content_name in content_img_names:
        content_img_path = os.path.join(CONTENT_IMAGES_PATH, f"{content_name}.jpg")
        content_img = get_img(content_img_path)
        content_img_resized = resize_img(img=content_img)
        content_img_tensor = img_to_tensor(content_img_resized, device)
    
        # forward pass
        output_network = network(content_img=content_img_tensor, style_img=style_img, alpha=alpha)
        stylised_img = tensor_to_img(output_network)

        file_dir = os.path.join(out_dir, f"{content_name}_{style_id}.jpg")
        os.makedirs(os.path.dirname(file_dir), exist_ok=True)

        save_img(stylised_img, file_dir)

    # denormalized_output = denormalize(output_network)
    # save_as_image(denormalized_output, output_path)

if __name__ == "__main__":
    content_img_names = [1, 2, 3, 4, 5, 6, 7, 8]
    style_img_names = ["abstract", "camille", "chronis", "circles", "gray", "starry_night", "sunset"]
    alpha = 0.4 # 1.0 style very strong, 0.8 normal combination with style, lower values less style
    model_path = "models/model_4_29570.pth"
    
    # Run all models:
    for style_id in style_img_names:
        stylise(content_img_names, style_id, model_path, alpha)

    print("Finished")
import torch
from data_pipeline import get_img, img_to_tensor, tensor_to_img, save_img
from style_transfer_network import StyleTransferNet
from datetime import datetime
import os

ALPHA = 1.0

@torch.no_grad()
def stylise(content_img_path: str, style_img_path: str, model_path: str, output_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = img_to_tensor(get_img(content_img_path), device)
    style_img = img_to_tensor(get_img(style_img_path), device)

    model = torch.load(model_path, map_location=device)
    network = StyleTransferNet().to(device).eval()
    network.decoder.load_state_dict(model["state_dict"], strict=True)

    # forward pass
    output_network = network(content_img=content_img, style_img=style_img, alpha=ALPHA)
    stylised_img = tensor_to_img(output_network)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(output_path, f"{timestamp}.jpg")
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    save_img(stylised_img, out_dir)

    # denormalized_output = denormalize(output_network)
    # save_as_image(denormalized_output, output_path)

def main():
    content_img_path = "../00_input_data/images/01_content/puppy.jpg"
    style_img_path = "../00_input_data/images/02_style/circles_paint.jpg"
    model_path = "models/model_4_29570.pth"
    output_path = "output_data/images"
    stylise(
        content_img_path=content_img_path,
        style_img_path=style_img_path,
        model_path=model_path,
        output_path=output_path
    )

if __name__ == "__main__":
    main()
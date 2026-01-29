from tkinter import Image

import torch
from data_pipeline import process_image, process_frame
import cv2 as cv
from style_transfer_network import StyleTransferNet

def stylise_video(video_path: str, style_img_path: str, model_path:str, output_path:str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    style_img = process_image(style_img_path, device)

    video = cv.VideoCapture(video_path)

    if not video.isOpened():
        raise RuntimeError("Could not open video")

    video_fps = video.get(cv.CAP_PROP_FPS)
    video_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

    model = torch.load(model_path, map_location=device)
    network = StyleTransferNet().to(device).eval()
    network.decoder.load_state_dict(model["state_dict"], strict=True)

    while True:
        status, frame = video.read()
        if not status:
            break
        content_frame = process_frame(frame, device)

        network_output = model(content_frame, style_img)

        # denormalize and save the output



def main():
    content_video_path = "videos/Traffic_Laramie_1.mp4"
    style_img_path = "images/wikiart/albrecht-durer_crucifixion-1498.jpg"
    model_path = "models/model_2_29570.pth"
    output_path = "outputs/video/..."
    stylise_video(
        video_path=content_video_path,
        style_img_path=style_img_path,
        model_path=model_path,
        output_path=output_path
    )

if __name__ == "__main__":
    main()
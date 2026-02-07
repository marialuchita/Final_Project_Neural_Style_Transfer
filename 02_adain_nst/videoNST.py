import torch
from data_pipeline import process_image, process_frame, tensor_to_frame
import cv2 as cv
from style_transfer_network import StyleTransferNet
import os
from datetime import datetime

ALPHA = 1.0

@torch.no_grad()
def stylise_video(video_path: str, style_img_path: str, model_path:str, output_folder:str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_folder, f"{timestamp}.mp4")


    # Style image
    style_img = process_image(style_img_path, device)

    # Video
    # video = cv.VideoCapture(video_path)
    video = cv.VideoCapture(0)
    if not video.isOpened():
        raise RuntimeError("Could not open video")

    video_fps = video.get(cv.CAP_PROP_FPS)
    video_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer =cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))
    
    if not writer.isOpened():
        raise RuntimeError("Could not open writer")
    
    # Load model
    model = torch.load(model_path, map_location=device)
    network = StyleTransferNet().to(device).eval()
    network.decoder.load_state_dict(model["state_dict"], strict=True)


    while True:
        status, frame = video.read()
        if not status:
            break
        content_frame = process_frame(frame, device)

        output_network = network(content_img=content_frame, style_img=style_img, alpha=ALPHA)
        # denormalize and save the output
        stylised_frame = tensor_to_frame(output_network)
        cv.imshow("Stylised", stylised_frame)
        writer.write(stylised_frame)
        if cv.waitKey(1) == ord('q'):
            break
    
    video.release()
    writer.release()
    cv.destroyAllWindows()

def main():
    content_video_path = "videos/Traffic_Laramie_1.mp4"
    style_img_path = "images/style/starry_night.jpg"
    model_path = "models/model_4_29570.pth"
    output_folder = "outputs/videos"
    stylise_video(
        video_path=content_video_path,
        style_img_path=style_img_path,
        model_path=model_path,
        output_folder=output_folder
    )

if __name__ == "__main__":
    main()
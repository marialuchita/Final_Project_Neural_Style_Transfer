import torch
import cv2 as cv
from transform_network import TransformNetwork
import os
from data_pipeline import get_img_from_frame, img_to_tensor, tensor_to_img
import time

STYLES = ["circles", "starry_night", "sunset"] # available models names
CONTENT_VIDEOS_PATH = f"../00_input_data/videos" # folder where the videos are located
OUTPUT_VIDEOS_FOLDER  = "output_data/videos/without_optical_flow" # folder where to save the stylised videos

@torch.no_grad()
def stylise_video(video_names: list, style_id:str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    os.makedirs(OUTPUT_VIDEOS_FOLDER, exist_ok=True)

    # Load model
    model_path = f"models/{style_id}.pth" 
    model = TransformNetwork().to(device).eval()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["transformer_state_dict"], strict=True)
    
    for video_name in video_names:
        out_dir = os.path.join(OUTPUT_VIDEOS_FOLDER, f"{video_name}_{style_id}.mp4")
        video_path = os.path.join(CONTENT_VIDEOS_PATH, f"{video_name}.mp4") 
        
        video = cv.VideoCapture(video_path)

        if not video.isOpened():
            raise RuntimeError("Could not open video")

        video_fps = video.get(cv.CAP_PROP_FPS)
        video_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        writer =cv.VideoWriter(out_dir, cv.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))
        
        if not writer.isOpened():
            raise RuntimeError("Could not open writer")
        
        start_time = time.time()

        while True:
            status, frame = video.read()
            if not status:
                break

            
            content_frame = img_to_tensor(get_img_from_frame(frame), device)
            output_network = model(content_frame)[0]  
            stylised_frame = tensor_to_img(output_network)
    
            writer.write(stylised_frame)

        end_time = time.time()
        print("total time ", (end_time-start_time)/60)
        video.release()
        writer.release()
        print("Saved:", out_dir)



if __name__ == "__main__":
    video_names = [1, 2, 3] 

    # Run one style model:
    style_id = STYLES[0]
    stylise_video(
        video_names,
        style_id
    )

    # Run all models:
    # for style_id in STYLES:
    #     stylise_video(
    #         video_names,
    #         style_id
    #     )
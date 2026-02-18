import torch
# from data_pipeline import img_to_tensor, frame_to_tensor, tensor_to_frame
import cv2 as cv
from transform_network import TransformNetwork
import os
from datetime import datetime
from data_pipeline import frame_to_tensor, tensor_to_frame



@torch.no_grad()
def stylise_video(video_path: str, model_path:str, output_folder:str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_folder, f"{timestamp}.mp4")

    # Video
    video = cv.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError("Could not open video")

    video_fps = video.get(cv.CAP_PROP_FPS)
    video_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer =cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))
    
    if not writer.isOpened():
        raise RuntimeError("Could not open writer")
    
    # Load model
    model = TransformNetwork().to(device).eval()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["transformer_state_dict"], strict=True)


    while True:
        status, frame = video.read()
        if not status:
            break
        content_frame = frame_to_tensor(frame, device)
        output_network = model(content_frame)[0]  
        stylised_frame = tensor_to_frame(output_network)
 
        writer.write(stylised_frame)
        # if cv.waitKey(1) == ord('q'):
        #     break
    
    video.release()
    writer.release()
    cv.destroyAllWindows()

def main():
 
    content_video_path = "../00_input_data/videos/119799-719443737_tiny.mp4"
    model_path = "models/20260203_132444_circles_paint/model_2_29570.pth"
    output_folder = "output_data/videos"
    stylise_video(
        video_path=content_video_path,
        model_path=model_path,
        output_folder=output_folder
    )

if __name__ == "__main__":
    main()
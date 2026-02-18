import torch
# from data_pipeline import img_to_tensor, frame_to_tensor, tensor_to_frame
import cv2 as cv
from transform_network import TransformNetwork
import os
from datetime import datetime
from data_pipeline import get_img_from_frame, to_tensor, tensor_to_frame
import numpy as np
import time

def compute_optical_flow(prev_frame_bgr: np.ndarray, current_frame_bgr: np.ndarray) -> np.ndarray:
    prev_gray = cv.cvtColor(prev_frame_bgr, cv.COLOR_BGR2GRAY)
    current_gray = cv.cvtColor(current_frame_bgr, cv.COLOR_BGR2GRAY)
    optical_flow = cv.calcOpticalFlowFarneback(prev_gray, current_gray, None, pyr_scale=0.5, levels=4, winsize=21, iterations=5, poly_n=7, poly_sigma=1.5, flags=0).astype(np.float32)
    return optical_flow

def warp_prev(img_bgr: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    grid_X, grid_Y = np.meshgrid(np.arange(w), np.arange(h))
    #print(grid_X,grid_Y)
    map_x = (grid_X - flow[..., 0]).astype(np.float32)
    map_y = (grid_Y - flow[..., 1]).astype(np.float32)
    return cv.remap(img_bgr, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)


def blend_with_current_stylisation(current_style_bgr: np.ndarray, prev_warp_bgr: np.ndarray) -> np.ndarray:
    beta = 0.5
    blended = (1.0 - beta) * current_style_bgr.astype(np.float32) + beta * prev_warp_bgr.astype(np.float32)
    clip_blended = np.clip(blended, 0, 255).astype(np.uint8)
    return clip_blended

def stabilize_with_optical_flow(previous_content: np.ndarray, previous_stylised: np.ndarray, current_content: np.ndarray, current_style: np.ndarray) -> np.ndarray:

    flow = compute_optical_flow(prev_frame_bgr=previous_content, current_frame_bgr=current_content)
    prev_warp = warp_prev(img_bgr=previous_stylised, flow=flow)
    blended_img = blend_with_current_stylisation(current_style_bgr=current_style, prev_warp_bgr=prev_warp)
    return blended_img

def get_resized_hw(h, w, short_size):
    if h > w:
        new_w = short_size
        new_h = int(round(h / w * short_size))
    else:
        new_h = short_size
        new_w = int(round(w / h * short_size))
    return new_h, new_w

def process_frame(frame, device, short_size=512):
    h, w = frame.shape[:2]
    new_h, new_w = get_resized_hw(h, w, short_size)
    resized_frame = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)
    tensor = to_tensor(get_img_from_frame(resized_frame), device)
    return resized_frame, tensor, new_h, new_w

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

    start_time = time.time()

    previous_content = None
    previous_stylised = None
    while True:
        status, frame = video.read()
        if not status:
            break
        

        content_frame, content_tensor, new_h, new_w = process_frame(frame, device)
        
        output_network = model(content_tensor)[0]  
        stylised_frame = tensor_to_frame(output_network)
        stylised_h, stylised_w = stylised_frame.shape[:2]

        if (stylised_h, stylised_w) != (new_h, new_w):
            if stylised_h > new_h or stylised_w > new_w:
                stylised_frame = cv.resize(stylised_frame, (new_w, new_h), interpolation=cv.INTER_AREA)
            else:
                stylised_frame = cv.resize(stylised_frame, (new_w, new_h), interpolation=cv.INTER_LINEAR)

        if previous_content is None:
            final_stylised_frame = stylised_frame
        else:
            final_stylised_frame = stabilize_with_optical_flow(previous_content=previous_content, previous_stylised=previous_stylised, current_content=content_frame, current_style=stylised_frame)
            

        stylised_frame_resized = cv.resize(final_stylised_frame, (video_width, video_height), interpolation=cv.INTER_LINEAR)
        writer.write(stylised_frame_resized)
        previous_content = content_frame
        previous_stylised = final_stylised_frame
        
        # if cv.waitKey(1) == ord('q'):
        #     break
    end_time = time.time()
    print("total time ", (end_time-start_time)/60)
    video.release()
    writer.release()
    cv.destroyAllWindows()

def main():
 
    content_video_path = "../00_input_data/videos/119799-719443737_tiny.mp4"
    model_path = "models/20260204_095223_starry_night/model_1_29570.pth"

    output_folder = "output_data/videos"
    stylise_video(
        video_path=content_video_path,
        model_path=model_path,
        output_folder=output_folder
    )

if __name__ == "__main__":
    main()
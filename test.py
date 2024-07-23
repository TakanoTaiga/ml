import cv2
from ultralytics import RTDETR
import gradio as gr
import tempfile
import os

# モデルのロード
model = RTDETR("rtdetr-l.pt")

# 推論関数の定義
def infer_video(video):
    cap = cv2.VideoCapture(video)
    output_frames = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=0.7, half=True)
            annotated_frame = results[0].plot()
            output_frames.append(annotated_frame)
    finally:
        cap.release()

    if not output_frames:
        return None

    # 出力動画を作成
    height, width, _ = output_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))

    try:
        for frame in output_frames:
            out.write(frame)
    finally:
        out.release()

    return video

# Gradioインターフェースの設定
iface = gr.Interface(
    fn=infer_video,
    inputs=gr.Video(label="Upload a Video"),
    outputs=gr.Video(label="Annotated Video"),
    title="RT-DETR Video Inference",
    description="Upload a video to perform inference using RT-DETR model"
)

if __name__ == "__main__":
    iface.launch()

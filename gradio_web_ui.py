import gradio as gr
from ultralytics import RTDETR
import pandas as pd
import io

ml_model_list = ["embryo_v0", "custom_2024", "robocon"]

model = RTDETR("rtdetr-l.pt")

def inference_image(image, mlmodel_name: str, confidence: float):
    results = model.predict(image, conf=confidence / 100, half=False)
    annotated_frame = results[0].plot()

    results = results[0].cpu()

    boxes_info = []

    for box_data in results.boxes:
        box = box_data.xywh[0]
        xmin = max(0, min(int(box[0] - box[2] / 2), 65535))
        ymin = max(0, min(int(box[1] - box[3] / 2), 65535))
        xmax = max(0, min(int(box[0] + box[2] / 2), 65535))
        ymax = max(0, min(int(box[1] + box[3] / 2), 65535))
        boxes_info.append([xmin, ymin, xmax, ymax, float(box_data.conf), model.names[int(box_data.cls)]])

    df = pd.DataFrame(boxes_info, columns=["xmin", "ymin", "xmax", "ymax", "confidence", "label"])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    return annotated_frame, csv_data

demo = gr.Interface(
    inference_image,
    [
        gr.Image(type="numpy", label="Upload an Image"),
        gr.Dropdown(
            ml_model_list, label="ML Model" ,info="Will add more animals later!"
        ),
        gr.Slider(0, 100, value=75, label="Confidence", step=5, info="Choose between 0% and 100%"),
    ],
    [
        gr.Image(type="numpy", label="result image"),
        gr.Textbox(label="Bounding Boxes CSV"),
    ]
)

if __name__ == "__main__":
    demo.launch()

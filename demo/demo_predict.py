import cv2
import os
from ultralytics import RTDETR

model = RTDETR('/home/taiga/ai_ws/ml-moriyaken/runs/train2/weights/best.pt')

image_folder = '/home/taiga/ai_ws/jpeg_data/1' #1

image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
image_files.sort()  # ファイル名でソート

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    
    frame = cv2.imread(image_path)
    if frame is None:
        continue
    
    results = model.predict(frame, conf=0.4)
    annotated_frame = results[0].plot()
    
    cv2.imshow("RT-DETR Inference", annotated_frame)
    
    if cv2.waitKey(100) != -1:
        break

cv2.destroyAllWindows()

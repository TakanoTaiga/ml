from ultralytics import RTDETR
model = RTDETR('rtdetr-l.pt')
results = model.train(data='./out_yaml/04_02_20_58_01_dataset/data.yaml', epochs=3, imgsz=640, project="./runs/")

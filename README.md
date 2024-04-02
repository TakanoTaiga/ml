# 1. Preparing the Dataset

Please prepare a large number of image data captured with your smartphone or camera. For testing purposes, I have prepared images of rock, paper, and scissors. Don't forget to convert them to the png or jpg format.

And this time, we will be using the **Easy-Peasy Ultimate AI Tool** that I created.

First, `git clone`:

```bash
git clone https://github.com/TakanoTaiga/ml.git
```

When you clone, you will get a repository with files and folders like this:

![Screenshot 2024-04-02 at 21.39.58.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/899689/99534135-00fd-6272-b6b5-f1caa3e10427.png)

Then, please put all the prepared images into the `input_image` folder. (No preprocessing is required at this stage)

Next, open a terminal, resize the images, convert them all to jpg, and rename them using the following script:

```bash
cd ./ml
python3 set_format.py
```

You will see that the `out_image` folder is generated. Check inside to ensure that there are plenty of images.

![Screenshot 2024-04-02 at 21.44.31.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/899689/736dd822-4c0f-75b4-c0e9-bc16b61921ee.png)

# Annotation

We will use the coco-annotator for annotation.

https://github.com/jsbroks/coco-annotator

It's easy to use, just use `docekr-compose`:

```bash
git clone https://github.com/jsbroks/coco-annotator.git
cd coco-annotator
docekr-compose up
```

Now, access [localhost:5000](http://localhost:5000/). You can stop it with ctrl-c, but afterward, start and stop using the following commands:

```bash:start
cd coco-annotator
docker-compose start
```

```bash:stop
cd coco-annotator
docker-compose stop
```

Once it's up, create a dataset in coco-annotator and copy the contents of the `out_image` folder prepared earlier into it. (Do not delete or crop any images within the `out_image` folder).

After annotating, download the JSON file of the annotation results and copy it to the `ml/input_label` folder.

# Training

After completing the data preparation, it's time for training. To train, you need to convert the current coco format to the yolo format. I've prepared a script for that, so let's execute it:

```
cd ml
python3 coco2yolo.py
```

You will see many folders generated. Once the `train_rtdetr_xxxxx.py` files are generated, setup is complete.

![Screenshot 2024-04-02 at 22.01.56.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/899689/83bc0ca7-1d16-20b9-c74b-c140485be942.png)

Now, let's start the training.

**For Jetson Environment:**
Start the container and then execute the generated python file:

```bash
./start.sh
```

After starting, (replace `xxxxx` appropriately):

```bash
python3 train_rtdetr_xxxxx.py
```

**For x86+NVIDIA GPU Environment:**
Execute the generated python file without starting the container:

```bash
python3 train_rtdetr_xxxxx.py
```

Once training is complete, the weight files and log data will be generated in `ml/run/trainNN`. This completes the training.

# Inference

You should find a `.pt` file in the `train/weight` folder. Place either the best or last one in a suitable folder. Then, save the following code to a Python file and execute it. It will automatically read from the camera and start inference. Replace `hogehoge.pt` accordingly:

```python
import cv2
from ultralytics import RTDETR

model = RTDETR('hogehoge.pt')

# Open the web camera stream
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    k = cv2.waitKey(1)

    if k != -1:
        break
    if success:
        results = model.predict(frame, conf=0.7, half=True)
        annotated_frame = results[0].plot()

        cv2.imshow("RT-DETR Inference", annotated_frame)
cap.release()
cv2.destroyAllWindows()
```

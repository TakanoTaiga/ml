docker run -it --rm --runtime nvidia --shm-size=32G -v /home/taiga/ml/datasets:/usr/src/datasets -v /home/taiga:/home/root -w /home/root/ml --network host ultralytics/ultralytics:latest-jetson

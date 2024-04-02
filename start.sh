ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker run -it --rm --runtime nvidia --shm-size=32G -v $ROOT/datasets:/usr/src/datasets -v $ROOT:/home/root -w /home/root --network host ultralytics/ultralytics:latest-jetson

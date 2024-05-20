#!/bin/bash

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Prevent running as root.
if [[ $(id -u) -eq 0 ]]; then
    print_error "This script cannot be executed with root privileges."
    print_error "Please re-run without sudo and follow instructions to configure docker for non-root user if needed."
    exit 1
fi

# Check if user can run docker without root.
RE="\<docker\>"
if [[ ! $(groups $USER) =~ $RE ]]; then
    print_error "User |$USER| is not a member of the 'docker' group and cannot run docker commands without sudo."
    print_error "Run 'sudo usermod -aG docker \$USER && newgrp docker' to add user to 'docker' group, then re-run this script."
    print_error "See: https://docs.docker.com/engine/install/linux-postinstall/"
    exit 1
fi

# Check if able to run docker commands.
if [[ -z "$(docker ps)" ]] ;  then
    print_error "Unable to run docker commands. If you have recently added |$USER| to 'docker' group, you may need to log out and log back in for it to take effect."
    print_error "Otherwise, please check your Docker installation."
    exit 1
fi

PLATFORM="$(uname -m)"

if [ $PLATFORM = "x86_64" ]; then
    echo "x86"
    docker run -it --rm --runtime nvidia --shm-size=32G -v $ROOT/datasets:/usr/src/datasets -v $ROOT:/home/root -w /home/root --network host ultralytics/ultralytics:latest
else
    echo "jetson"
    docker run -it --rm --runtime nvidia --shm-size=32G -v $ROOT/datasets:/usr/src/datasets -v $ROOT:/home/root -w /home/root --network host ultralytics/ultralytics:latest-jetson
fi
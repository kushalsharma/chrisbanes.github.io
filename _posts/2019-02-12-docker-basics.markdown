---
layout: post
title: Beginners guide to Docker
date: '2019-02-12'
cover_image: /content/images/docker-card.png
---


### Pull the image to local machine 
    docker pull nvcr.io/nvidia/tensorflow:<xx.xx>-py<x>

### Run the image locally and commit installed packages
    nvidia-docker run -it --rm -v [local_dir:container_dir] [nvcr.io/nvidia/tensorflow:<xx.xx>-py<x>]
    pip install [packages]

While the image is running, open a new terminal :

    docker ps
    docker commit [imageId] [nvcr.io/nvidia/tensorflow:<xx.xx>-py<x>-v<>]
    docker images

Exit running image

### Save image as tar file
    docker images
    docker save imageId > [/path/to/file.tar]

### Copy image tar to remote machine
    rsync -avP [/path/to/file.tar] user@host:/path/to/dir

### Load image on remote machine
    docker load < [/path/to/file.tar]
    docker images
    
### Run image on remote machine
    nvidia-docker run -it --rm -v [local_dir:container_dir] [nvcr.io/nvidia/tensorflow:<xx.xx>-py<x>-v<>]

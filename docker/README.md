# Use the container

```bash
$ DOCKER_IMG=username/aiog:dev && docker build -f docker/Dockerfile -t $DOCKER_IMG --build-arg BASE_IMAGE=nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04 .

$ DOCKER_IMG=username/aiog:dev && docker run --rm -it --entrypoint /bin/bash \
    --gpus all \
    --shm-size 96G \
    -v $(pwd):/app \
    $DOCKER_IMG
```

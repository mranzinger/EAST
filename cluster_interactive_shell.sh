#!/bin/sh

submit_job --cpu 10 --gpu 1 \
           --partition 'batch' \
           --mounts "$MOUNTS" \
           --workdir "$SRC_DIR/EAST" \
           --image `cat docker_image` \
           --coolname \
           --interactive \
           -c "bash"

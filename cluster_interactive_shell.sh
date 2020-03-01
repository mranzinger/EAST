#!/bin/sh

submit_job --cpu 80 --gpu 8 \
           --partition 'batch' \
           --mounts "$MOUNTS" \
           --workdir "$SRC_DIR/EAST" \
           --image `cat docker_image` \
           --coolname \
           --interactive \
           -c "bash"

#!/bin/sh

submit_job --cpu 40 --gpu 4 \
           --partition 'batch_32GB' \
           --mounts "$MOUNTS" \
           --workdir "$SRC_DIR/EAST" \
           --image `cat docker_image` \
           --coolname \
           --interactive \
           -c "bash"

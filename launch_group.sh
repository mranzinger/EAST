#!/bin/bash

for i in {1..10}
do
    ./run_job.py python dist_run.py train.py --results $OUTPUT_DIR/east/baseline/run_${i}
    ./run_job.py python dist_run.py train.py --train_dataset /home/dcg-adlr-mranzinger-data.cosmos1100/srnet_synthetic/real_data/real_25/t1_r6_e3_k1_fixed --results $OUTPUT_DIR/east/real_25/t1_r6_e3_k1/run_${i}
done

#run_tensorboard --logdir $OUTPUT_DIR/east

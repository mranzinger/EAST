#!/bin/bash

./run_job.py python dist_run.py train.py --results $OUTPUT_DIR/east/baseline
./run_job.py python dist_run.py train.py --train_dataset /home/dcg-adlr-mranzinger-data.cosmos1100/srnet_synthetic/baseline_d5/t1_r6_e3_k1 --results $OUTPUT_DIR/east/baseline_d5/t1_r6_e3_k1
./run_job.py python dist_run.py train.py --train_dataset /home/dcg-adlr-mranzinger-data.cosmos1100/srnet_synthetic/baseline_d5/t1_r6_e3_k1_p0p2 --results $OUTPUT_DIR/east/baseline_d5/t1_r6_e3_k1_p0p2
./run_job.py python dist_run.py train.py --train_dataset /home/dcg-adlr-mranzinger-data.cosmos1100/srnet_synthetic/baseline_d5/t1_r6_e3_k1_p0p2_procall --results $OUTPUT_DIR/east/baseline_d5/t1_r6_e3_k1_p0p2_procall

#run_tensorboard --logdir $OUTPUT_DIR/east

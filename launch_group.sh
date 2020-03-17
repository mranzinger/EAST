#!/bin/bash

./run_job.py python dist_run.py train.py --train_dataset /home/dcg-adlr-mranzinger-data.cosmos1100/scene-text/icdar/incidental_text/train --epochs 600 --results $OUTPUT_DIR/east/baseline
./run_job.py python dist_run.py train.py --train_dataset /home/dcg-adlr-mranzinger-data.cosmos1100/scene-text/icdar/incidental_text/train --epochs 600 --results $OUTPUT_DIR/east/baseline_o2 --opt 2
./run_job.py python dist_run.py train.py --train_dataset /home/dcg-adlr-mranzinger-data.cosmos1100/srnet_synthetic/new_model/h128_n1/incidental/t1 --epochs 300 --results $OUTPUT_DIR/east/h128_n1 --opt 0
./run_job.py python dist_run.py train.py --train_dataset /home/dcg-adlr-mranzinger-data.cosmos1100/srnet_synthetic/new_model/h128_n1/incidental/t1 --epochs 300 --results $OUTPUT_DIR/east/h128_n1_o2 --opt 2
./run_job.py python dist_run.py train.py --train_dataset /home/dcg-adlr-mranzinger-data.cosmos1100/srnet_synthetic/new_model/h128_n2/incidental/t1 --epochs 300 --results $OUTPUT_DIR/east/h128_n2_o2 --opt 2
./run_job.py python dist_run.py train.py --train_dataset /home/dcg-adlr-mranzinger-data.cosmos1100/srnet_synthetic/new_model/h128_n2/incidental/t1 --epochs 300 --results $OUTPUT_DIR/east/h128_n2 --opt 0
./run_job.py python dist_run.py train.py --train_dataset /home/dcg-adlr-mranzinger-data.cosmos1100/srnet_synthetic/new_model/h128_n4/incidental/t1 --epochs 300 --results $OUTPUT_DIR/east/h128_n4 --opt 0
./run_job.py python dist_run.py train.py --train_dataset /home/dcg-adlr-mranzinger-data.cosmos1100/srnet_synthetic/new_model/h128_n4/incidental/t1 --epochs 300 --results $OUTPUT_DIR/east/h128_n4_o2 --opt 2
run_tensorboard --logdir $OUTPUT_DIR/east

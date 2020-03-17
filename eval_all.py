import argparse
import os

from eval import eval_model
from model import EAST
from utils import resolve_checkpoint_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EAST Multi-Evaluation')
    parser.add_argument('--root', type=str,
                        default='/home/dcg-adlr-mranzinger-output.cosmos1101/east')
    parser.add_argument('--dataset', type=str, default='/home/dcg-adlr-mranzinger-data.cosmos1100/scene-text/icdar/incidental_text/val/images',
                        help='Path to the images to test against')

    args = parser.parse_args()

    model = EAST(False)

    for experiment in sorted(os.listdir(args.root)):
        chk = resolve_checkpoint_path(os.path.join(args.root, experiment, 'checkpoints'), load_best=True)

        print(f'Using checkpoint: {chk}')

        submit_path = './submit'
        eval_model(model, chk, args.dataset, submit_path)

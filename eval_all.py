import argparse
import os

from eval import eval_model
from model import EAST
from utils import resolve_checkpoint_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EAST Multi-Evaluation')
    parser.add_argument('--root', type=str,
                        default='/home/dcg-adlr-mranzinger-output.cosmos1101/east')
    parser.add_argument('--dataset', type=str, default='/home/dcg-adlr-mranzinger-data.cosmos1100/scene-text/icdar/incidental_text/',
                        help='Path to the images to test against')

    args = parser.parse_args()

    model = EAST(False)

    paths = []

    for dirpath, dirnames, filenames in os.walk(args.root):
        for dirname in dirnames:
            if dirname == 'checkpoints':
                experiment = os.path.join(dirpath, dirname)

                try:
                    chk = resolve_checkpoint_path(experiment, load_best=True)
                    paths.append(chk)
                except:
                    pass

    paths.sort()

    for dataset in ['val', 'relabeled_val']:
        dataset = os.path.join(args.dataset, dataset)
        print(f'\n\n----------------------\nDataset: {dataset}')
        for chk in paths:
            print(f'\nUsing checkpoint: {chk}')

            submit_path = './submit'
            eval_model(model, chk, dataset, submit_path)

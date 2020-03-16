# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

"""
Entrypoint utility for launching distributed torch jobs.

NOTE: All command line arguments are forwarded to the child script,
      including those relevant to only this script. If the dependent script
      uses argparse, then you can parse command line arguments using
      `args = parser.parse_known_args()[0]` instead of `args = parser.parse_args()`.

Usage: dist_run --nproc_per_node=<num allocated GPUs> <path to script> <script args>...
"""

import argparse
import importlib
import logging
import logging.config
import os
import psutil
import signal
import sys

import torch
import torch.multiprocessing as mp

from utils import configure_logging


class SignalHandler:
    def __init__(self, child_procs):
        self.child_procs = child_procs

    def __call__(self, incoming_signal, frame):
        print("Signal %d detected in process %d " % ( incoming_signal, os.getpid() ))
        print("Forwarding to children " )
        for child in self.child_procs:
            print("Will now pass the signal %d to child process %d" % ( incoming_signal, child.pid ) )
            os.kill( child.pid, incoming_signal)
        if incoming_signal in [ signal.SIGUSR1,signal.SIGUSR2 ]:
            # This is the most important part - we return silently and will be allowed to keep running.
            return
        else:
            sys.exit(1)


def _set_signal_handlers(child_procs):
    signal_handler = SignalHandler(child_procs)
    print("Setting signal handlers in process %d" % os.getpid())
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGUSR1, signal_handler)
    signal.signal(signal.SIGUSR2, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def _run_function(rank, world_size, script_path):
    """
    Entrypoint for the GPU worker process.

    Args:
        rank (int): The rank of the current worker process.
        world_size (int): The total number of worker processes.
        script_path (str): Path to the python script to execute.
    """
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)

    configure_logging(rank)

    logger = logging.getLogger(__name__)

    # Get the name of the script without the extension
    script_name = os.path.splitext(os.path.basename(script_path))[0]

    script_dir = os.path.dirname(script_path).replace('/', '.')

    logger.info('Loading script "{}" as module "{}" and package "{}".'
                .format(script_path, script_name, script_dir))

    module = importlib.import_module('{}'.format(script_name),
                                     package=script_dir)

    module.main()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Distributed launcher for pytorch.")
    parser.add_argument('-n', '--nproc_per_node', type=int, default=-1,
                        help="The number of processes to launch.")

    # positional
    parser.add_argument("launch_script", type=str,
                        help="The full path to the single GPU module to run.")

    # rest of the arguments for the dest script
    #parser.add_argument('training_script_args', nargs=argparse.REMAINDER)

    args, rest = parser.parse_known_args()

    world_size = args.nproc_per_node
    if world_size == -1:
        world_size = torch.cuda.device_count()

    os.environ['WORLD_SIZE'] = str(world_size)
    # Low level parallel constructs actually hurt the performance fairly significantly
    # because we already employ a high level of parallelism at higher API layers. So,
    # this disables most forms of bad parallelism. FPS improvement for default experiments
    # is somewhere around 100-300 fps just with this.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    script_path = args.launch_script
    sys.argv = [script_path] + rest

    spawn_context = mp.spawn(_run_function, nprocs=world_size, args=(world_size, script_path), join=False)

    _set_signal_handlers(spawn_context.processes)

    # Wait for the child processes to exit
    while not spawn_context.join():
        pass

    sys.exit(0)

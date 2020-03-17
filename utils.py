from datetime import datetime
import logging
import math
import os
import psutil
import sys

from py3nvml.py3nvml import nvmlDeviceGetIndex, nvmlInit
from py3nvml.py3nvml import nvmlShutdown, nvmlSystemGetTopologyGpuSet
import torch
import torchvision.utils as vutils
import cv2
import numpy
import numpy as np


logger = logging.getLogger(__name__)

class IgnorePrint:
    DEV_NULL = open(os.devnull, 'w')

    def __enter__(self):
        self._old_stdout = sys.stdout
        sys.stdout = self.DEV_NULL

    def __exit__(self, *args):
        sys.stdout = self._old_stdout


def set_affinity(rank, log_values=False):
    """
    Sets the CPU affinity for the current process to be local to its respective GPU.

    NOTE: `rank` is considered to be equivalent to the GPU index.

    Certain systems have complex hardware topologies (such as our systems in Cosmos and AVDC),
    and as such, there are certain CPU cores that have faster interconnects with certain
    GPU cards. This function will force linux to only schedule the current process to run on
    CPU cores that are fast for the associated GPU.

    Args:
        rank (int): The rank of the current process (and GPU index).
        log_values (bool): Optionally log the before and after values.
    """
    try:
        num_cpus = os.cpu_count()

        nvmlInit()

        rank_cpus = []
        # nvmlSystemGetTopologyGpuSet prints the number of GPUs each time it's
        # called, so this will suppress those prints
        with IgnorePrint():
            for i in range(num_cpus):
                for d in nvmlSystemGetTopologyGpuSet(i):
                    d_index = nvmlDeviceGetIndex(d)
                    if d_index == rank:
                        rank_cpus.append(i)
                        break

        process = psutil.Process()
        old_affinity = _dense_list_to_spans(process.cpu_affinity())

        process.cpu_affinity(rank_cpus)

        if log_values:
            new_affinity = _dense_list_to_spans(process.cpu_affinity())

            logger.info('Old CPU affinity: {}'.format(old_affinity))
            logger.info('New CPU affinity: {}'.format(new_affinity))

        nvmlShutdown()
    except Exception as e:
        logger.warning("Failed to set the process affinity due to error: {}".format(e))


def configure_logging(rank=0):
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        level=logging.INFO,
        stream=sys.stdout)

    if rank != 0:
        logging.getLogger().handlers.clear()
        logging.getLogger().addHandler(logging.NullHandler())
        #logging.getLogger().setLevel(logging.WARNING)


def _dense_list_to_spans(values):
    """
    Converts a list of integers in increasing order to a span string representing the same set.

    e.g. [0, 1, 5, 6, 7] -> ['0-1', '5-7']

    Args:
        values (list<int>): List of ints in increasing order.

    Returns:
        span (list<str>): List of contiguous spans.
    """
    if not values:
        return []

    first_index = 0
    spans = []

    def _get_span_string(start_index, end_index):
        if start_index == end_index:
            return str(values[start_index])
        return '{}-{}'.format(values[start_index], values[end_index])

    for i in range(1, len(values)):
        prev_value = values[i - 1]
        curr_value = values[i]

        if curr_value != (prev_value + 1):
            spans.append(_get_span_string(first_index, i - 1))
            first_index = i

    spans.append(_get_span_string(first_index, len(values) - 1))

    return spans

def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    torch.cuda.synchronize()

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def clip_grad(model):
    for h in model.parameters():
        if h.grad is not None:
            h.grad.data.clamp_(-0.01, 0.01)

def write_images_to_tensorboard(writer, images, step, prefix='train'):
    """
    Write a dictionary of images to tensorboard
    """
    for image in images:
        name = prefix + '/' + image
        image_grid = vutils.make_grid(images[image].cpu(),
                normalize=True, scale_each=False)
        writer.add_image(name, image_grid, step)


def set_num_threads():
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def mask_resize(mask, scale_factor):
    fh, fw = scale_factor

    # We want to dilate the mask using a kernel size that's nearly
    # as large as the inverse scale factor. This will have the effect of
    # preserving details that are smaller than the downsample window
    inv_fh, inv_fw = int(math.ceil(max(1, 1 / fh))), int(math.ceil(max(1, 1 / fw)))

    kernel = np.ones((inv_fh, inv_fw), np.uint8)

    dil_mask = cv2.dilate(mask, kernel)

    rs_mask = cv2.resize(dil_mask, (0, 0), fx=fw, fy=fh, interpolation=cv2.INTER_CUBIC)

    return rs_mask


class LoaderWorkerProcessInit:
    """Process initialization function for data loader workers."""

    def __init__(self, rank, random_seed):
        """
        Initialization.

        Args:
            rank (int): The rank of the current master process.
            random_seed (int): The global random seed for the experiment.
        """
        self.rank = rank
        self.epoch = 0
        self.random_seed = random_seed

    def set_epoch(self, epoch):
        """Sets the epoch which will be used during the next launch."""
        self.epoch = epoch

    def __call__(self, worker_idx):
        """Configures things such as logging and random seems."""
        configure_logging(self.rank)

        set_num_threads()

        seed = int(1e7) + self.random_seed + int(1e5) * self.rank + 100 * worker_idx + self.epoch
        torch.manual_seed(seed)
        numpy.random.seed(seed)

        set_affinity(self.rank)

def resolve_checkpoint_path(checkpoint_path, load_best=False):
    if not checkpoint_path or os.path.isfile(checkpoint_path):
        return checkpoint_path

    if not os.path.isdir(checkpoint_path):
        raise ValueError(f"The value for '{checkpoint_path}' is not valid!")

    best_path = os.path.join(checkpoint_path, 'best.pth')
    if os.path.isfile(best_path) and load_best:
        return best_path

    best_path = None
    best_epoch = None
    for fname in os.listdir(checkpoint_path):
        try:
            basename, ext = os.path.splitext(fname)
            if ext != '.pth':
                continue
            epoch = int(basename.split('_')[2])
            if best_epoch is None or epoch > best_epoch:
                best_epoch = epoch
                best_path = fname
        except Exception as e:
            logger.warning(f'Invalid checkpoint file "{fname}". Error: {e}')

    if best_path is None:
        raise ValueError(f"No suitable checkpoints found in path: {checkpoint_path}")

    return os.path.join(checkpoint_path, best_path)

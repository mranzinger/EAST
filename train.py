import argparse
import os
import time
import logging
import sys

import numpy as np
import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.optim import lr_scheduler

import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp, optimizers

from dataset import custom_dataset
from model import EAST
from loss import Loss
from fast_data_loader import FastDataLoader
from utils import barrier, LoaderWorkerProcessInit, configure_logging, set_affinity
from meter import MeterDict

DataLoader = FastDataLoader
#DataLoader = data.DataLoader

logger = logging.getLogger('EAST')

global rank
global world_size
rank = 0
world_size = 1


def train(train_img_path, train_gt_path, pths_path, results_path, batch_size,
		  lr, num_workers, epoch_iter, interval,
		  opt_level=0,
		  checkpoint_path=None):
	tensorboard_dir = os.path.join(results_path, 'logs')
	checkpoints_dir = os.path.join(results_path, 'checkpoints')
	if rank == 0:
		os.makedirs(tensorboard_dir, exist_ok=True)
		os.makedirs(checkpoints_dir, exist_ok=True)
	barrier()

	try:
		logger.info('Importing AutoResume lib...')
		from userlib.auto_resume import AutoResume as auto_resume
		auto_resume.init()
		logger.info('Success!')
	except:
		logger.info('Failed!')
		auto_resume = None

	file_num = len(os.listdir(train_img_path))
	trainset = custom_dataset(train_img_path, train_gt_path)

	if world_size > 1:
		train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
	else:
		train_sampler = None

	worker_init = LoaderWorkerProcessInit(rank, 43)
	train_loader = DataLoader(trainset,
							  batch_size=batch_size,
                              shuffle=train_sampler is None,
							  sampler=train_sampler,
							  num_workers=num_workers,
							  pin_memory=True,
							  drop_last=True,
							  worker_init_fn=worker_init)

	criterion = Loss()
	torch.cuda.set_device(rank)
	device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
	model = EAST()
	model.to(device)

	model = apex.parallel.convert_syncbn_model(model)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	model, optimizer = amp.initialize(model, optimizer, opt_level=f'O{opt_level}')

	start_epoch = 0
	if auto_resume is not None:
		auto_resume_details = auto_resume.get_resume_details()
		if auto_resume_details is not None:
			logger.info('Detected that this is a resumption of a previous job!')
			checkpoint_path = auto_resume_details['CHECKPOINT_PATH']

	if checkpoint_path:
		logger.info(f'Loading checkpoint at path "{checkpoint_path}"...')
		checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		amp.load_state_dict(checkpoint['amp_state'])
		start_epoch = checkpoint['epoch']
		logger.info('Done')

	data_parallel = False
	if torch.distributed.is_initialized():
		logger.info(f'DataParallel: Using {torch.cuda.device_count()} devices!')
		model = DDP(model)
		data_parallel = True

	for param_group in optimizer.param_groups:
		param_group.setdefault('initial_lr', lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer,
										 milestones=[epoch_iter // 2],
										 gamma=0.1,
										 last_epoch=start_epoch)

	steps_per_epoch = len(train_loader)
	step = steps_per_epoch * start_epoch
	if rank == 0:
		logger.info('Initializing Tensorboard')
		writer = SummaryWriter(tensorboard_dir, purge_step=step)

	loss_meters = MeterDict(reset_on_value=True)
	time_meters = MeterDict(reset_on_value=True)

	logger.info('Training')
	model.train()

	train_start_time = time.time()

	for epoch in range(start_epoch, epoch_iter):
		if train_sampler is not None:
			train_sampler.set_epoch(epoch)

		epoch_loss = 0
		epoch_time = time.time()
		start_time = time.time()

		for i, batch in enumerate(train_loader):
			optimizer.zero_grad()

			batch = [
				b.cuda(rank, non_blocking=True)
				for b in batch
			]

			img, gt_score, gt_geo, ignored_map = batch
			barrier()
			time_meters['batch_time'].add_sample(time.time() - start_time)

			pred_score, pred_geo = model(img)

			loss, details = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

			epoch_loss += loss.detach().item()

			with amp.scale_loss(loss, optimizer) as loss_scaled:
				loss_scaled.backward()
			optimizer.step()

			barrier()
			time_meters['step_time'].add_sample(time.time() - start_time)

			details['global'] = loss.detach().item()

			for k, v in details.items():
				loss_meters[k].add_sample(v)

			if i % 10 == 0:
				logger.info(f'\tStep [{i+1}/{len(train_loader)}]')

			start_time = time.time()
			step += 1
		scheduler.step()

		term_requested = auto_resume is not None and auto_resume.termination_requested()

		if rank == 0:
			times = { k: m.value() for k, m in time_meters.items() }
			losses = { k: m.value() for k, m in loss_meters.items() }

			times['epoch'] = time.time() - epoch_time

			logger.info(f'Epoch is [{epoch+1}/{epoch_iter}], time consumption is {times}, batch_loss is {losses}')

			for k, v in times.items():
				writer.add_scalar(f'performance/{k}', v, step)
			for k, v in losses.items():
				writer.add_scalar(f'loss/{k}', v, step)
			writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], step)

			if term_requested or (epoch + 1) % interval == 0:
				state_dict = model.module.state_dict() if data_parallel else model.state_dict()
				optim_state = optimizer.state_dict()

				save_path = os.path.join(checkpoints_dir, 'model_epoch_{}.pth'.format(epoch+1))
				logger.info(f'Saving checkpoint to "{save_path}"...')
				torch.save({
					'model': state_dict,
					'optimizer': optim_state,
					'amp_state': amp.state_dict(),
					'epoch': epoch + 1,
				}, save_path)
				logger.info(f'Done')

		if term_requested:
			logger.warning('Termination requested! Exiting...')
			if rank == 0:
				auto_resume.request_resume(
					user_dict={
						'CHECKPOINT_PATH': save_path,
						'EPOCH': epoch
					}
				)
			break

	logger.info(f'Finished training!!! Took {time.time()-train_start_time:0.3f} seconds!')

def resolve_checkpoint_path(checkpoint_path):
	if not checkpoint_path or os.path.isfile(checkpoint_path):
		return checkpoint_path

	if not os.path.isdir(checkpoint_path):
		raise ValueError(f"The value for '{checkpoint_path}' is not valid!")

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
		except:
			logger.warning(f'Invalid checkpoint file "{fname}". Error: {e}')

	if best_path is None:
		raise ValueError(f"No suitable checkpoints found in path: {checkpoint_path}")

	return os.path.join(checkpoint_path, best_path)



def _main():
	torch.backends.cudnn.benchmark = True
	set_affinity(rank, log_values=True)

	parser = argparse.ArgumentParser(description='EAST training!')
	parser.add_argument('-d', '--dataset', type=str, required=True,
						help='The path to the training dataset.')
	parser.add_argument('--results', type=str, required=True,
						help='Where to store the training results.')
	parser.add_argument('--opt', type=int, default=0,
						help='The optimization level to use.')
	parser.add_argument('--chk', type=str, required=False,
						help='Resume training from a previously saved checkpoint.')

	args = parser.parse_args()

	train_img_path = os.path.join(args.dataset, 'images')
	train_gt_path = os.path.join(args.dataset, 'gt')


	# train_img_path = os.path.abspath('../ICDAR_2015/train_img')
	# train_gt_path  = os.path.abspath('../ICDAR_2015/train_gt')
	pths_path      = './pths'
	batch_size     = 24 if world_size == 1 else 12
	lr             = 1e-3 if world_size == 1 else 5e-4 * world_size
	num_workers    = 8
	epoch_iter     = 600
	save_interval  = 5
	train(train_img_path, train_gt_path, pths_path,
		  args.results, batch_size, lr, num_workers, epoch_iter, save_interval,
		  opt_level=args.opt,
		  checkpoint_path=resolve_checkpoint_path(args.chk))

def main():
	"""Entrypoint for the dist_run distributed mode."""
	global rank, world_size
	logger.info('Initializing process group')
	if not torch.distributed.is_initialized():
		torch.distributed.init_process_group(
			backend='nccl',
			group_name='env://',
		)
	rank = torch.distributed.get_rank()
	world_size = torch.distributed.get_world_size()

	_main()


if __name__ == '__main__':
	configure_logging()
	logger.warning("It's recommended to use 'dist_run' to launch this script.")
	logger.info(sys.argv)
	_main()

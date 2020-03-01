import argparse
import os
import time
import logging

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


def train(train_img_path, train_gt_path, pths_path, results_path, batch_size, lr, num_workers, epoch_iter, interval, opt_level):
	tensorboard_dir = os.path.join(results_path, 'logs')
	checkpoints_dir = os.path.join(results_path, 'checkpoints')
	if rank == 0:
		os.makedirs(tensorboard_dir, exist_ok=True)
		os.makedirs(checkpoints_dir, exist_ok=True)
	barrier()

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

	data_parallel = False
	if torch.distributed.is_initialized():
		logger.info(f'DataParallel: Using {torch.cuda.device_count()} devices!')
		model = DDP(model)
		data_parallel = True

	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)

	if rank == 0:
		logger.info('Initializing Tensorboard')
		writer = SummaryWriter(tensorboard_dir, purge_step=0)

	steps_per_epoch = len(train_loader)

	loss_meters = MeterDict(reset_on_value=True)
	time_meters = MeterDict(reset_on_value=True)

	logger.info('Training')
	model.train()

	step = 0
	for epoch in range(epoch_iter):
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
			scheduler.step()

			barrier()
			time_meters['step_time'].add_sample(time.time() - start_time)

			details['global'] = loss.detach().item()

			for k, v in details.items():
				loss_meters[k].add_sample(v)

			if rank == 0 and step % 5 == 0:
				times = { k: m.value() for k, m in time_meters.items() }
				losses = { k: m.value() for k, m in loss_meters.items() }

				logger.info(f'Epoch is [{epoch+1}/{epoch_iter}], mini-batch is [{i+1}/{steps_per_epoch}], time consumption is {times}, batch_loss is {losses}')

				for k, v in times.items():
					writer.add_scalar(f'performance/{k}', v, step)
				for k, v in losses.items():
					writer.add_scalar(f'loss/{k}', v, step)
				writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], step)
			start_time = time.time()
			step += 1

		logger.info('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		logger.info(time.asctime(time.localtime(time.time())))
		logger.info('='*50)
		if rank == 0 and (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(checkpoints_dir, 'model_epoch_{}.pth'.format(epoch+1)))


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

	args = parser.parse_args()

	train_img_path = os.path.join(args.dataset, 'images')
	train_gt_path = os.path.join(args.dataset, 'gt')


	# train_img_path = os.path.abspath('../ICDAR_2015/train_img')
	# train_gt_path  = os.path.abspath('../ICDAR_2015/train_gt')
	pths_path      = './pths'
	batch_size     = 12
	lr             = 5e-4 * world_size
	num_workers    = 8
	epoch_iter     = 600
	save_interval  = 5
	train(train_img_path, train_gt_path, pths_path, args.results, batch_size, lr, num_workers, epoch_iter, save_interval, args.opt)

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

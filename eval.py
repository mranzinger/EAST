import argparse

import time
import torch
import subprocess
import os
from model import EAST
from detect import detect_dataset
import numpy as np
import shutil


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
	if os.path.exists(submit_path):
		shutil.rmtree(submit_path)
	os.mkdir(submit_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(False).to(device)
	model.load_state_dict(torch.load(model_name))
	model.eval()

	start_time = time.time()
	#detect_dataset(model, device, test_img_path, submit_path)
	os.chdir(submit_path)
	res = subprocess.getoutput('zip -q submit.zip *.txt')
	print(res)
	shutil.move('submit.zip', '../submit.zip')
	# res = subprocess.getoutput('mv submit.zip ../')
	os.chdir('../')
	res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
	print(res)
	os.remove('./submit.zip')
	print('eval time is {}'.format(time.time()-start_time))

	if not save_flag:
		shutil.rmtree(submit_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='EAST Evaluation')
	parser.add_argument('--model', type=str, required=True,
						help='Path to the model checkpoint')
	parser.add_argument('--dataset', type=str, default='/home/dcg-adlr-mranzinger-data.cosmos1100/scene-text/icdar/incidental_text/val/images',
						help='Path to the images to test against')

	args = parser.parse_args()

	#model_name = './pths/east_vgg16.pth'
	model_name = args.model
	#test_img_path = os.path.abspath('../ICDAR_2015/test_img')
	submit_path = './submit'
	eval_model(model_name, args.dataset, submit_path)

'''
This codes implements the Perso Re-Identification solution proposed in
'DaliID: Distortion-Adaptive Learned Invariance for Identification - a Robust Technique for Face Recognition and Person Re-Identification' paper
published in IEEE Access
'''
import os
import copy

import torch
import torchreid
import torchvision

from torchvision.models import resnet50, densenet121, inception_v3
from torch.nn import functional as F

from torchvision.transforms import ToTensor, Compose, Normalize, Resize, RandomErasing, RandomHorizontalFlip, ColorJitter
from torchvision.transforms import RandomCrop, functional, GaussianBlur, Grayscale, ToPILImage
from torch.nn import Module, Dropout, BatchNorm1d, Linear, AdaptiveAvgPool2d, CrossEntropyLoss, ReLU, AvgPool2d, AdaptiveMaxPool2d
from torch.nn import functional as F
from torch import nn

import numpy as np
import time
import argparse
import joblib

from Encoders import getDCNN, getEnsembles
from validateModels import validationManager, MSMT17_validator
from datasetUtils import get_dataset_samples_and_statistics, load_dataset
from train_encodersKIT import trainer

from random import shuffle
from termcolor import colored
from collections import defaultdict

from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN, KMeans, OPTICS, cluster_optics_xi
#from GPUClustering import GPUOPTICS, GPUDBSCAN

from getFeatures import extractFeatures, get_subset_one_encoder
from torch.backends import cudnn

from PIL import Image
import matplotlib.pyplot as plt

from tabulate import tabulate

np.random.seed(12)
torch.manual_seed(12)
cudnn.deterministic = True

'''
* Perform learning rate decay
'''



def main(gpu_ids, img_height, img_width, model_name, model_path, base_lr, weight_decay, P, K, tau, beta, lambda_proxy, num_iter, number_of_epoches, 
			momentum_on_feature_extraction, dataset, turbulence_dir_path, is_clean_training, kind_of_transform, dir_to_save, dir_to_save_metrics, version, eval_freq):

	print("Git Branch:", os.system("git branch"))
	print(gpu_ids)
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
	os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

	num_gpus = torch.cuda.device_count()
	print("Num GPU's:", num_gpus)

	gpu_indexes = np.arange(num_gpus).tolist()
	#gpu_indexes = np.array(list(map(int, gpu_ids.split(","))))
	print("Allocated GPU's for model:", gpu_indexes)

	model_online, model_momentum = getDCNN(gpu_indexes, model_name)

	class_number = 0
	train_images_dataset = []
	
	if dataset == "MSMT17":
		train_images_dataset, val_images_dataset, queries_images_dataset, gallery_images_dataset = load_dataset(dataset)
	else:
		train_images_dataset, gallery_images_dataset, queries_images_dataset = load_dataset(dataset)
	
	Nc = len(np.unique(train_images_dataset[:,1]))
	
	validator = validationManager.getValidator(dataset)
	validator.setParameters(img_height, img_width, False, gpu_indexes[0])
	validator.validate(queries_images_dataset, gallery_images_dataset, model_online)

	cmc_progress = []	
	mAP_progress = []
	reliable_data_progress = []
	lr_values = []
	average_across_training = np.array([])

	perc = 1.0
	batch_size = 512
	lr_value = base_lr

	optimizer = torch.optim.Adam(model_online.parameters(), lr=base_lr, weight_decay=weight_decay)
	number_of_iterations = num_iter
	
	total_feature_extraction_time = 0
	total_clustering_time = 0
	total_finetuning_time = 0

	best_cmc = 0
	best_balanced_accuracy = 0.0
	best_iter = 0

	selected_images = train_images_dataset 
	selected_labels = np.int32(train_images_dataset[:,1])
	labels = np.unique(selected_labels)
	num_training_samples = train_images_dataset.shape[0]
	print("Number of training examples: %d" % num_training_samples)

	labels_dict = {labels[idx]: idx for idx in np.arange(len(labels))}

	model_trainer = trainer(dataset, selected_images, model_name, labels_dict, img_height, img_width,
							turbulence_dir_path, is_clean_training, kind_of_transform, optimizer, P, K, tau, beta, lambda_proxy, number_of_epoches, 
																									model_online, model_momentum, gpu_indexes, version)

	if dataset == 'MSMT17':
		validation_validator = MSMT17_validator(train_images_dataset, val_images_dataset, model_trainer, dir_to_save)
		validation_validator.validate(0)


	print("labels dict", labels_dict)

	base_lr_values01 = np.linspace(base_lr, base_lr, num=100)
	base_lr_values02 = np.linspace(base_lr/10, base_lr/10, num=100)
	base_lr_values03 = np.linspace(base_lr/100, base_lr/100, num=50)
	base_lr_values = np.concatenate((base_lr_values01, base_lr_values02, base_lr_values03))

	base_weight_decay_value = np.linspace(weight_decay, weight_decay, num=number_of_epoches)

	t0_pipeline = time.time()
	for pipeline_iter in range(1, number_of_epoches+1):

		t0 = time.time()
		print("###============ Iteration number %d/%d ============###" % (pipeline_iter, number_of_epoches))
		
		lr_value = base_lr_values[pipeline_iter-1]
		weight_decay_value = base_weight_decay_value[pipeline_iter-1]
		lambda_lr_warmup(model_trainer.optimizer, lr_value, weight_decay_value)

		print(colored("Learning Rate: %.5f, Weight Decay: %.7f, Beta value: %.5f, Lambda proxy: %.5f, tau value: %.5f" % (lr_value, weight_decay_value, beta, lambda_proxy, tau), "cyan"))
		
		model_trainer.train(selected_images, selected_labels, number_of_iterations, pipeline_iter)
		
		tf = time.time()
		dt_finetuning = tf - t0
		total_finetuning_time += dt_finetuning
		
		if pipeline_iter % eval_freq == 0:

			if dataset == 'MSMT17':
				validation_validator.validate(pipeline_iter)
				validator.validate(queries_images_dataset, gallery_images_dataset, model_trainer.model_online)
				validator.validate(queries_images_dataset, gallery_images_dataset, model_trainer.model_momentum)

			else:
				cmc, mAP, _ = validator.validate(queries_images_dataset, gallery_images_dataset, model_online)
				validator.validate(queries_images_dataset, gallery_images_dataset, model_momentum)
				
				if cmc[0] > best_cmc:
					best_cmc = cmc[0]
					best_iter = pipeline_iter
					
					torch.save(model_trainer.model_online.state_dict(), "%s/model_online_%s_%s.h5" % (dir_to_save, model_name, version))
					torch.save(model_trainer.model_momentum.state_dict(), "%s/model_momentum_%s_%s.h5" % (dir_to_save, model_name, version))

				cmc_progress.append(cmc)
				mAP_progress.append(mAP)

				joblib.dump(cmc_progress, "%s/CMC_%s_%s" % (dir_to_save_metrics, model_name, version))
				joblib.dump(mAP_progress, "%s/mAP_%s_%s" % (dir_to_save_metrics, model_name, version))

				print("Best R1: %.2f and best iter: %d" % (best_cmc*100, best_iter))


		#np.save("%s/average_max_proxies_distances_across_training_%s_%s.npy" % (dir_to_save_metrics, model_name, version), model_trainer.average_across_training)
		#reliable_data_progress.append(ratio_of_reliable_data)

		#joblib.dump(reliable_data_progress, "%s/reliability_progress_%s_%s" % (dir_to_save_metrics, model_name, version))
		#joblib.dump(lr_values, "%s/lr_progress_%s" % (dir_to_save_metrics, version))

		#joblib.dump(progress_loss, "%s/loss_progress_%s_%s_%s" % (dir_to_save_metrics, "To" + dataset, model_name, version))
		#joblib.dump(number_of_clusters, "%s/number_clusters_%s_%s_%s" % (dir_to_save_metrics, "To" + dataset, model_name, version))

	tf_pipeline = time.time()
	total_pipeline_time = tf_pipeline - t0_pipeline

	mean_feature_extraction_time = total_feature_extraction_time/number_of_epoches
	mean_clustering_time = total_clustering_time/number_of_epoches
	mean_finetuning_time = total_finetuning_time/number_of_epoches

	print(total_feature_extraction_time, total_clustering_time, total_finetuning_time)
	print("Mean Feature Extraction and Reranking Time: %f" % mean_feature_extraction_time)
	print("Mean Clustering Time: %f" % mean_clustering_time)
	print("Mean Finetuning Time: %f" % mean_finetuning_time)
	print("Total pipeline Time:  %f" % total_pipeline_time)


def lambda_lr_warmup(optimizer, lr_value, weight_decay_value):

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr_value
		param_group['weight_decay'] = weight_decay_value

## New Model Definition for ResNet50
class ResNet50ReID(Module):
    
	def __init__(self, model_base):
		super(ResNet50ReID, self).__init__()


		self.conv1 = model_base.conv1
		self.bn1 = model_base.bn1
		self.relu = model_base.relu
		self.maxpool = model_base.maxpool
		self.layer1 = model_base.layer1
		self.layer2 = model_base.layer2
		self.layer3 = model_base.layer3
		self.layer4 = model_base.layer4

		self.layer4[0].conv2.stride = (1,1)
		self.layer4[0].downsample[0].stride = (1,1)

		self.global_avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
		self.global_maxpool = AdaptiveMaxPool2d(output_size=(1, 1))
		self.last_bn = BatchNorm1d(2048)

		self.head_fc01 = Linear(2048, 2048)
		self.head_bn01 = BatchNorm1d(2048)
		self.head_dropout = Dropout(p=0.2)


	
	def forward(self, x, multipart=False, attention_map=False):
		
		x = self.conv1(x)
		x = self.bn1(x)
		#x = self.relu(x) # Do not discomment!
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		#x = x[:,:,8:,:]

		if multipart:
			x_upper = x[:,:,:8,:]
			x_middle = x[:,:,4:12,:]
			x_lower = x[:,:,8:,:]

			# Upper part pooling
			x_avg_upper = self.global_avgpool(x_upper)
			x_max_upper = self.global_maxpool(x_upper)
			#x_upper = x_avg_upper + x_max_upper
			x_upper = x_max_upper
		
			x_upper = x_upper.view(x_upper.size(0), -1)
			output_upper = self.last_bn(x_upper)

			# Middle part pooling
			x_avg_middle = self.global_avgpool(x_middle)
			x_max_middle = self.global_maxpool(x_middle)
			#x_middle = x_avg_middle + x_max_middle
			x_middle = x_max_middle
		
			x_middle = x_middle.view(x_middle.size(0), -1)
			output_middle = self.last_bn(x_middle)

			# Lower part pooling
			x_avg_lower = self.global_avgpool(x_lower)
			x_max_lower = self.global_maxpool(x_lower)
			#x_lower = x_avg_lower + x_max_lower
			x_lower = x_max_lower
		
			x_lower = x_lower.view(x_lower.size(0), -1)
			output_lower = self.last_bn(x_lower)

			# Whole Body pooling
			x_avg = self.global_avgpool(x)
			x_max = self.global_maxpool(x)
			#x = x_avg + x_max
			x = x_max
		
			x = x.view(x.size(0), -1)
			output = self.last_bn(x)

			return output_upper, output_middle, output_lower, output

		if attention_map:
			print(x.shape)
			heatmap = torch.sum(x, dim=1,keepdim=True)[0][0]
			print(heatmap.shape)
			return heatmap


		x_avg = self.global_avgpool(x)
		x_max = self.global_maxpool(x)
		#NSA
		x = x_avg + x_max
		#x = x_max
		x = x.view(x.size(0), -1)

		#output = self.last_bn(x)
		x = self.head_fc01(x)
		x = self.head_bn01(x)
		#x = self.relu(x)
		output = self.head_dropout(x)
		return output


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Define the parameters')
	
	parser.add_argument('--gpu_ids', type=str, default="7", help='GPU IDs')
	parser.add_argument('--img_height', type=int, default=256, help='Image height')
	parser.add_argument('--img_width', type=int, default=128, help='Image width')
	parser.add_argument('--model_name', type=str, help='Backbone name')
	parser.add_argument('--model_path', type=str, help='path_to_the_backbones_pretrained_weights')
	parser.add_argument('--lr', type=float, default=3.5e-4, help='Learning Rate')
	parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
	parser.add_argument('--P', type=int, default=16, help='Number of Persons')
	parser.add_argument('--K', type=int, default=12, help='Number of samples per person')
	parser.add_argument('--tau', type=float, default=0.05, help='tau value used on softmax triplet loss')
	parser.add_argument('--beta', type=float, default=0.999, help='beta used on self-Ensembling')
	parser.add_argument('--lambda_proxy', type=float, default=0.4, help='tuning prameter of Softmax Triplet Loss')
	parser.add_argument('--num_iter', type=int, default=1, help='Number of iterations on an epoch')
	parser.add_argument('--number_of_epoches', type=int, default=250, help='Number of epoches')
	parser.add_argument('--momentum_on_feature_extraction', type=int, default=0, 
																		help='If it is the momentum used on feature extraction')	
	parser.add_argument('--dataset', type=str, help='Name of the dataset')
	parser.add_argument('--turbulence_dir_path', type=str, help='Path to the directory with atmospheric turbulance images')
	parser.add_argument('--is_clean_training', action="store_true", help='define if it is clean training or distortion')
	
	# NSA
	parser.add_argument('--kind_of_transform', type=int, default=1, help='kind of transform: 1 for AT training, 0 for clean training')
	parser.add_argument('--path_to_save_models', type=str, help='Path to save models')
	parser.add_argument('--path_to_save_metrics', type=str, help='Path to save metrics (mAP, CMC, ...)')
	parser.add_argument('--version', type=str, help='Path to save models')
	parser.add_argument('--eval_freq', type=int, help='Evaluation Frequency along training')
	
	args = parser.parse_args()

	gpu_ids = args.gpu_ids
	img_height = args.img_height 
	img_width = args.img_width
	base_lr = args.lr
	weight_decay = args.weight_decay
	model_name = args.model_name
	model_path = args.model_path
	
	P = args.P
	K = args.K
	
	tau = args.tau
	beta = args.beta
	
	lambda_proxy = args.lambda_proxy
	num_iter = args.num_iter
	number_of_epoches = args.number_of_epoches

	momentum_on_feature_extraction = bool(args.momentum_on_feature_extraction)

	dataset = args.dataset
	turbulence_dir_path = args.turbulence_dir_path
	is_clean_training = args.is_clean_training
	kind_of_transform = args.kind_of_transform
	dir_to_save = args.path_to_save_models
	dir_to_save_metrics = args.path_to_save_metrics
	version = args.version
	eval_freq = args.eval_freq

	main(gpu_ids, img_height, img_width, model_name, model_path, base_lr, weight_decay, P, K, tau, beta, lambda_proxy, num_iter, number_of_epoches, 
			momentum_on_feature_extraction, dataset, turbulence_dir_path, is_clean_training, kind_of_transform, dir_to_save, dir_to_save_metrics, version, eval_freq)


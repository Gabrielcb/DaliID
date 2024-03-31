import torch
import torchreid
import torchvision

from torchvision.models import resnet50, densenet121, inception_v3
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, RandomErasing, RandomHorizontalFlip, ColorJitter
from torchvision.transforms import RandomCrop, functional, GaussianBlur, Grayscale, ToPILImage, RandomGrayscale
from torch.nn import Module, Dropout, BatchNorm1d, Linear, AdaptiveAvgPool2d, CrossEntropyLoss, ReLU, AvgPool2d, AdaptiveMaxPool2d, CrossEntropyLoss
from torch.nn import functional as F
from torch import nn

import os
import copy

from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import numpy as np
import time
import argparse
import joblib

import matplotlib.pyplot as plt

from Encoders import getDCNN, getEnsembles
from datasetUtils import get_dataset_samples_and_statistics
from losses import BatchWeightedCenterLoss, BatchWeightedSoftmaxTripletLoss, BatchWeightedSoftmaxAllTripletLoss, BatchWeightedSoftmaxAllCosineLoss
from losses import BatchWeightedProxyLoss, BatchWeightedCrossEntropyLoss, BatchSoftmaxTripletLoss, BatchSoftmaxAllTripletLoss
from losses import BatchCameraHardLoss, distortionLoss

import random
from termcolor import colored
from tqdm import tqdm

from sklearn.metrics import confusion_matrix

from getFeatures import extractFeatures
from torch.backends import cudnn

np.random.seed(12)
torch.manual_seed(12)
cudnn.deterministic = True


class trainer(object):


	def __init__(self, dataset, selected_images, model_name, labels_dict, img_height, img_width,
							turbulance_dir_path, is_clean_training, kind_of_transform, optimizer, P, K, tau, beta, lambda_proxy, number_of_epoches, 
																									model_online, model_momentum, gpu_indexes, version):

		self.dataset = dataset
		self.selected_images = selected_images
		self.model_name = model_name
		self.labels_dict = labels_dict
		self.img_height = img_height
		self.img_width = img_width
		self.turbulance_dir_path = turbulance_dir_path
		self.is_clean_training = is_clean_training
		self.kind_of_transform = kind_of_transform
		self.num_proxies = 5 # You might change that
		self.optimizer = optimizer
		self.P = P 
		self.K = K
		self.tau = tau
		self.beta = beta
		self.lambda_proxy = lambda_proxy
		self.number_of_epoches = number_of_epoches
		self.model_online = model_online
		self.model_momentum = model_momentum
		self.gpu_indexes = gpu_indexes
		self.version = version

	def train(self, selected_images, selected_labels, number_of_iterations, current_epoch):

		
		event_dataset = samplePKBatches(self.dataset, selected_images, selected_labels, self.img_height, self.img_width,
															self.turbulance_dir_path, self.kind_of_transform, K=self.K)

		all_labels = np.unique(selected_labels)
		num_classes = np.unique(all_labels).shape[0]
		batchLoader = DataLoader(event_dataset, batch_size=min(self.P, num_classes), num_workers=8, 
														pin_memory=True, shuffle=True, drop_last=True, collate_fn=collate_fn_PK)


		#breakpoint()
		#batch_images, batch_labels, turbulence_strengths = event_dataset.__getitem__(0)		
		
		keys = list(self.model_online.state_dict().keys())
		size = len(keys) 

		num_batches_computed = 0
		lambda_redundancy = 5e-3
		lambda_bt = 1e-3
		lambda_instance = 0.0
		eps = 1e-7

		lambda_grayscale = 1.0
		lambda_rgb2grayscale = 1.0
		lambda_grayscale2rgb = 1.0
		lambda_distortion = 1.0

		# Selecting proxies for current iteration
		self.model_online.eval()

		undersampled_images = selected_images

		print("Number of samples for proxies generation: %d" % undersampled_images.shape[0])

		undersampled_feature_vectors = extractFeatures(undersampled_images, self.img_height, self.img_width, self.model_online, 500, gpu_index=self.gpu_indexes[0])
		undersampled_images_labels = selected_labels
		
		centers_labels = np.unique(undersampled_images_labels)
		num_classes = len(centers_labels)

		centers = []

		all_proxies = []
		proxies_labels = []

		mean_max_distance = 0.0
		
		for label in centers_labels:

			label_samples = undersampled_feature_vectors[undersampled_images_labels == label]

			proxies_idx, max_dist_between_proxies = selectProxiesByTriagulation(label_samples, num_proxies=5)
			mean_max_distance += max_dist_between_proxies

			all_proxies.append(label_samples[proxies_idx])
			proxies_labels.append(np.array([label]*len(proxies_idx)))

			center = torch.mean(undersampled_feature_vectors[undersampled_images_labels == label], dim=0, keepdim=True)
			centers.append(center)

		
		centers = torch.cat(centers, dim=0)
		centers = centers/torch.norm(centers, dim=1, keepdim=True)
		centers = centers.cuda(self.gpu_indexes[0])

		all_proxies = torch.cat(all_proxies, dim=0)
		all_proxies = all_proxies/torch.norm(all_proxies, dim=1, keepdim=True)
		all_proxies = all_proxies.cuda(self.gpu_indexes[0])

		proxies_labels = np.concatenate(proxies_labels, axis=0)

		# Calculate minimum distance between negative proxies
		proxies_dist = torch.cdist(all_proxies, all_proxies, p=2.0)
		temp_proxies_labels = np.array([proxies_labels])
		temp_labels = temp_proxies_labels.repeat(temp_proxies_labels.shape[1], axis=0)
		mask = torch.Tensor(np.int32(temp_labels == temp_labels.T)).cuda(self.gpu_indexes[0])
		proxies_dist = mask*torch.max(proxies_dist) + (1-mask)*proxies_dist
		min_distance = torch.min(proxies_dist).item()

		mean_max_distance = mean_max_distance/num_classes
		print("Mean Max Proxies Positive Distances: %.3f, Min Negative Distance: %.3f" % (mean_max_distance, min_distance))

		self.model_online.train()
		self.model_momentum.eval()

		for inner_iter in np.arange(number_of_iterations):
		
			print(colored("Iteration number: %d/%d" % (inner_iter+1, number_of_iterations), "green"))

			iteration_center = 0.0
			iteration_proxy_loss = 0.0
			iteration_loss = 0.0

			iteration_weights_sum = 0.0

			total_corrects = 0
			total_batch_size = 0

			number_of_batches_on_epoch = len(batchLoader)

			for batch in tqdm(batchLoader):

				initilized = False
				for imgs, labels, samples_distortion in batch:
			
					if initilized:
						batch_imgs = torch.cat((batch_imgs, imgs), dim=0)
						batch_labels = torch.cat((batch_labels, labels), dim=0)
						batch_distortions = np.concatenate((batch_distortions, samples_distortion), axis=0)
					else:
						batch_imgs = imgs
						batch_labels = labels
						batch_distortions = samples_distortion
						initilized = True

				batch_imgs = batch_imgs.cuda(self.gpu_indexes[0])
				batch_distortions = torch.Tensor(batch_distortions).long().cuda(self.gpu_indexes[0])
			
				if batch_imgs.shape[0] <= 2:
					continue

				batch_fvs_unnorm = self.model_online(batch_imgs)
				batch_fvs = batch_fvs_unnorm/(torch.norm(batch_fvs_unnorm, dim=1, keepdim=True)+1e-9)
			
				batch_center_loss, batch_acc_bal, average_max_prob = BatchWeightedCenterLoss(batch_fvs, batch_labels, 
															batch_distortions, centers, centers_labels, current_epoch, 
															self.number_of_epoches, 0, tau=self.tau, gpu_index=self.gpu_indexes[0])

				batch_proxy_loss = BatchWeightedProxyLoss(batch_fvs, batch_labels, batch_distortions, all_proxies, proxies_labels, current_epoch, 
																self.number_of_epoches, top_negs=50, tau=self.tau, gpu_index=self.gpu_indexes[0])


				batch_loss = batch_center_loss + self.lambda_proxy*batch_proxy_loss

				iteration_center += batch_center_loss.item()
				iteration_proxy_loss += batch_proxy_loss.item()
				iteration_loss += batch_loss.item()
				
				self.optimizer.zero_grad()
				batch_loss.backward()
				self.optimizer.step()

				self.model_online.eval()
				model_online_weights = self.model_online.state_dict()
				model_momentum_weights = self.model_momentum.state_dict()

				for i in range(size):	
					model_momentum_weights[keys[i]] =  self.beta*model_momentum_weights[keys[i]] + (1-self.beta)*model_online_weights[keys[i]].detach()
					
				self.model_momentum.load_state_dict(model_momentum_weights)
				self.model_online.train()
		
				# Decay term
				weights_sum = 0
				for param in self.model_online.parameters():
					weights_sum += param.pow(2).sum().item()

				iteration_weights_sum += weights_sum

				num_batches_computed += 1
		
			iteration_center = iteration_center/number_of_batches_on_epoch
			iteration_proxy_loss = iteration_proxy_loss/number_of_batches_on_epoch
			iteration_loss = iteration_loss/number_of_batches_on_epoch
			iteration_weights_sum = iteration_weights_sum/number_of_batches_on_epoch

			print(colored("Batches computed: %d" % num_batches_computed, "cyan"))
			print(colored("Mean Center Loss: %.7f, Mean Proxy Loss: %.7f" % (iteration_center, iteration_proxy_loss), "yellow"))
			print(colored("Mean Final Loss: %.7f" % iteration_loss, "yellow"))
			print(colored("Mean Weights Sum: %.2f" % iteration_weights_sum, "yellow"))	

									
		self.model_online.eval()
		self.model_momentum.eval()

	
def selectProxiesByTriagulation(X, num_proxies=5):

    dist = torch.cdist(X,X,p=2.0)
    n = dist.shape[0]
    #cumulative_vector = torch.zeros(n)
    cumulative_vector = torch.ones(n)*torch.max(dist)
    proxies = [np.random.choice(n)]

    num_proxies = min(num_proxies, n)

    i = 0
    #while True:
    for j in range(num_proxies-1):
        #previous_proxies = np.unique(proxies)
        sample_idx = proxies[i]
        #print(sample_idx)
        #cumulative_vector += dist[sample_idx]
        cumulative_vector = np.minimum(cumulative_vector, dist[sample_idx])
        #furthest_idx = torch.argmax(cumulative_vector)
        furthest_idx = torch.argsort(cumulative_vector)[-1]
        proxies.append(furthest_idx.item())
        #post_proxies = np.unique(proxies)

        #if len(previous_proxies) == len(post_proxies):
        #  break
        
        i += 1

    proxies = torch.Tensor(proxies).long()
    #proxies_fvs = X[proxies]
    max_dist_between_proxies = torch.max(dist[proxies, :][:, proxies]).item()

    return proxies, max_dist_between_proxies

def collate_fn(batch):
	return torch.cat(batch, dim=0)

def collate_fn_PK(batch):
	return batch

class samplePKBatches(Dataset):
    
	def __init__(self, dataset, images, labels, img_height, img_width, turbulance_dir_path, kind_of_transform, K=4, turb_strength=0):

		self.images_names = images[:,0]
		self.number_of_images = len(self.images_names)
		self.true_ids = images[:,1]
		self.reid_instances = images[:,3]
		self.labels = labels
		self.labels_set = np.unique(labels)
		self.K = K

		np.random.shuffle(self.labels_set)

		self.img_height = img_height
		self.img_width = img_width
		self.dataset = dataset
		self.turbulance_dir_path = turbulance_dir_path
		self.kind_of_transform = kind_of_transform
		self.turb_strength = turb_strength

		self.transform = Compose([Resize((img_height, img_width), interpolation=functional.InterpolationMode.BICUBIC), 
									RandomCrop((img_height, img_width), padding=10), 
									RandomHorizontalFlip(p=0.5), 
									ColorJitter(brightness=0.4, contrast=0.3, saturation=0.4, hue=0.0), 
									#RandomGrayscale(p=0.5),
									ToTensor(),
									RandomErasing(p=1.0, scale=(0.05, 0.30)),
									Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	
	def __getitem__(self, idx):

		pseudo_identity = self.labels_set[idx]
		images_identity = self.images_names[self.labels == pseudo_identity]
		true_identities = self.true_ids[self.labels == pseudo_identity]
		reid_instances = self.reid_instances[self.labels == pseudo_identity]

		selected_images_idx = np.random.choice(images_identity.shape[0], size=min(images_identity.shape[0], self.K), replace=False)
		#selected_images_idx = np.random.choice(images_identity.shape[0], size=self.K, replace=True)

		selected_images = images_identity[selected_images_idx]
		selected_true_identities = true_identities[selected_images_idx]
		selected_reid_instances = reid_instances[selected_images_idx]
		
		final_true_identities = []
		batch_images = []
		samples_distortion = []


		for img_idx in np.arange(len(selected_images)):

			img_name = selected_images[img_idx] 
			imgPIL = torchreid.utils.tools.read_image(img_name)
			reid_inst = selected_reid_instances[img_idx]

			if reid_inst == 'person':
				#augmented_img = torch.stack([transform_person_augmentation(imgPIL)])
				if self.kind_of_transform == 0:
	
					augmented_img = torch.stack([self.transform(imgPIL)])

					final_true_identities.append(selected_true_identities[img_idx])
					samples_distortion.append(0)

					if len(batch_images) == 0:
						batch_images = augmented_img
					else:
						batch_images = torch.cat((batch_images, augmented_img), dim=0)

				# This kind of transform always considered a pair of an image put in the bacth, where 
				# one of the images is the original one and the other is a AT image randomly selected from a 
				# strength value
				elif self.kind_of_transform == 1:

					turb_strength = np.random.choice([1,2,3,4,5])
					# Load atmospheric turbulance simulation
					img_name = selected_images[img_idx].split("/")[-1][:-4]

					if self.dataset == "MSMT17":
						pid_on_path = img_name.split("_")[0]
						img_turbulance_path = os.path.join(self.turbulance_dir_path, pid_on_path + "_" + img_name + "_turbstrength%d.jpg" % turb_strength)
					else:
						img_turbulance_path = os.path.join(self.turbulance_dir_path, img_name + "_turbstrength%d.jpg" % turb_strength)

					imgPIL_AT = torchreid.utils.tools.read_image(img_turbulance_path)

					#imgPIL.save("original.jpg")
					#imgPIL_AT.save("original_TA_BT.jpg")

					img_aug01 = torch.stack([self.transform(imgPIL)])
					img_aug02 = torch.stack([self.transform(imgPIL_AT)])

					final_true_identities.append(selected_true_identities[img_idx])
					final_true_identities.append(selected_true_identities[img_idx])	

					samples_distortion.append(0)
					samples_distortion.append(turb_strength)

					if len(batch_images) == 0:
						batch_images = torch.cat((img_aug01, img_aug02), dim=0)
					else:
						batch_images = torch.cat((batch_images, img_aug01, img_aug02), dim=0)

		batch_labels = torch.ones(batch_images.shape[0])*pseudo_identity
		final_true_identities = np.array(final_true_identities)
		samples_distortion = np.array(samples_distortion, dtype=np.int32)

		return batch_images, batch_labels, samples_distortion
	
	def __len__(self):
		return len(self.labels_set)
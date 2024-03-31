### ============ main74 ============ ###
# Curriculum Learning for Atmospheric Turbulance Person Re-Identification 
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

from torch.utils.data import Dataset, DataLoader

import numpy as np
import time
import argparse
import joblib

from Encoders import getDCNN
from validateModels import validateOnDatasets, validate_with_valSet
from datasetUtils import get_dataset_samples_and_statistics, load_dataset

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

from config import cfg
from make_models import make_model

np.random.seed(12)
torch.manual_seed(12)
cudnn.deterministic = True

'''
* Perform learning rate decay
'''



def main(gpu_ids, img_height, img_width, model_name01, model_path01, model_name02, model_path02, stronger_levels_leave_early, 
        eval_no_heads, eval_weighting, multiple_output, targets, train_file_path, queries_file_path, gallery_file_path, turbulance_dir_path, cfg):

	print("Git Branch:", os.system("git branch"))
	print(gpu_ids)
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
	os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

	num_gpus = torch.cuda.device_count()
	print("Num GPU's:", num_gpus)

	gpu_indexes = np.arange(num_gpus).tolist()
	#gpu_indexes = np.array(list(map(int, gpu_ids.split(","))))
	print("Allocated GPU's for model:", gpu_indexes)

	class_number = 0
	train_images_target = []

	if train_file_path and queries_file_path and gallery_file_path:
		train_images_target = np.load(train_file_path)
		queries_images_target = np.load(queries_file_path)
		gallery_images_target = np.load(gallery_file_path)

		new_ids = np.array([id_name[1:] for id_name in train_images_target[:,1]])
		train_images_target[:,1] = new_ids

		new_ids = np.array([id_name[1:] for id_name in queries_images_target[:,1]])
		queries_images_target[:,1] = new_ids

		new_ids = np.array([id_name[1:] for id_name in gallery_images_target[:,1]])
		gallery_images_target[:,1] = new_ids

		# Removing probes with "range"
		selected_samples_in_probe = []
		for query_dist in queries_images_target[:,3]:
			if "range" in query_dist:
				selected_samples_in_probe.append(False)
			else:
				selected_samples_in_probe.append(True)

		selected_samples_in_probe = np.array(selected_samples_in_probe)
		queries_images_target = queries_images_target[selected_samples_in_probe]		
		print("Probe size after removing close_range images:", queries_images_target.shape)
		

		# Removing ids on probe that are NOT in gallery
		selected_samples_in_probe = np.ones(queries_images_target.shape[0])
		ids_just_in_probe = np.setdiff1d(queries_images_target[:,1], gallery_images_target[:,1])
		print("Number of identities just in probe set: %d" % len(ids_just_in_probe))

		for id_name in ids_just_in_probe:
			id_idx = np.where(queries_images_target[:,1] == id_name)[0]
			selected_samples_in_probe[id_idx] = 0

		selected_samples_in_probe = np.bool_(selected_samples_in_probe)
		queries_images_target = queries_images_target[selected_samples_in_probe]
		print(np.setdiff1d(queries_images_target[:,1], gallery_images_target[:,1]))

		print(train_images_target.shape, queries_images_target.shape, gallery_images_target.shape)

		train_classes = np.unique(train_images_target[:,1])
		nc = len(train_classes)
		print("Number of classes: %d" % nc)

		for tc in train_classes:
			tc_idxes = np.where(train_images_target[:,1] == tc)[0]
			train_images_target[tc_idxes,1] = str(class_number)
			class_number += 1

		train_images_target = np.column_stack((train_images_target, np.array(['BRIAR']*train_images_target.shape[0]))) 
	

	targets_names = targets.split(',')
	if len(targets_names) == 1 and targets != "BRIAR":
		if targets == "MSMT17":
			train_images_target, val_images_target, queries_images_target, gallery_images_target = load_dataset(targets)
		else:
			train_images_target, gallery_images_target, queries_images_target = get_dataset_samples_and_statistics([targets])
	else:
		targets_names = np.setdiff1d(targets.split(','), "BRIAR")
		train_images_target_second, gallery_images_target_second, queries_images_target_second = [], [], []
		
		if len(targets_names) > 0:
			for target in targets_names:
				if target == "MSMT17":
					train_images, val_images, queries_images, gallery_images = load_dataset(target)
					queries_images = [queries_images]
					gallery_images = [gallery_images]
				else:
					train_images, gallery_images, queries_images = get_dataset_samples_and_statistics([target])

				train_classes = np.unique(train_images[:,1])
				nc = len(train_classes)
				print("Number of classes: %d" % nc, class_number)

				classes_idxes = []
				for tc in train_classes:
					tc_idxes = np.where(train_images[:,1] == tc)[0]
					classes_idxes.append(tc_idxes)	

				for tc_idxes in classes_idxes:
					train_images[tc_idxes,1] = str(class_number)
					class_number += 1
					
				train_images = np.column_stack((train_images, np.array(['0']*train_images.shape[0]), np.array([target]*train_images.shape[0]))) 
				train_images_target_second.append(train_images)
				gallery_images_target_second.append(gallery_images[0])
				queries_images_target_second.append(queries_images[0])
				
			train_images_target_second = np.concatenate(train_images_target_second, axis=0)
			
			if len(train_images_target) > 0:
				train_images_target = np.concatenate((train_images_target, train_images_target_second))
			else:
				train_images_target = train_images_target_second

	Nc = len(np.unique(train_images_target[:,1]))
	
    # Loading first model
	if model_name01 == "TransReID":
		# Change camera_num and view_num if using SIE or JIM. We are using baseline for comparison
		camera_num = 0
		view_num = 0 
		num_class = np.unique(train_images_target[:,1]).shape[0]
		print("Number of classes: %d" % num_class)
		model_online = make_model(cfg, num_class, camera_num, view_num)
		model_momentum = make_model(cfg, num_class, camera_num, view_num)

		model_online = nn.DataParallel(model_online, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		model_momentum.load_state_dict(model_online.state_dict())

		model_online = model_online.cuda(gpu_indexes[0])
		model_online = model_online.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	else:
		model01, _ = getDCNN(gpu_indexes, model_name01, is_clean_training=True, stronger_levels_leave_early=stronger_levels_leave_early)

	if model_path01:
		model01.load_state_dict(torch.load(model_path01))
		model01 = model01.cuda(gpu_indexes[0])
		model01 = model01.eval()

    # Loading second model
	if model_name02 == "TransReID":
		# Change camera_num and view_num if using SIE or JIM. We are using baseline for comparison
		camera_num = 0
		view_num = 0 
		num_class = np.unique(train_images_target[:,1]).shape[0]
		print("Number of classes: %d" % num_class)
		model_online = make_model(cfg, num_class, camera_num, view_num)
		model_momentum = make_model(cfg, num_class, camera_num, view_num)

		model_online = nn.DataParallel(model_online, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		model_momentum.load_state_dict(model_online.state_dict())

		model_online = model_online.cuda(gpu_indexes[0])
		model_online = model_online.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	else:
		model02, _ = getDCNN(gpu_indexes, model_name02, is_clean_training=False, stronger_levels_leave_early=stronger_levels_leave_early)

	if model_path02:
		model02.load_state_dict(torch.load(model_path02))
		model02 = model02.cuda(gpu_indexes[0])
		model02 = model02.eval()

		
	if targets == "MSMT17":
		balanced_accuracy = validate_with_valSet(train_images_target, val_images_target, img_height, img_width, model01, 
																							gpu_index=gpu_indexes[0], verbose=1)

		balanced_accuracy = validate_with_valSet(train_images_target, val_images_target, img_height, img_width, model02, 
																							gpu_index=gpu_indexes[0], verbose=1)

		validateOnDatasets(targets_names, [queries_images_target], [gallery_images_target], img_height, img_width, 
																			model_name01, model01, gpu_indexes, multipart=False)
                
		validateOnDatasets(targets_names, [queries_images_target], [gallery_images_target], img_height, img_width, 
																			model_name02, model02, gpu_indexes, multipart=False)
                
	elif "BRIAR" in targets:
		targets_names = ["BRIAR"]
		#validateOnDatasets(["BRIAR"], [queries_images_target], [gallery_images_target], img_height, img_width, 
		#																	model_name, model_online, gpu_indexes, multipart=multipart)
	else:
		validateOnDatasets(targets_names, queries_images_target, gallery_images_target, img_height, img_width, 
																			model_name01, model01, gpu_indexes, multipart=False)
        
		validateOnDatasets(targets_names, queries_images_target, gallery_images_target, img_height, img_width, 
																			model_name02, model02, gpu_indexes, multipart=False)


	if targets == "DeepChange":
		queries = queries_images_target[0][2]
		gallery = gallery_images_target[0][2]
	elif targets == "Market":
		queries = queries_images_target[0]
		gallery = gallery_images_target[0]
	else:
		queries = queries_images_target
		gallery = gallery_images_target

	model01.eval()
	model02.eval()

	# Model 01
	queries_fvs01 = extractFeatures(queries, img_height, img_width, model01, 500, eval_no_heads, eval_weighting, multiple_output, gpu_indexes[0])
	gallery_fvs01 = extractFeatures(gallery, img_height, img_width, model01, 500, eval_no_heads, eval_weighting, multiple_output, gpu_indexes[0])

	queries_fvs01 = queries_fvs01/torch.norm(queries_fvs01, dim=1, keepdim=True)
	gallery_fvs01 = gallery_fvs01/torch.norm(gallery_fvs01, dim=1, keepdim=True)

	distmat01 = 1.0 - torch.mm(queries_fvs01, gallery_fvs01.T)
	distmat01 = distmat01.numpy()

	#if rerank:
	#   print('Applying person re-ranking ...')
	#  distmat_qq = torchreid.metrics.compute_distance_matrix(queries_fvs, queries_fvs, metric="euclidean")
	# distmat_gg = torchreid.metrics.compute_distance_matrix(gallery_fvs, gallery_fvs, metric="euclidean")
	# distmat = torchreid.utils.re_ranking(distmat, distmat_qq, distmat_gg)

	del queries_fvs01, gallery_fvs01
	calculate_metrics(distmat01, queries, gallery)
        
	# Model 02
	queries_fvs02 = extractFeatures(queries, img_height, img_width, model02, 500, eval_no_heads, eval_weighting, multiple_output, gpu_indexes[0])
	gallery_fvs02 = extractFeatures(gallery, img_height, img_width, model02, 500, eval_no_heads, eval_weighting, multiple_output, gpu_indexes[0])

	queries_fvs02 = queries_fvs02/torch.norm(queries_fvs02, dim=1, keepdim=True)
	gallery_fvs02 = gallery_fvs02/torch.norm(gallery_fvs02, dim=1, keepdim=True)

	distmat02 = 1.0 - torch.mm(queries_fvs02, gallery_fvs02.T)
	distmat02 = distmat02.numpy()

	#if rerank:
	#   print('Applying person re-ranking ...')
	#  distmat_qq = torchreid.metrics.compute_distance_matrix(queries_fvs, queries_fvs, metric="euclidean")
	# distmat_gg = torchreid.metrics.compute_distance_matrix(gallery_fvs, gallery_fvs, metric="euclidean")
	# distmat = torchreid.utils.re_ranking(distmat, distmat_qq, distmat_gg)

	del queries_fvs02, gallery_fvs02

	calculate_metrics(distmat02, queries, gallery)
        
	distmat_ensemble = (distmat01+distmat02)/2
	calculate_metrics(distmat_ensemble, queries, gallery)


def calculate_metrics(distmat, queries, gallery):

	# Compute Ranks
	ranks=[1, 5, 10, 20]
	
	print('Computing CMC and mAP ...')

	cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[:,1], gallery[:,1], 
	                            queries[:,2], gallery[:,2], use_metric_cuhk03=False)

	#del distmat
	print('** Results **')
	print('mAP: {:.2%}'.format(mAP))
	print('CMC curve')
	for r in ranks:
		print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))
        


class sample(Dataset):
    
    def __init__(self, Set, img_height, img_width):
        self.set = Set        
        self.transform_person = Compose([Resize((img_height, img_width), interpolation=3), ToTensor(), 
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        
        sample = self.set[idx]
        imgPIL = torchreid.utils.tools.read_image(sample[0])
        img = torch.stack([self.transform_person(imgPIL)])
        return img[0]
                 
    def __len__(self):
        return self.set.shape[0]


def extractFeatures(subset, img_height, img_width, model, batch_size, eval_no_heads, eval_weighting, multiple_output, gpu_index=0, verbose=True):

    dataSubset = sample(subset, img_height, img_width)
    loader = DataLoader(dataSubset, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    start = time.time()
    subset_fvs = []
    subset_head01 = []
    subset_head02 = []
    for batch_idx, batch in enumerate(loader):

        #fvs = getFVs(batch, model, gpu_index)
        with torch.no_grad():
            batch_gpu = batch.cuda(gpu_index)

            if multiple_output:
                fvs_backbone, fvs_head01, fvs_head02 = model(batch_gpu, None, eval_no_heads, eval_weighting, multiple_output)
		
                fvs_backbone = fvs_backbone.data.cpu()
                fvs_head01 = fvs_head01.data.cpu()
                fvs_head02 = fvs_head02.data.cpu()

                subset_fvs.append(fvs_backbone)
                subset_head01.append(fvs_head01)
                subset_head02.append(fvs_head02)
		
            else:
                fv = model(batch_gpu, None, eval_no_heads, eval_weighting)
                fvs = fv.data.cpu()

                if len(subset_fvs) == 0:
                    subset_fvs = fvs
                else:
                    subset_fvs = torch.cat((subset_fvs, fvs), 0)
		    
    if multiple_output:
        subset_fvs = torch.cat(subset_fvs, dim=0)
        subset_head01 = torch.cat(subset_head01, dim=0)
        subset_head02 = torch.cat(subset_head02, dim=0)
            
    end = time.time()
    if verbose:
        print("Features extracted in %.2f seconds" % (end-start))

    if multiple_output:
        return subset_fvs, subset_head01, subset_head02
    else:
        return subset_fvs
    

class libmr:
    def __init__(self, saved_model=None, translateAmount=1):
        self.translateAmount = translateAmount
        if saved_model:
            self.wbFits = torch.zeros(saved_model["Scale"].shape[0], 2)
            self.wbFits[:, 1] = saved_model["Scale"]
            self.wbFits[:, 0] = saved_model["Shape"]
            self.sign = saved_model["signTensor"]
            self.translateAmount = saved_model["translateAmountTensor"]
            self.smallScoreTensor = saved_model["smallScoreTensor"]
        return

    def tocpu(self):
        self.wbFits = self.wbFits.cpu()
        self.smallScoreTensor = self.smallScoreTensor.cpu()

    def return_all_parameters(self):
        return dict(
            Scale=self.wbFits[:, 1],
            Shape=self.wbFits[:, 0],
            signTensor=self.sign,
            translateAmountTensor=self.translateAmount,
            smallScoreTensor=self.smallScoreTensor,
        )

    def FitLow(self, data, tailSize, isSorted=False, gpu=0):
        """
        data --> 5000 weibulls on 0 dim
             --> 10000 distances for each weibull on 1 dim
        """
        self.sign = -1
        max_tailsize_in_1_chunk = 100000
        if tailSize <= max_tailsize_in_1_chunk:
            self.splits = 1
            to_return = self._weibullFitting(data, tailSize, isSorted, gpu)
        else:
            self.splits = tailSize // max_tailsize_in_1_chunk + 1
            to_return = self._weibullFilltingInBatches(data, tailSize, isSorted, gpu)
        return to_return

    def FitHigh(self, data, tailSize, isSorted=False, gpu=0):
        self.sign = 1
        self.splits = 1
        return self._weibullFitting(data, tailSize, isSorted, gpu)

    def compute_weibull_object(self, distances):
        self.deviceName = distances.device
        scale_tensor = self.wbFits[:, 1]
        shape_tensor = self.wbFits[:, 0]
        if self.sign == -1:
            distances = -distances
        if len(distances.shape) == 1:
            distances = distances.repeat(shape_tensor.shape[0], 1)
        smallScoreTensor = self.smallScoreTensor
        if len(self.smallScoreTensor.shape) == 2:
            smallScoreTensor = self.smallScoreTensor[:, 0]
        distances = (
            distances
            + self.translateAmount
            - smallScoreTensor.to(self.deviceName)[None, :]
        )
        weibulls = torch.distributions.weibull.Weibull(
            scale_tensor.to(self.deviceName),
            shape_tensor.to(self.deviceName),
            validate_args=False,
        )
        distances = distances.clamp(min=0)
        return weibulls, distances

    def wscore(self, distances, isReversed=False):
        """
        This function can calculate scores from various weibulls for a given set of distances
        :param distances: a 2-D tensor with the number of rows equal to number of samples and number of columns equal to number of weibulls
        Or
        a 1-D tensor with number of elements equal to number of test samples
        :return:
        """
        weibulls, distances = self.compute_weibull_object(distances)
        if isReversed:
            return 1 - weibulls.cdf(distances)
        else:
            return weibulls.cdf(distances)

    def _weibullFitting(self, dataTensor, tailSize, isSorted=False, gpu=0):
        self.deviceName = dataTensor.device
        if isSorted:
            sortedTensor = dataTensor
        else:
            if self.sign == -1:
                dataTensor = -dataTensor
            sortedTensor = torch.topk(
                dataTensor, tailSize, dim=1, largest=True, sorted=True
            ).values

        smallScoreTensor = sortedTensor[:, tailSize - 1].unsqueeze(1)
        processedTensor = sortedTensor + self.translateAmount - smallScoreTensor
        # Returned in the format [Shape,Scale]
        wbFits = self._fit(processedTensor)
        if self.splits == 1:
            self.wbFits = wbFits
            self.smallScoreTensor = smallScoreTensor
        return wbFits, smallScoreTensor

    def _weibullFilltingInBatches(self, dataTensor, tailSize, isSorted=False, gpu=0):
        N = dataTensor.shape[0]
        dtype = dataTensor.dtype
        batchSize = int(np.ceil(N / self.splits))
        resultTensor = torch.zeros(size=(N, 2), dtype=dtype)
        reultTensor_smallScoreTensor = torch.zeros(size=(N, 1), dtype=dtype)
        for batchIter in range(int(self.splits - 1)):
            startIndex = batchIter * batchSize
            endIndex = startIndex + batchSize
            data_batch = dataTensor[startIndex:endIndex, :].cuda(gpu)
            result_batch, result_batch_smallScoreTensor = self._weibullFitting(
                data_batch, tailSize, isSorted
            )
            resultTensor[startIndex:endIndex, :] = result_batch.cpu()
            reultTensor_smallScoreTensor[
                startIndex:endIndex, :
            ] = result_batch_smallScoreTensor.cpu()

        # process the left-over
        startIndex = (self.splits - 1) * batchSize
        endIndex = N

        data_batch = dataTensor[startIndex:endIndex, :].cuda(gpu)
        result_batch, result_batch_smallScoreTensor = self._weibullFitting(
            data_batch, tailSize, isSorted
        )
        resultTensor[startIndex:endIndex, :] = result_batch.cpu()
        reultTensor_smallScoreTensor[
            startIndex:endIndex, :
        ] = result_batch_smallScoreTensor.cpu()

        self.wbFits = resultTensor
        self.smallScoreTensor = reultTensor_smallScoreTensor

    def _fit(self, data, iters=100, eps=1e-6):
        """
        Adapted from: https://github.com/mlosch/python-weibullfit/blob/0fc6fbe5103c5a2e3ac3374433978f0b816b70be/weibull/backend_pytorch.py#L5
        Adds functionality to fit multiple weibull models in a single tensor using 2D torch tensors.
        Fits multiple 2-parameter Weibull distributions to the given data using maximum-likelihood estimation.
        :param data: 2d-tensor of samples. Each value must satisfy x > 0.
        :param iters: Maximum number of iterations
        :param eps: Stopping criterion. Fit is stopped ff the change within two iterations is smaller than eps.
        :return: tensor with first column Shape, and second Scale these can be (NaN, NaN) if a fit is impossible.
            Impossible fits may be due to 0-values in data.
        """
        k = torch.ones(data.shape[0]).double().to(self.deviceName)
        k_t_1 = k.clone()
        ln_x = torch.log(data)
        computed_params = torch.zeros(data.shape[0], 2).double().to(self.deviceName)
        not_completed = torch.ones(data.shape[0], dtype=torch.bool).to(self.deviceName)
        for t in range(iters):
            if torch.all(torch.logical_not(not_completed)):
                break
            # Partial derivative df/dk
            x_k = data ** torch.transpose(k.repeat(data.shape[1], 1), 0, 1)
            x_k_ln_x = x_k * ln_x
            fg = torch.sum(x_k, dim=1)
            del x_k
            ff = torch.sum(x_k_ln_x, dim=1)
            ff_prime = torch.sum(x_k_ln_x * ln_x, dim=1)
            del x_k_ln_x
            ff_by_fg = ff / fg
            del ff
            f = ff_by_fg - torch.mean(ln_x, dim=1) - (1.0 / k)
            f_prime = (ff_prime / fg - (ff_by_fg ** 2)) + (1.0 / (k * k))
            del ff_prime, fg
            # Newton-Raphson method k = k - f(k;x)/f'(k;x)
            k -= f / f_prime
            computed_params[not_completed * torch.isnan(f), :] = (
                torch.tensor([float("nan"), float("nan")]).double().to(self.deviceName)
            )
            not_completed[abs(k - k_t_1) < eps] = False
            computed_params[torch.logical_not(not_completed), 0] = k[
                torch.logical_not(not_completed)
            ]
            lam = torch.mean(
                data ** torch.transpose(k.repeat(data.shape[1], 1), 0, 1), dim=1
            ) ** (1.0 / k)
            # Lambda (scale) can be calculated directly
            computed_params[torch.logical_not(not_completed), 1] = lam[
                torch.logical_not(not_completed)
            ]
            k_t_1 = k.clone()
        return computed_params  # Shape (SC), Scale (FE)


class Meta_Recognition(object):
    def __init__(self):
        self.mr = libmr()
        
    def metarec(self,scorematrix,topk,use_columns=True,killscale=1):
        if(use_columns):  # if columns contain match scores identies in rows so max row is the expected match
            scores = torch.transpose(scorematrix.clone().detach(),0,1) #make copy as we will destroy stuff to fit weibulls, but then need all for final renorm\n",

            tval, tindex = torch.topk(scores,topk,dim=1)
            kill=torch.zeros_like(scores).scatter_(1,tindex, tval)
            scores =  scores - killscale*kill #doing modified MR--reduce  top scores, but keep something so others are increased if there is a really large gap
            scores = torch.nan_to_num(scores,0) #replace nan's with 0 so does not impact fitting.  No transpose if person = Column
            self.mr.FitHigh(scores, int(scores.shape[1]-topk-1),isSorted=False)  
            scores = self.mr.wscore(scorematrix)  #may need transpose if person in row
            scores = torch.nan_to_num(scores,0) #replace nan's with 0 incase any weibull was unstable and gave 0
            return scores
        else:
            scores = scorematrix.clone().detach() #make copy as we will destroy stuff to fit weibulls, but then need all for final renorm\n",
            tval, tindex = torch.topk(scores,topk,dim=1)
            kill=torch.zeros_like(scores).scatter_(1,tindex, tval)
            scores =  scores - killscale*kill #doing MR--whipe out top scores
            scores = torch.nan_to_num(scores.T,0) #replace nan's with 0 so does not impact fitting.  Transpose as we fit per row
            self.mr.FitHigh(scores, int(scores.shape[1]-topk-1),isSorted=False)  
            scores = self.mr.wscore(scorematrix)          #if we need transpose
            scores = torch.nan_to_num(scores,0) #replace nan's with 0 incase any weibull was unstable and gave 0  
            return scores

    def mrfuse(self, scores01, scores02, scores03):
        
        #print(scores01[:10, :10])
        # apply MR to scores and return them
        wscores01=self.metarec(scores01,20,use_columns=False) #use_columns = true by default
        wscores02=self.metarec(scores02,20,use_columns=False)
        wscores03=self.metarec(scores03,20,use_columns=False)

        print(wscores01[:5, :5])
        print(wscores02[:5, :5])
        print(wscores03[:5, :5])
        #print(scores01)
        #print(scores02)
        #scores03=self.metarec(scores03,1)

        scores = (wscores01*scores01 + wscores02*scores02 + wscores03*scores03)/(wscores01+wscores02+wscores03)
        #scores = (scores01+scores02)/2
        return scores.numpy()
    

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Define the parameters')
	
	parser.add_argument('--gpu_ids', type=str, default="7", help='GPU IDs')
	parser.add_argument('--img_height', type=int, default=256, help='Image height')
	parser.add_argument('--img_width', type=int, default=128, help='Image width')
	parser.add_argument('--model_name01', type=str, help='Backbone name')
	parser.add_argument('--model_name02', type=str, help='Backbone name')
	parser.add_argument('--model_path01', type=str, help='path_to_the_backbones_pretrained_weights')
	parser.add_argument('--model_path02', type=str, help='path_to_the_backbones_pretrained_weights')
	parser.add_argument('--stronger_levels_leave_early', action='store_true', help='Defines if the stronger turbulance levels (4 and 5) leave first. If not, 0 and 1 leave first')
	parser.add_argument('--eval_no_heads', action='store_true', help='Decide if the heads will be considered or not in feature extraction')	
	parser.add_argument('--eval_weighting', action='store_true', help='Decide the heads output will be weighted combined or just one of them selected (hard weighting)')	
	parser.add_argument('--multiple_output', action="store_true", help='define if the output of all exits will be used for evaluation')
	parser.add_argument('--targets', type=str, help='Name of target dataset')
	parser.add_argument('--train_file_path', type=str, help='Path to the npy with training images')
	parser.add_argument('--queries_file_path', type=str, help='Path to the npy with query/probe images')
	parser.add_argument('--gallery_file_path', type=str, help='Path to the npy with gallery images')
	parser.add_argument('--turbulance_dir_path', type=str, help='Path to the directory with atmospheric turbulance images')
	# NSA
	parser.add_argument('--config_file', default="",type=str, help='Path to the config file if TransReID is used')
	parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
	
	args = parser.parse_args()

	gpu_ids = args.gpu_ids
	img_height = args.img_height
	# NSA 
	img_width = args.img_width
	model_name01 = args.model_name01
	model_name02 = args.model_name02
	model_path01 = args.model_path01
	model_path02 = args.model_path02
	stronger_levels_leave_early = args.stronger_levels_leave_early
	eval_no_heads = args.eval_no_heads
	eval_weighting = args.eval_weighting
	multiple_output = args.multiple_output
	
	targets = args.targets
	train_file_path = args.train_file_path
	queries_file_path = args.queries_file_path
	gallery_file_path = args.gallery_file_path
	turbulance_dir_path = args.turbulance_dir_path

	if args.config_file != "":
		cfg.merge_from_file(args.config_file)

	cfg.merge_from_list(args.opts)
    #NNSA
	cfg.freeze()

	main(gpu_ids, img_height, img_width, model_name01, model_path01, model_name02, model_path02, stronger_levels_leave_early, eval_no_heads, eval_weighting, multiple_output, 
																		targets, train_file_path, queries_file_path, gallery_file_path, turbulance_dir_path, cfg)


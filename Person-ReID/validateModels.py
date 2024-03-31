import torch
import torchreid
import torchvision
from torchvision.models import resnet50, densenet121, inception_v3 #, vit_b_16
from torch.nn import Module, Dropout, BatchNorm1d, BatchNorm2d, Linear, AdaptiveAvgPool2d, CrossEntropyLoss, Softmax, ReLU, AdaptiveMaxPool2d, Conv2d
from torch.nn import functional as F
from torch import nn

import numpy as np
from termcolor import colored

from getFeatures import extractFeatures, extractFeaturesMultiView, extractFeaturesDual, extractFeaturesMultiPart

import warnings

try:
    from torchreid.metrics.rank_cylib.rank_cy import evaluate_cy
    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )

class validateModels:
     
    def setParameters(self, img_height, img_width, rerank, gpu_index):
         self.img_height = img_height
         self.img_width = img_width 
         self.rerank = rerank 
         self.gpu_index = gpu_index
         

    def validate(self, queries, gallery, model):

        model.eval()
        queries_fvs = extractFeatures(queries, self.img_height, self.img_width, model, 500, self.gpu_index)
        gallery_fvs = extractFeatures(gallery, self.img_height, self.img_width, model, 500, self.gpu_index)

        queries_fvs = queries_fvs/torch.norm(queries_fvs, dim=1, keepdim=True)
        gallery_fvs = gallery_fvs/torch.norm(gallery_fvs, dim=1, keepdim=True)

        #distmat = torchreid.metrics.compute_distance_matrix(queries_fvs, gallery_fvs, metric="euclidean")
        #distmat = torch.cdist(queries_fvs, gallery_fvs, p=2.0)
        
        distmat = 1.0 - torch.mm(queries_fvs, gallery_fvs.T)
    
        #if rerank:
        #   print('Applying person re-ranking ...')
        #  distmat_qq = torchreid.metrics.compute_distance_matrix(queries_fvs, queries_fvs, metric="euclidean")
        # distmat_gg = torchreid.metrics.compute_distance_matrix(gallery_fvs, gallery_fvs, metric="euclidean")
        # distmat = torchreid.utils.re_ranking(distmat, distmat_qq, distmat_gg)

        del queries_fvs, gallery_fvs
        cmc, mAP = self.calculateMetrics(distmat, queries, gallery)

        return cmc, mAP, distmat

    
    def calculateMetrics(self, distmat, queries, gallery):

        distmat = distmat.numpy()

        #compute Ranks
        ranks = [1,5,10]
        print('Computing CMC and mAP ...')
        cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[:,1], gallery[:,1], 
                                                            queries[:,2], gallery[:,2], use_metric_cuhk03=False)

        print('** Results **')
        print('mAP: {:.2%}'.format(mAP))
        print('Ranks:')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r-1]))
        return cmc, mAP
    

class validateBRIAR(validateModels):

    def __init__(self):
        super(validateModels, self).__init__()
    
    def calculateMetrics(self, distmat, queries, gallery):

        nq = queries.shape[0]
        gt = queries[:,1].reshape(nq, 1)
        cmc = []

        #compute Ranks
        ranks = [1,5,10,20]
        print('Computing CMC and mAP ...')
        ranked_idx = torch.argsort(distmat, dim=1)[:,:20]
        predicted = gallery[:,1][ranked_idx]

        matching = gt == predicted

        print('** Results **')
        print('Ranks:')
        for r in ranks:
            rank_value = np.mean(np.sum(matching[:,:r], axis=1) > 0)
            print('Rank-{:<3}: {:.2%}'.format(r, rank_value))
            cmc.append(rank_value)

        return cmc, 0
    

class validationManager:

    @staticmethod
    def getValidator(dataset_name):
         
        if dataset_name == "BRIAR":
            validator = validateBRIAR()
        else:
            validator = validateModels()

        return validator
    
class MSMT17_validator:

	def __init__(self, train_images, val_images, trainer, dir_to_save):

		self.train_images = train_images
		self.val_images = val_images
		self.img_height = trainer.img_height
		self.img_width  = trainer.img_width
		self.gpu_index = trainer.gpu_indexes[0]
		self.model_name = trainer.model_name 
		self.version = trainer.version
		self.trainer = trainer
		self.best_accuracy = 0.0
		self.best_iter = 0
		self.dir_to_save = dir_to_save

	def validate(self, pipeline_iter):

		balanced_accuracy_online = self.validate_with_valSet(self.trainer.model_online)
		balanced_accuracy_momentum = self.validate_with_valSet(self.trainer.model_momentum)

		if balanced_accuracy_online > self.best_accuracy or balanced_accuracy_momentum > self.best_accuracy:

			if balanced_accuracy_online > balanced_accuracy_momentum:
				self.best_accuracy = balanced_accuracy_online
			else:
				self.best_accuracy = balanced_accuracy_momentum

			self.best_iter = pipeline_iter
		
			torch.save(self.trainer.model_online.state_dict(), "%s/model_online_bestACC_%s_%s.h5" % (self.dir_to_save, self.model_name, self.version))
			torch.save(self.trainer.model_momentum.state_dict(), "%s/model_momentum_bestACC_%s_%s.h5" % (self.dir_to_save, self.model_name, self.version))

		print("Best Balanced Accuracy: {:.2%} and best iter: {}".format(self.best_accuracy, self.best_iter))

	def validate_with_valSet(self, model):

		model.eval()
		selected_fvs = extractFeatures(self.train_images, self.img_height, self.img_width, model, 500, self.gpu_index)
		val_fvs = extractFeatures(self.val_images, self.img_height, self.img_width, model, 500, self.gpu_index)

		selected_fvs = selected_fvs/torch.norm(selected_fvs, dim=1, keepdim=True)
		val_fvs = val_fvs/torch.norm(val_fvs, dim=1, keepdim=True)

		identities_labels = np.int32(self.train_images[:,1])
		labels = np.unique(identities_labels)

		centers = []

		# Selecting reliable samples
		for label in labels:
			center = torch.mean(selected_fvs[identities_labels == label], dim=0, keepdim=True)
			if len(centers) == 0:
				centers = center
			else:
				centers = torch.cat((centers, center), dim=0)
				
		centers = centers/torch.norm(centers, dim=1, keepdim=True)

		S = torch.mm(val_fvs, centers.T)
		highest_similarities, closest_centers_idxes = torch.topk(S, k=5, dim=1, largest=True)
		closest_centers = labels[closest_centers_idxes.numpy()]

		true_matches = np.int32(self.val_images[:,1]) == closest_centers[:,0]

		identities_labels = np.int32(self.val_images[:,1])
		labels = np.unique(identities_labels)

		balanced_acc = 0.0
		for label in labels:
			predictions = true_matches[identities_labels == label]
			TPR = np.sum(predictions)/predictions.shape[0]
			balanced_acc += TPR
		
		balanced_acc = balanced_acc/labels.shape[0]
		print("Balanced Accuracy on Validation Set: {:.3%}".format(balanced_acc))

		return balanced_acc
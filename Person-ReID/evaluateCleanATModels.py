import torch
import torchreid
import torchvision
from torchvision.models import resnet50, densenet121, inception_v3 #, vit_b_16
from torch.nn import Module, Dropout, BatchNorm1d, BatchNorm2d, Linear, AdaptiveAvgPool2d, CrossEntropyLoss, Softmax, ReLU, AdaptiveMaxPool2d, Conv2d
from torch.nn import functional as F
from torch import nn

import numpy as np
from sklearn.metrics import roc_curve
from tabulate import tabulate

from Encoders import getDCNN, getEnsembles
from getFeatures import extractFeatures, extractFeaturesMultiView, extractFeaturesDual, extractFeaturesMultiPart
from datasetUtils import get_dataset_samples_and_statistics, load_dataset, load_general_set
from termcolor import colored

import warnings
import argparse
import os


try:
    from torchreid.metrics.rank_cylib.rank_cy import evaluate_cy
    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )

def main(gpu_ids, img_height, img_width, model_name, model_path_clean, model_path_distortion, dataset, version):

    print("Git Branch:", os.system("git branch"))
    print(gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    num_gpus = torch.cuda.device_count()
    print("Num GPU's:", num_gpus)

    gpu_indexes = np.arange(num_gpus).tolist()
    #gpu_indexes = np.array(list(map(int, gpu_ids.split(","))))
    print("Allocated GPU's for model:", gpu_indexes)

    if dataset == "MSMT17":
        train_images, _, queries_images, gallery_images = load_dataset(dataset)
    else:
        train_images, gallery_images, queries_images = load_dataset(dataset)

    gallery_size = gallery_images.shape[0]
    query_size = queries_images.shape[0]

    num_queries_ids = len(np.unique(queries_images[:,1]))
    num_queries_cameras = len(np.unique(queries_images[:,2]))
    
    num_gallery_ids = len(np.unique(gallery_images[:,1]))
    num_gallery_cameras = len(np.unique(gallery_images[:,2]))

    datasets_descriptions = []
    datasets_descriptions.append([dataset, gallery_size, num_gallery_ids, num_gallery_cameras, 
                                            query_size, num_queries_ids, num_queries_cameras])

    print("Nossa Senhora de Guadalupe")
    print(tabulate(datasets_descriptions, headers=['Dataset', '#Gallery Samples', '#Gallery IDs', '#Gallery Cameras',
                                                                '#Query Samples', '#Query IDs', '#Query Cameras']))

    if model_name == "resnet50":
        model_clean = resnet50(pretrained=True)
        model_clean = ResNet50ReID(model_clean)
        
        model_distortion = resnet50(pretrained=True)
        model_distortion = ResNet50ReID(model_distortion)

    elif model_name == "osnet":
        model_clean = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
        model_clean = OSNETReID(model_clean)

        model_distortion = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
        model_distortion = OSNETReID(model_distortion)
    
    model_clean = nn.DataParallel(model_clean, device_ids=gpu_indexes)
    model_distortion = nn.DataParallel(model_distortion, device_ids=gpu_indexes)
    
    model_clean.load_state_dict(torch.load(model_path_clean))
    model_distortion.load_state_dict(torch.load(model_path_distortion))

    model_clean = model_clean.cuda(gpu_indexes[0])
    model_clean = model_clean.eval()

    model_distortion = model_distortion.cuda(gpu_indexes[0])
    model_distortion = model_distortion.eval()

    #print(model_clean.module.feature)
    
    print(colored("Extraction features with both GAP + GMP for Baseline ...", "yellow"))
    queries_fvs_clean = extractFeatures(queries_images, img_height, img_width, model_clean, 500, gpu_index=gpu_indexes[0])
    queries_fvs_distortion = extractFeatures(queries_images, img_height, img_width, model_distortion, 500, gpu_index=gpu_indexes[0])

    gallery_fvs_clean = extractFeatures(gallery_images, img_height, img_width, model_clean, 500, gpu_index=gpu_indexes[0])
    gallery_fvs_distortion = extractFeatures(gallery_images, img_height, img_width, model_distortion, 500, gpu_index=gpu_indexes[0])

    queries_fvs_concatenated = torch.cat((queries_fvs_clean, queries_fvs_distortion), dim=1)
    gallery_fvs_concatenated = torch.cat((gallery_fvs_clean, gallery_fvs_distortion), dim=1)

    queries_fvs_concatenated = queries_fvs_concatenated/torch.norm(queries_fvs_concatenated, dim=1, keepdim=True)
    gallery_fvs_concatenated = gallery_fvs_concatenated/torch.norm(gallery_fvs_concatenated, dim=1, keepdim=True)

    concatenated_distmat = 1.0 - torch.mm(queries_fvs_concatenated, gallery_fvs_concatenated.T)
    concatenated_distmat = concatenated_distmat.numpy()
    
    print(colored("Obtaining metrics concatenation ...", "yellow"))
    calculateMetrics(queries_images, gallery_images, concatenated_distmat)
    
    queries_fvs_clean = queries_fvs_clean/torch.norm(queries_fvs_clean, dim=1, keepdim=True)
    queries_fvs_distortion = queries_fvs_distortion/torch.norm(queries_fvs_distortion, dim=1, keepdim=True)

    gallery_fvs_clean = gallery_fvs_clean/torch.norm(gallery_fvs_clean, dim=1, keepdim=True)
    gallery_fvs_distortion = gallery_fvs_distortion/torch.norm(gallery_fvs_distortion, dim=1, keepdim=True)

    clean_distmat = 1.0 - torch.mm(queries_fvs_clean, gallery_fvs_clean.T)
    clean_distmat = clean_distmat.numpy()

    distortion_distmat = 1.0 - torch.mm(queries_fvs_distortion, gallery_fvs_distortion.T)
    distortion_distmat = distortion_distmat.numpy()

    distmat_simple_ensemble = (clean_distmat+distortion_distmat)/2

    calculateMetrics(queries_images, gallery_images, clean_distmat)
    calculateMetrics(queries_images, gallery_images, distortion_distmat)

    print(colored("Obtaining metrics for simple ensemble ...", "yellow"))
    calculateMetrics(queries_images, gallery_images, distmat_simple_ensemble)

    print(colored("Obtaining metrics using GAP to get magnitudes and weight features (result provided in the paper) ...", "yellow"))
    queries_fvs_clean_gap_magnitudes, queries_fvs_clean_gap = getWeightsByMagnitude(queries_images, "gap", img_height, img_width, 
                                                                                            model_clean, gpu_indexes)

    queries_fvs_distortion_gap_magnitudes, queries_fvs_distortion_gap = getWeightsByMagnitude(queries_images, "gap", img_height, img_width, 
                                                                                            model_distortion, gpu_indexes)

    gallery_fvs_clean_gap_magnitudes, gallery_fvs_clean_gap = getWeightsByMagnitude(gallery_images, "gap", img_height, img_width, 
                                                                                            model_clean, gpu_indexes)
    gallery_fvs_distortion_gap_magnitudes, gallery_fvs_distortion_gap = getWeightsByMagnitude(gallery_images, "gap", img_height, img_width, 
                                                                                            model_distortion, gpu_indexes)


    #final_queries_fvs = (queries_fvs_clean*queries_fvs_clean_gap_magnitudes + queries_fvs_distortion*queries_fvs_distortion_gap_magnitudes)/(queries_fvs_clean_gap_magnitudes+queries_fvs_distortion_gap_magnitudes)
    #final_gallery_fvs = (gallery_fvs_clean*gallery_fvs_clean_gap_magnitudes + gallery_fvs_distortion*gallery_fvs_distortion_gap_magnitudes)/(gallery_fvs_clean_gap_magnitudes+gallery_fvs_distortion_gap_magnitudes)

    #final_queries_fvs = final_queries_fvs/torch.norm(final_queries_fvs, dim=1, keepdim=True)
    #final_gallery_fvs = final_gallery_fvs/torch.norm(final_gallery_fvs, dim=1, keepdim=True)

    clean_gap_weights = torch.maximum(queries_fvs_clean_gap_magnitudes.repeat(1,gallery_size), gallery_fvs_clean_gap_magnitudes.T.repeat(query_size,1))
    distortion_gap_weights = torch.maximum(queries_fvs_distortion_gap_magnitudes.repeat(1,gallery_size), gallery_fvs_distortion_gap_magnitudes.T.repeat(query_size,1))

    ensembled_distmat = (clean_gap_weights*clean_distmat + distortion_gap_weights*distortion_distmat)/(clean_gap_weights+distortion_gap_weights)

    #calculateMetrics(queries_images, gallery_images, ensembled_distmat, pooling="GAP", version=version)
    calculateMetrics(queries_images, gallery_images, ensembled_distmat)

    '''
    queries_fvs_clean_gap = queries_fvs_clean_gap/torch.norm(queries_fvs_clean_gap, dim=1, keepdim=True)
    gallery_fvs_clean_gap = gallery_fvs_clean_gap/torch.norm(gallery_fvs_clean_gap, dim=1, keepdim=True)

    calculateMetrics(queries_images, gallery_images, queries_fvs_clean_gap, gallery_fvs_clean_gap)

    queries_fvs_distortion_gap = queries_fvs_distortion_gap/torch.norm(queries_fvs_distortion_gap, dim=1, keepdim=True)
    gallery_fvs_distortion_gap = gallery_fvs_distortion_gap/torch.norm(gallery_fvs_distortion_gap, dim=1, keepdim=True)

    calculateMetrics(queries_images, gallery_images, queries_fvs_distortion_gap, gallery_fvs_distortion_gap)
    '''

    print(colored("Obtaining metrics using GMP to get magnitudes and weight features ...", "yellow"))
    queries_fvs_clean_gmp_magnitudes, queries_fvs_clean_gmp = getWeightsByMagnitude(queries_images, "gmp", img_height, img_width, 
                                                                                            model_clean, gpu_indexes)

    queries_fvs_distortion_gmp_magnitudes, queries_fvs_distortion_gmp = getWeightsByMagnitude(queries_images, "gmp", img_height, img_width, 
                                                                                            model_distortion, gpu_indexes)

    gallery_fvs_clean_gmp_magnitudes, gallery_fvs_clean_gmp = getWeightsByMagnitude(gallery_images, "gmp", img_height, img_width, 
                                                                                            model_clean, gpu_indexes)
    gallery_fvs_distortion_gmp_magnitudes, gallery_fvs_distortion_gmp = getWeightsByMagnitude(gallery_images, "gmp", img_height, img_width, 
                                                                                            model_distortion, gpu_indexes)


    #final_queries_fvs = (queries_fvs_clean*queries_fvs_clean_gmp_magnitudes + queries_fvs_distortion*queries_fvs_distortion_gmp_magnitudes)/(queries_fvs_clean_gmp_magnitudes+queries_fvs_distortion_gmp_magnitudes)
    #final_gallery_fvs = (gallery_fvs_clean*gallery_fvs_clean_gmp_magnitudes + gallery_fvs_distortion*gallery_fvs_distortion_gmp_magnitudes)/(gallery_fvs_clean_gmp_magnitudes+gallery_fvs_distortion_gmp_magnitudes)

    #final_queries_fvs = final_queries_fvs/torch.norm(final_queries_fvs, dim=1, keepdim=True)
    #final_gallery_fvs = final_gallery_fvs/torch.norm(final_gallery_fvs, dim=1, keepdim=True)

    clean_gmp_weights = torch.maximum(queries_fvs_clean_gmp_magnitudes.repeat(1,gallery_size), gallery_fvs_clean_gmp_magnitudes.T.repeat(query_size,1))
    distortion_gmp_weights = torch.maximum(queries_fvs_distortion_gmp_magnitudes.repeat(1,gallery_size), gallery_fvs_distortion_gmp_magnitudes.T.repeat(query_size,1))

    ensembled_distmat = (clean_gmp_weights*clean_distmat + distortion_gmp_weights*distortion_distmat)/(clean_gmp_weights+distortion_gmp_weights)
    calculateMetrics(queries_images, gallery_images, ensembled_distmat)
    
    '''
    queries_fvs_clean_gmp = queries_fvs_clean_gmp/torch.norm(queries_fvs_clean_gmp, dim=1, keepdim=True)
    gallery_fvs_clean_gmp = gallery_fvs_clean_gmp/torch.norm(gallery_fvs_clean_gmp, dim=1, keepdim=True)

    calculateMetrics(queries_images, gallery_images, queries_fvs_clean_gmp, gallery_fvs_clean_gmp)

    queries_fvs_distortion_gmp = queries_fvs_distortion_gmp/torch.norm(queries_fvs_distortion_gmp, dim=1, keepdim=True)
    gallery_fvs_distortion_gmp = gallery_fvs_distortion_gmp/torch.norm(gallery_fvs_distortion_gmp, dim=1, keepdim=True)

    calculateMetrics(queries_images, gallery_images, queries_fvs_distortion_gmp, gallery_fvs_distortion_gmp)
    '''

    print(colored("Obtaining metrics using Both (GAP+GMP) to get magnitudes and weight features ...", "yellow"))
    queries_fvs_clean_both_magnitudes, queries_fvs_clean_both = getWeightsByMagnitude(queries_images, "both", img_height, img_width, 
                                                                                            model_clean, gpu_indexes)

    queries_fvs_distortion_both_magnitudes, queries_fvs_distortion_both = getWeightsByMagnitude(queries_images, "both", img_height, img_width, 
                                                                                            model_distortion, gpu_indexes)

    gallery_fvs_clean_both_magnitudes, gallery_fvs_clean_both = getWeightsByMagnitude(gallery_images, "both", img_height, img_width, 
                                                                                            model_clean, gpu_indexes)
    gallery_fvs_distortion_both_magnitudes, gallery_fvs_distortion_both = getWeightsByMagnitude(gallery_images, "both", img_height, img_width, 
                                                                                            model_distortion, gpu_indexes)


    #final_queries_fvs = (queries_fvs_clean*queries_fvs_clean_both_magnitudes + queries_fvs_distortion*queries_fvs_distortion_both_magnitudes)/(queries_fvs_clean_both_magnitudes+queries_fvs_distortion_both_magnitudes)
    #final_gallery_fvs = (gallery_fvs_clean*gallery_fvs_clean_both_magnitudes + gallery_fvs_distortion*gallery_fvs_distortion_both_magnitudes)/(gallery_fvs_clean_both_magnitudes+gallery_fvs_distortion_both_magnitudes)

    #final_queries_fvs = final_queries_fvs/torch.norm(final_queries_fvs, dim=1, keepdim=True)
    #final_gallery_fvs = final_gallery_fvs/torch.norm(final_gallery_fvs, dim=1, keepdim=True)

    clean_both_weights = torch.maximum(queries_fvs_clean_both_magnitudes.repeat(1,gallery_size), gallery_fvs_clean_both_magnitudes.T.repeat(query_size,1))
    distortion_both_weights = torch.maximum(queries_fvs_distortion_both_magnitudes.repeat(1,gallery_size), gallery_fvs_distortion_both_magnitudes.T.repeat(query_size,1))

    ensembled_distmat = (clean_both_weights*clean_distmat + distortion_both_weights*distortion_distmat)/(clean_both_weights+distortion_both_weights)
    calculateMetrics(queries_images, gallery_images, ensembled_distmat)

    '''
    queries_fvs_clean_both = queries_fvs_clean_both/torch.norm(queries_fvs_clean_both, dim=1, keepdim=True)
    gallery_fvs_clean_both = gallery_fvs_clean_both/torch.norm(gallery_fvs_clean_both, dim=1, keepdim=True)

    calculateMetrics(queries_images, gallery_images, queries_fvs_clean_both, gallery_fvs_clean_both)

    queries_fvs_distortion_both = queries_fvs_distortion_both/torch.norm(queries_fvs_distortion_both, dim=1, keepdim=True)
    gallery_fvs_distortion_both = gallery_fvs_distortion_both/torch.norm(gallery_fvs_distortion_both, dim=1, keepdim=True)

    calculateMetrics(queries_images, gallery_images, queries_fvs_distortion_both, gallery_fvs_distortion_both)
    '''


def getWeightsByMagnitude(subset, pooling, img_height, img_width, model, gpu_indexes):

    model.module.feature = pooling
    print(model.module.feature)
    fvs = extractFeatures(subset, img_height, img_width, model, 500, gpu_index=gpu_indexes[0])
    fvs_magnitudes = torch.norm(fvs, dim=1, keepdim=True)
    model.module.feature = "both"
    return fvs_magnitudes, fvs/fvs_magnitudes


def calculateMetrics(queries_images, gallery_images, distmat, pooling=None, version=None):

    # Compute Ranks
    ranks=[1, 5, 10, 20]
    
    print('Computing CMC and mAP ...')

    cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries_images[:,1], gallery_images[:,1], 
                                                queries_images[:,2], gallery_images[:,2], use_metric_cuhk03=False)

    #del distmat
    print('** Results **')
    print('mAP: {:.2%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))

    if pooling:
        num_queries = queries_images.shape[0]
        num_gallery = gallery_images.shape[0]
        queries_ids = queries_images[:,1].reshape(1,num_queries)
        gallery_ids = gallery_images[:,1].reshape(1,num_gallery)
        queries_id_repeated = np.repeat(queries_ids, num_gallery, axis=0).T
        gallery_id_repeated = np.repeat(gallery_ids, num_queries, axis=0)
        print(queries_id_repeated.shape, gallery_id_repeated.shape)

        OneByOneLabels = np.int32(queries_id_repeated == gallery_id_repeated).flatten()
        OneByOnePredictions = 1.0 - distmat.flatten()/2.0

        fpr, tpr, thresholds = roc_curve(OneByOneLabels, OneByOnePredictions, pos_label=1)
        np.save("FPR_%s" % version, fpr)
        np.save("TPR_%s" % version, tpr)
        np.save("Thresholds_%s" % version, thresholds)
        print("ROC Curve calculated!")

        
## New Model Definition for ResNet50
class ResNet50ReID(Module):
    
    def __init__(self, model_base, feature="both"):
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
        self.eps = 1e-6
        self.feature = feature

    
    def forward(self, x, multipart=False, attention_map=False):
        
        x = self.conv1(x)
        x = self.bn1(x)
        #x = self.relu(x) # Do not discomment!
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
    
        x_avg = self.global_avgpool(x)
        x_max = self.global_maxpool(x)

        if self.feature == "gap":
            x = x_avg
        elif self.feature == "gmp":
            x = x_max
        elif self.feature == "both":
            x = x_avg + x_max

        x = x.view(x.size(0), -1)
        output = self.last_bn(x)
        return output


## New Definition for OSNET
class OSNETReID(Module):
    
    def __init__(self, model_base, feature="both"):
        super(OSNETReID, self).__init__()

        self.conv1 = model_base.conv1
        self.maxpool = model_base.maxpool
        self.conv2 = model_base.conv2
        self.conv3 = model_base.conv3
        self.conv4 = model_base.conv4
        self.conv5 = model_base.conv5
        self.avgpool = model_base.global_avgpool
        self.maxpool02 = AdaptiveMaxPool2d(output_size=(1, 1))
        #self.fc = model_base.fc
        self.last_bn = BatchNorm1d(512)
        self.feature = feature


    def forward(self, x):
        
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        v_avg = self.avgpool(x)
        v_max = self.maxpool02(x)

        if self.feature == "gap":
            v = v_avg
        elif self.feature == "gmp":
            v = v_max
        elif self.feature == "both":
            v = v_avg + v_max

        v = v.view(v.size(0), -1)
        output = self.last_bn(v)
        #output = self.fc(v)
        return output


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Define the UDA parameters')
    
    parser.add_argument('--gpu_ids', type=str, default="7", help='GPU IDs')
    parser.add_argument('--img_height', type=int, default=256, help='Image height')
    parser.add_argument('--img_width', type=int, default=128, help='Image width')
    parser.add_argument('--model_name', type=str, help='Backbone name')
    parser.add_argument('--model_path_clean', type=str, help='path_to_the_backbones_pretrained_weights')
    parser.add_argument('--model_path_distortion', type=str, help='path_to_the_backbones_pretrained_weights')    
    parser.add_argument('--dataset', type=str, help='Name of target dataset')
    parser.add_argument('--version', type=str, help='version to save metrics')
    
    args = parser.parse_args()

    gpu_ids = args.gpu_ids
    img_height = args.img_height 
    img_width = args.img_width
    model_name = args.model_name
    model_path_clean = args.model_path_clean
    model_path_distortion = args.model_path_distortion
    dataset = args.dataset
    version = args.version

    main(gpu_ids, img_height, img_width, model_name, model_path_clean, model_path_distortion, dataset, version)
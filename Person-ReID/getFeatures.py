import torch
import time

import torchreid
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize, Resize, Grayscale
from torch.utils.data import Dataset, DataLoader

import os

class sample(Dataset):
    
    def __init__(self, dataset, set_data, turbulance_dir_path, turb_strength, img_height, img_width):
        self.set = set_data 
        self.turbulance_dir_path = turbulance_dir_path
        self.turb_strength = turb_strength
        self.dataset = dataset
       
        self.transform_person = Compose([Resize((img_height, img_width), interpolation=3), ToTensor(), 
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
              
    def __getitem__(self, idx):
        
        sample = self.set[idx]

        if self.turbulance_dir_path:
            # Load atmospheric turbulance simulation
            img_name = sample[0].split("/")[-1][:-4]

            if self.dataset == "MSMT17":
                pid_on_path = img_name.split("_")[0]
                img_turbulance_path = os.path.join(self.turbulance_dir_path, pid_on_path + "_" + img_name + "_turbstrength%d.jpg" % self.turb_strength)
            else:
                img_turbulance_path = os.path.join(self.turbulance_dir_path, img_name + "_turbstrength%d.jpg" % self.turb_strength)

            imgPIL = torchreid.utils.tools.read_image(img_turbulance_path)
        
        else:
            imgPIL = torchreid.utils.tools.read_image(sample[0])

        img = torch.stack([self.transform_person(imgPIL)])
        return img[0]
                 
    def __len__(self):
        return self.set.shape[0]


def extractFeatures(subset, img_height, img_width, model, batch_size, gpu_index=0, dataset=None, turbulance_dir_path=None, turb_strength=None):

    model.eval()
    
    dataSubset = sample(dataset, subset, turbulance_dir_path, turb_strength, img_height, img_width)
    loader = DataLoader(dataSubset, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    start = time.time()
    subset_fvs = []
    for batch_idx, batch in enumerate(loader):

        #fvs = getFVs(batch, model, gpu_index)
        with torch.no_grad():
            batch_gpu = batch.cuda(gpu_index)
            fv = model(batch_gpu)
            fvs = fv.data.cpu()

        if len(subset_fvs) == 0:
            subset_fvs = fvs
        else:
            subset_fvs = torch.cat((subset_fvs, fvs), 0)
            
    end = time.time()
    print("Features extracted in %.2f seconds" % (end-start))
    return subset_fvs


def extractFeatureMaps(subset, img_height, img_width, model, batch_size, gpu_index=0, eval_mode=True, sobel=False, grayscale=False, BGR=False):

    if eval_mode:
        model.eval()
    else:
        model.train()
        
    if sobel:
        dataSubset = sampleSobel(subset)
    else:
        dataSubset = sample(subset, grayscale, img_height, img_width, BGR)

    loader = DataLoader(dataSubset, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    start = time.time()
    subset_fvs = []
    subset_featuremaps = []
    for batch_idx, batch in enumerate(loader):

        with torch.no_grad():
            batch_gpu = batch.cuda(gpu_index)
            feature_map, fv = model(batch_gpu)
            fvs = fv.data.cpu()
            feature_maps = feature_map.data.cpu()

        if len(subset_fvs) == 0:
            subset_fvs = fvs
            subset_featuremaps = feature_maps
        else:
            subset_fvs = torch.cat((subset_fvs, fvs), 0)
            subset_featuremaps = torch.cat((subset_featuremaps, feature_maps), 0)
            
    end = time.time()
    print("Features extracted in %.2f seconds" % (end-start))
    return subset_featuremaps, subset_fvs

def extractFeaturesMultiPart(subset, img_height, img_width, model, batch_size, gpu_index=0, eval_mode=True, sobel=False, grayscale=False, BGR=False):

    if eval_mode:
        model.eval()
    else:
        model.train()
    
    if sobel:
        dataSubset = sampleSobel(subset)
    else:
        dataSubset = sample(subset, grayscale, img_height, img_width, BGR)

    loader = DataLoader(dataSubset, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    start = time.time()

    subset_fvs = []
    subset_fvs_upper = []
    subset_fvs_middle = []
    subset_fvs_lower = []

    for batch_idx, batch in enumerate(loader):

        #fvs = getFVs(batch, model, gpu_index)
        with torch.no_grad():
            batch_gpu = batch.cuda(gpu_index)
            fv_upper, fv_middle, fv_lower, fv = model(x=batch_gpu, multipart=True)
            fvs_upper = fv_upper.data.cpu()
            fvs_middle = fv_middle.data.cpu()
            fvs_lower = fv_lower.data.cpu()
            fvs = fv.data.cpu()

        if len(subset_fvs_upper) == 0:
            subset_fvs_upper = fvs_upper
            subset_fvs_middle = fvs_middle
            subset_fvs_lower = fvs_lower
            subset_fvs = fvs
        else:
            subset_fvs_upper = torch.cat((subset_fvs_upper, fvs_upper), 0)
            subset_fvs_middle = torch.cat((subset_fvs_middle, fvs_middle), 0)
            subset_fvs_lower = torch.cat((subset_fvs_lower, fvs_lower), 0)
            subset_fvs = torch.cat((subset_fvs, fvs), 0)
            
    end = time.time()
    print("Features extracted in %.2f seconds" % (end-start))

    return subset_fvs_upper, subset_fvs_middle, subset_fvs_lower, subset_fvs

def extractFeaturesDual(subset, model, batch_size, gpu_index=0, eval_mode=True):

    if eval_mode:
        model.eval()
    else:
        model.train()
        
    dataSubset = sample(subset)
    loader = DataLoader(dataSubset, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    start = time.time()

    subset_fvs = []
    subset_fvs_id = []
    subset_fvs_bias = []

    for batch_idx, batch in enumerate(loader):

        #fvs = getFVs(batch, model, gpu_index)
        with torch.no_grad():
            batch_gpu = batch.cuda(gpu_index)
            fvs, fvs_id, fvs_bias = model(batch_gpu)
            fvs = fvs.data.cpu()
            fvs_id = fvs_id.data.cpu()
            fvs_bias = fvs_bias.data.cpu()

        if len(subset_fvs_id) == 0:
            subset_fvs = fvs
            subset_fvs_id = fvs_id
            subset_fvs_bias = fvs_bias
        else:
            subset_fvs = torch.cat((subset_fvs, fvs), 0)
            subset_fvs_id = torch.cat((subset_fvs_id, fvs_id), 0)
            subset_fvs_bias = torch.cat((subset_fvs_bias, fvs_bias), 0)
            
    end = time.time()
    print("Features extracted in %.2f seconds" % (end-start))
    #print(subset_fvs, subset_fvs_id.shape, subset_fvs_bias.shape)
    return subset_fvs, subset_fvs_id, subset_fvs_bias

def collate_fn(batch):
    print("Batch", len(batch[0]))
    return batch

def extractFeaturesMultiView(subset, model, batch_size, gpu_index=0, eval_mode=True):

    if eval_mode:
        model.eval()
    else:
        model.train()
        
    dataSubset = sample(subset)
    loader = DataLoader(dataSubset, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    start = time.time()

    subset_global_fvs = []
    subset_spatial_fvs = []
    subset_channel_fvs = []

    for batch_idx, batch in enumerate(loader):

        #fvs = getFVs(batch, model, gpu_index)
        with torch.no_grad():
            batch_gpu = batch.cuda(gpu_index)
            global_fv, spatial_fv, channel_fv = model(batch_gpu)

            global_fvs = global_fv.data.cpu()
            spatial_fvs = spatial_fv.data.cpu()
            channel_fvs = channel_fv.data.cpu()

        if len(subset_global_fvs) == 0:
            subset_global_fvs = global_fvs
            subset_spatial_fvs = spatial_fvs
            subset_channel_fvs = channel_fvs
        else:
            subset_global_fvs = torch.cat((subset_global_fvs, global_fvs), 0)
            subset_spatial_fvs = torch.cat((subset_spatial_fvs, spatial_fvs), 0)
            subset_channel_fvs = torch.cat((subset_channel_fvs, channel_fvs), 0)
            
    end = time.time()
    print("Features extracted in %.2f seconds" % (end-start))

    return subset_global_fvs, subset_spatial_fvs, subset_channel_fvs

def get_subset(selected_sample, train_set, perc_closest, encoder01, encoder02, encoder03, batch_size=500, gpu_index=0):

    imgPIL = torchreid.utils.tools.read_image(selected_sample[0])
    reid_instance = selected_sample[3]

    if reid_instance == "person":    
        img = torch.stack([transform_person(imgPIL)])
    elif reid_instance == "object":
        img = torch.stack([transform_vehicle(imgPIL)])

    encoder01.eval()
    encoder02.eval()
    encoder03.eval()

    with torch.no_grad():
        selected_feature01 = encoder01(img.cuda(gpu_index)).data.cpu()
        selected_feature02 = encoder02(img.cuda(gpu_index)).data.cpu()
        selected_feature03 = encoder03(img.cuda(gpu_index)).data.cpu()

    selected_feature01 = selected_feature01/torch.norm(selected_feature01, dim=1, keepdim=True)
    selected_feature02 = selected_feature02/torch.norm(selected_feature02, dim=1, keepdim=True)
    selected_feature03 = selected_feature03/torch.norm(selected_feature03, dim=1, keepdim=True)

    dataset = sample(train_set)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    start = time.time()
    total_similarities = []

    for batch_idx, batch in enumerate(loader):

        with torch.no_grad():
            batch_gpu = batch.cuda(gpu_index)

            batch_fvs01 = encoder01(batch_gpu).data.cpu()
            batch_fvs02 = encoder02(batch_gpu).data.cpu()
            batch_fvs03 = encoder03(batch_gpu).data.cpu()
        
        batch_fvs01 = batch_fvs01/torch.norm(batch_fvs01, dim=1, keepdim=True)
        batch_fvs02 = batch_fvs02/torch.norm(batch_fvs02, dim=1, keepdim=True)
        batch_fvs03 = batch_fvs03/torch.norm(batch_fvs03, dim=1, keepdim=True)

        sim01 = torch.mm(selected_feature01, batch_fvs01.T)
        sim02 = torch.mm(selected_feature02, batch_fvs02.T)
        sim03 = torch.mm(selected_feature03, batch_fvs03.T)

        res_sim = (sim01 + sim02 + sim03)/3

        if len(total_similarities) > 0:
            total_similarities = torch.cat((total_similarities, res_sim), dim=1)
        else:
            total_similarities = res_sim


    sorted_idx = torch.argsort(total_similarities, dim=1, descending=True)[0]
    topK = int(len(train_set)*perc_closest)
    subset = train_set[sorted_idx[:topK]] 
    
    end = time.time()
    print("Subset calculated in %.2f seconds" % (end-start))

    return subset

def get_subset_one_encoder(selected_sample, train_set, topK, encoder, batch_size=500, gpu_index=0):

    imgPIL = torchreid.utils.tools.read_image(selected_sample[0])
    reid_instance = selected_sample[3]

    if reid_instance == "person":    
        img = torch.stack([transform_person(imgPIL)])
    elif reid_instance == "object":
        img = torch.stack([transform_vehicle(imgPIL)])

    encoder.eval()
    with torch.no_grad():
        selected_feature = encoder(img.cuda(gpu_index)).data.cpu()

    
    selected_feature = selected_feature/torch.norm(selected_feature, dim=1, keepdim=True)
   
    dataset = sample(train_set)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    start = time.time()
    total_similarities = []

    for batch_idx, batch in enumerate(loader):

        with torch.no_grad():
            batch_gpu = batch.cuda(gpu_index)
            batch_fvs = encoder(batch_gpu).data.cpu()
            
        
        batch_fvs = batch_fvs/torch.norm(batch_fvs, dim=1, keepdim=True)
        sim = torch.mm(selected_feature, batch_fvs.T)
    
        if len(total_similarities) > 0:
            total_similarities = torch.cat((total_similarities, sim), dim=1)
        else:
            total_similarities = sim

    
    sorted_idx = torch.argsort(total_similarities, dim=1, descending=True)[0]
    #topK = int(len(train_set)*perc_closest)
    selected_indexes = sorted_idx[:topK]
    non_selected_indexes = sorted_idx[topK:]
    #subset = train_set[sorted_idx[:topK]] 
    end = time.time()
    print("Subset calculated in %.2f seconds" % (end-start))

    return selected_indexes, non_selected_indexes 


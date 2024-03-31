import torch
import torchreid
import torchvision
from torchvision.models import resnet50, densenet121, inception_v3, vit_b_16, efficientnet_b0
from torch.nn import Module, Dropout, BatchNorm1d, BatchNorm2d, Linear, AdaptiveAvgPool2d, CrossEntropyLoss, Softmax, ReLU, AdaptiveMaxPool2d, Conv2d
from torch.nn import functional as F
from torch import nn

import numpy as np
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

# VISION TRANSFORMER IS USING ONLY PERSON IMAGE DIMENSION! TO APPLY IT TO VEHICLES OR OTHER OBJECTS, YOU MUST 
# CHANGE THE IMAGE DIMENSIONS FOR 224 X 224 OR CHANGE THE CODE TO ACCEPT ARBITRARY DIMENSIONS (only on function calling)

def getDCNN(gpu_indexes, model_name, embedding_size=None):

	if model_name == "resnet50":

		if embedding_size is None:
			embedding_size = 2048

		# loading ResNet50
		model_source = resnet50(pretrained=True)
		model_source = ResNet50ReID(model_source)
		
		model_momentum = resnet50(pretrained=True)
		model_momentum = ResNet50ReID(model_momentum)
		
		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)
		
		model_momentum.load_state_dict(model_source.state_dict())
		
		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()
		
		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	elif model_name == "resnet50Seg":

		if embedding_size is None:
			embedding_size = 2048

		# loading ResNet50
		model_source = resnet50(pretrained=True)
		model_source = ResNet50SegReID(model_source)
		print(1)
		model_momentum = resnet50(pretrained=True)
		model_momentum = ResNet50SegReID(model_momentum)
		print(2)
		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)
		print(3)
		model_momentum.load_state_dict(model_source.state_dict())
		print(4)
		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()
		print(5)
		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	elif model_name == "resnet50IBN":

		if embedding_size is None:
			embedding_size = 2048

		# loading ResNet50
		model_source = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
		model_source = ResNet50IBNReID(model_source)
		
		model_momentum = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
		model_momentum = ResNet50IBNReID(model_momentum)
		
		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		#model_source = convert_model(model_source)
		#model_momentum = convert_model(model_momentum)
		
		model_momentum.load_state_dict(model_source.state_dict())
		
		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()
		
		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	elif model_name == "resnet101IBN":

		if embedding_size is None:
			embedding_size = 2048

		# loading ResNet50
		model_source = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=True)
		model_source = ResNet101IBNReID(model_source)
		
		model_momentum = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=True)
		model_momentum = ResNet101IBNReID(model_momentum)
		
		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)
		
		#model_source = convert_model(model_source)
		#model_momentum = convert_model(model_momentum)

		model_momentum.load_state_dict(model_source.state_dict())
		
		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()
		
		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	elif model_name == "osnet":

		if embedding_size is None:
			embedding_size = 512

		# loading OSNet	
		model_source = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
		model_source = OSNETReID(model_source)

		model_momentum = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
		model_momentum = OSNETReID(model_momentum)

		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		model_momentum.load_state_dict(model_source.state_dict())

		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	elif model_name == "densenet121":

		if embedding_size is None:
			embedding_size = 2048

		# loading DenseNet121
		model_source = densenet121(pretrained=True)
		model_source = DenseNet121ReID(model_source)

		model_momentum = densenet121(pretrained=True)
		model_momentum = DenseNet121ReID(model_momentum)

		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		model_momentum.load_state_dict(model_source.state_dict())

		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	elif model_name == "inceptionV3":

		if embedding_size is None:
			embedding_size = 2048

		# Loading inceptionV3
		model_source = inception_v3(pretrained=True)
		model_source = inceptionV3ReID(model_source)

		model_momentum = inception_v3(pretrained=True)
		model_momentum = inceptionV3ReID(model_momentum)

		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		model_momentum.load_state_dict(model_source.state_dict())

		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	elif model_name == "ViT":

		if embedding_size is None:
			embedding_size = 2048

		# Loading inceptionV3
		model_source = vit_b_16(pretrained=True)
		model_source = ViTReID(model_source, 224, 224)

		model_momentum = vit_b_16(pretrained=True)
		model_momentum = ViTReID(model_momentum, 224, 224)

		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		model_momentum.load_state_dict(model_source.state_dict())

		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()


	elif model_name == "efficientnetB0":

		if embedding_size is None:
			embedding_size = 2048

		# Loading inceptionV3
		model_source = efficientnet_b0(pretrained=True)
		model_source = efficientnetB0ReID(model_source)

		model_momentum = efficientnet_b0(pretrained=True)
		model_momentum = efficientnetB0ReID(model_momentum)

		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		model_momentum.load_state_dict(model_source.state_dict())

		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	return model_source, model_momentum



def getEnsembles(gpu_indexes):

	# loading ResNet50
	model_source_resnet50 = resnet50(pretrained=True)
	model_source_resnet50 = ResNet50ReID(model_source_resnet50)

	model_momentum_resnet50 = resnet50(pretrained=True)
	model_momentum_resnet50 = ResNet50ReID(model_momentum_resnet50)

	model_source_resnet50 = nn.DataParallel(model_source_resnet50, device_ids=gpu_indexes)
	model_momentum_resnet50 = nn.DataParallel(model_momentum_resnet50, device_ids=gpu_indexes)

	model_momentum_resnet50.load_state_dict(model_source_resnet50.state_dict())

	model_source_resnet50 = model_source_resnet50.cuda(gpu_indexes[0])
	model_source_resnet50 = model_source_resnet50.eval()

	model_momentum_resnet50 = model_momentum_resnet50.cuda(gpu_indexes[0])
	model_momentum_resnet50 = model_momentum_resnet50.eval()

	# loading OSNet	
	model_source_osnet = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
	model_source_osnet = OSNETReID(model_source_osnet)

	model_momentum_osnet = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
	model_momentum_osnet = OSNETReID(model_momentum_osnet)

	model_source_osnet = nn.DataParallel(model_source_osnet, device_ids=gpu_indexes)
	model_momentum_osnet = nn.DataParallel(model_momentum_osnet, device_ids=gpu_indexes)

	model_momentum_osnet.load_state_dict(model_source_osnet.state_dict())

	model_source_osnet = model_source_osnet.cuda(gpu_indexes[0])
	model_source_osnet = model_source_osnet.eval()

	model_momentum_osnet = model_momentum_osnet.cuda(gpu_indexes[0])
	model_momentum_osnet = model_momentum_osnet.eval()

	# loading DenseNet121
	model_source_densenet121 = densenet121(pretrained=True)
	model_source_densenet121 = DenseNet121ReID(model_source_densenet121)

	model_momentum_densenet121 = densenet121(pretrained=True)
	model_momentum_densenet121 = DenseNet121ReID(model_momentum_densenet121)

	model_source_densenet121 = nn.DataParallel(model_source_densenet121, device_ids=gpu_indexes)
	model_momentum_densenet121 = nn.DataParallel(model_momentum_densenet121, device_ids=gpu_indexes)

	model_momentum_densenet121.load_state_dict(model_source_densenet121.state_dict())

	model_source_densenet121 = model_source_densenet121.cuda(gpu_indexes[0])
	model_source_densenet121 = model_source_densenet121.eval()

	model_momentum_densenet121 = model_momentum_densenet121.cuda(gpu_indexes[0])
	model_momentum_densenet121 = model_momentum_densenet121.eval()

	return model_source_resnet50, model_momentum_resnet50, model_source_osnet, model_momentum_osnet, model_source_densenet121, model_momentum_densenet121
	


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
		
		
	def forward(self, x):
		
		x = self.conv1(x)
		x = self.bn1(x)
		#x = self.relu(x) # Do not discomment!
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		
		x_avg = self.global_avgpool(x)
		#x_avg = torch.mean(x, dim=(2,3), keepdim=True)
		#x_std = torch.std(x, dim=(2,3), keepdim=True)
		x_max = self.global_maxpool(x)
		x = x_avg + x_max
		#c = x_max
		
		x = x.view(x.size(0), -1)
		
		output = self.last_bn(x)
		return output
		
		
			
## New Model Definition for ResNet50
class ResNet50SegReID(Module):
    
	def __init__(self, model_base):
		super(ResNet50SegReID, self).__init__()


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

	
	def forward(self, x, seg_mask=None):
		
		x = self.conv1(x)
		x = self.bn1(x)
		#x = self.relu(x) # Do not discomment!
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		# attention module
		if seg_mask is not None:
			x = x*seg_mask

		x_avg = self.global_avgpool(x)
		x_max = self.global_maxpool(x)
		x = x_avg + x_max
		#x = x_max
		x = x.view(x.size(0), -1)

		output = self.last_bn(x)
		return output

## New Model Definition for ResNet50 with two outputs
class DualResNet50ReID(Module):
    
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

		self.last_bn = BatchNorm2d(2048)
		self.id_bn = BatchNorm1d(2048)
		self.bias_bn = BatchNorm1d(2048)

		self.id_conv1x1 = Conv2d(2048, 2048, kernel_size=1, stride=1)
		self.bias_conv1x1 = Conv2d(2048, 2048, kernel_size=1, stride=1)

	
	def forward(self, x):
		
		x = self.conv1(x)
		x = self.bn1(x)
		#x = self.relu(x) # Do not discomment!
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x_id = self.id_conv1x1(x)
		x_bias = self.bias_conv1x1(x)

		x_id_avg = self.global_avgpool(x_id)
		x_id_max = self.global_maxpool(x_id)
		x_id = x_id_avg + x_id_max
		x_id = x_id.view(x_id.size(0), -1)
		output_id = self.id_bn(x_id)

		x_bias_avg = self.global_avgpool(x_bias)
		x_bias_max = self.global_maxpool(x_bias)
		x_bias = x_bias_avg + x_bias_max
		x_bias = x_bias.view(x_bias.size(0), -1)
		output_bias = self.bias_bn(x_bias)

		return output_id, output_bias
	

class ResNet50IBNReID(Module):
    
	def __init__(self, model_base):
		super(ResNet50IBNReID, self).__init__()


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

		self.conv1x1_layer4 = torch.nn.Conv2d(2048, 1, (1,1), stride=1)
		self.conv1x1_squeeze_layer4 = torch.nn.Conv2d(4096, 1024, (1,1), stride=1)
		self.conv1x1_expand_layer4 = torch.nn.Conv2d(1024, 2048, (1,1), stride=1)

		self.attribute_classifier = torch.nn.Linear(2048, 88, bias=True)
		
		self.last_bn = BatchNorm1d(2048)

	
	def forward(self, x, multipart=False, attention_map=False):
		
		x = self.conv1(x)
		x = self.bn1(x)
		#x = self.relu(x) # Do not discomment!
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		#x = self.spatial_channel_attention(x, "layer4")

		x_avg = self.global_avgpool(x)
		x_max = self.global_maxpool(x)
		x = x_avg + x_max
		
		x = x.view(x.size(0), -1)
		
		output = self.last_bn(x)

		#attribute_prediction = torch.sigmoid(self.attribute_classifier(output))

		return output #, attribute_prediction

	def spatial_channel_attention(self, x, layer_name):

		if layer_name == "layer4":
			space_attention = torch.sigmoid(self.conv1x1_layer4(x))

			x_avg = self.global_avgpool(x)
			x_max = self.global_maxpool(x)
			gp = torch.cat((x_avg, x_max), dim=1)

			squeezed_fv = torch.nn.ReLU(inplace=True)(self.conv1x1_squeeze_layer4(gp))
			channel_attention = torch.sigmoid(self.conv1x1_expand_layer4(squeezed_fv))

		attention_x = x*space_attention + x*channel_attention + x

		return attention_x


class ResNet101IBNReID(Module):
    
	def __init__(self, model_base):
		super(ResNet101IBNReID, self).__init__()


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

		self.conv1x1_layer4 = torch.nn.Conv2d(2048, 1, (1,1), stride=1)
		self.conv1x1_squeeze_layer4 = torch.nn.Conv2d(4096, 1024, (1,1), stride=1)
		self.conv1x1_expand_layer4 = torch.nn.Conv2d(1024, 2048, (1,1), stride=1)

		self.attribute_classifier = torch.nn.Linear(2048, 88, bias=True)

		self.last_bn = BatchNorm1d(2048)

	
	def forward(self, x, multipart=False, attention_map=False):
		
		x = self.conv1(x)
		x = self.bn1(x)
		#x = self.relu(x) # Do not discomment!
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		#x = self.spatial_channel_attention(x, "layer4")
		
		x_avg = self.global_avgpool(x)
		x_max = self.global_maxpool(x)
		x = x_avg + x_max
		
		x = x.view(x.size(0), -1)
		
		output = self.last_bn(x)

		#attribute_prediction = torch.sigmoid(self.attribute_classifier(output))

		return output #, attribute_prediction

	def spatial_channel_attention(self, x, layer_name):

		if layer_name == "layer4":
			space_attention = torch.sigmoid(self.conv1x1_layer4(x))

			x_avg = self.global_avgpool(x)
			x_max = self.global_maxpool(x)
			gp = torch.cat((x_avg, x_max), dim=1)

			squeezed_fv = torch.nn.ReLU(inplace=True)(self.conv1x1_squeeze_layer4(gp))
			channel_attention = torch.sigmoid(self.conv1x1_expand_layer4(squeezed_fv))

		attention_x = x*space_attention + x*channel_attention + x

		return attention_x

## New Model Definition for DenseNet121
class DenseNet121ReID(Module):
    
	def __init__(self, model_base, Nc=None):
		super(DenseNet121ReID, self).__init__()

		self.model_base = model_base.features
		self.gap = AdaptiveAvgPool2d(1)
		self.gmp = AdaptiveMaxPool2d(output_size=(1, 1))
		self.last_bn = BatchNorm1d(2048)

		if Nc:
			self.classification = Linear(2048, Nc, bias=False)

	def forward(self, x):
		
		x = self.model_base(x)
		x = F.relu(x, inplace=True)

		x_avg = self.gap(x)
		x_max = self.gmp(x)
		x = x_avg + x_max
		x = torch.cat([x,x], dim=1)

		x = x.view(x.size(0), -1)

		output = self.last_bn(x)

		if self.training:
			output = output/torch.norm(output, dim=1, keepdim=True)
			probs = self.classification(output)
			#print(probs)
			return output, probs

		return output 

## New Definition for OSNET
class OSNETReID(Module):
    
	def __init__(self, model_base, embedding_size=512, Nc=None):
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

		if Nc:
			self.classification = Linear(512, Nc, bias=False)


	def forward(self, x, distortion_levels=None, eval_no_heads=False, eval_weighting=False, multiple_output=False):
		
		x = self.conv1(x)
		x = self.maxpool(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)

		v_avg = self.avgpool(x)
		v_max = self.maxpool02(x)
		v = v_avg + v_max
		v = v.view(v.size(0), -1)
		output = self.last_bn(v)
		#output = self.fc(v)

		#if self.training:
		#	output = output/torch.norm(output, dim=1, keepdim=True)
		#	probs = self.classification(output)
			#print(probs)
		#	return output, probs

		return output

class inceptionV3ReID(Module):
    
	def __init__(self, model_base, embedding_size=2048):
		super(inceptionV3ReID, self).__init__()


		self.Conv2d_1a_3x3 = model_base.Conv2d_1a_3x3
		self.Conv2d_2a_3x3 = model_base.Conv2d_2a_3x3
		self.Conv2d_2b_3x3 = model_base.Conv2d_2b_3x3 
		self.maxpool1 = model_base.maxpool1
		self.Conv2d_3b_1x1 = model_base.Conv2d_3b_1x1
		self.Conv2d_4a_3x3 = model_base.Conv2d_4a_3x3
		self.maxpool2 = model_base.maxpool2
		self.Mixed_5b = model_base.Mixed_5b
		self.Mixed_5c = model_base.Mixed_5c
		self.Mixed_5d = model_base.Mixed_5d
		self.Mixed_6a = model_base.Mixed_6a
		self.Mixed_6b = model_base.Mixed_6b
		self.Mixed_6c = model_base.Mixed_6c
		self.Mixed_6d = model_base.Mixed_6d
		self.Mixed_6e = model_base.Mixed_6e
		self.Mixed_7a = model_base.Mixed_7a
		self.Mixed_7b = model_base.Mixed_7b
		self.Mixed_7c = model_base.Mixed_7c
		
		self.global_avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
		self.global_maxpool = AdaptiveMaxPool2d(output_size=(1, 1))
		self.last_bn = BatchNorm1d(2048)
		

	def forward(self, x):

		x = self.Conv2d_1a_3x3(x)
		# N x 32 x 149 x 149
		x = self.Conv2d_2a_3x3(x)
		# N x 32 x 147 x 147
		x = self.Conv2d_2b_3x3(x)
		# N x 64 x 147 x 147
		x = self.maxpool1(x)
		# N x 64 x 73 x 73
		x = self.Conv2d_3b_1x1(x)
		# N x 80 x 73 x 73
		x = self.Conv2d_4a_3x3(x)
		# N x 192 x 71 x 71
		x = self.maxpool2(x)
		# N x 192 x 35 x 35
		x = self.Mixed_5b(x)
		# N x 256 x 35 x 35
		x = self.Mixed_5c(x)
		# N x 288 x 35 x 35
		x = self.Mixed_5d(x)
		# N x 288 x 35 x 35
		x = self.Mixed_6a(x)
		# N x 768 x 17 x 17
		x = self.Mixed_6b(x)
		# N x 768 x 17 x 17
		x = self.Mixed_6c(x)
		# N x 768 x 17 x 17
		x = self.Mixed_6d(x)
		# N x 768 x 17 x 17
		# NSA
		x = self.Mixed_6e(x)
		# N x 768 x 17 x 17
		x = self.Mixed_7a(x)
		# N x 1280 x 8 x 8
		x = self.Mixed_7b(x)
		# N x 2048 x 8 x 8
		x = self.Mixed_7c(x)
		# N x 2048 x 8 x 8
		# Adaptive average pooling
		
		x_avg = self.global_avgpool(x)
		x_max = self.global_maxpool(x)
		x = x_avg + x_max
		x = x.view(x.size(0), -1)

		output = self.last_bn(x)
		return output


## New Definition for Vision Transform
class ViTReID(Module):

	def __init__(self, model_base, img_height, img_width, patch_size=16, stride_size=16):
		super(ViTReID, self).__init__()

		self.patch_size = patch_size
		self.image_size = model_base.image_size
		self.hidden_dim = model_base.hidden_dim
		self.class_token = model_base.class_token

		self.conv_proj = model_base.conv_proj
		self.encoder = model_base.encoder
		self.heads = model_base.heads

		if img_height != 224 or img_width != 224:
			seq_length = (img_height // patch_size) * (img_width // patch_size)
			seq_length += 1

			self.encoder.pos_embedding = nn.Parameter(torch.empty(1, seq_length, self.hidden_dim).normal_(std=0.02))

		self.last_bn = BatchNorm1d(768)

	def _process_input(self, x):
		n, c, h, w = x.shape
		p = self.patch_size
		#torch._assert(h == self.image_size, "Wrong image height!")
		#torch._assert(w == self.image_size, "Wrong image width!")
		n_h = h // p
		n_w = w // p

		# (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
		x = self.conv_proj(x)
		# (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
		x = x.reshape(n, self.hidden_dim, n_h * n_w)

		# (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
		# The self attention layer expects inputs in the format (N, S, E)
		# where S is the source sequence length, N is the batch size, E is the
		# embedding dimension
		x = x.permute(0, 2, 1)

		return x


	def forward(self, x):

		# Reshape and permute the input tensor
		x = self._process_input(x)
		n = x.shape[0]

		# Expand the class token to the full batch
		batch_class_token = self.class_token.expand(n, -1, -1)
		x = torch.cat([batch_class_token, x], dim=1)

		x = self.encoder(x)

		# Classifier "token" as used by standard language architectures
		x = x[:, 0]

		#x = self.heads(x)
		x = self.last_bn(x)
		return x

## New Definition for Vision Transform
class efficientnetB0ReID(Module):

	def __init__(self, model_base, Nc=None):
		super(efficientnetB0ReID, self).__init__()

		self.features = model_base.features
		self.global_avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
		self.global_maxpool = AdaptiveMaxPool2d(output_size=(1, 1))
		self.last_bn = BatchNorm1d(1280)

		if Nc:
			self.classification = Linear(1280, Nc, bias=False)

	def forward(self, x):
		x = self.features(x)
		
		x_avg = self.global_avgpool(x)
		x_max = self.global_maxpool(x)
		x_avg = x_avg.view(x_avg.size(0), -1)
		x_max = x_max.view(x_max.size(0), -1)
		x = x_avg + x_max
		
		#x = torch.cat((x_upper, x_middle, x_lower), dim=1)
		#x = self.fc01(x)
		output = self.last_bn(x)

		#print(feature_map_probabilities, self.training)
		#if self.training:
		#	output = output/torch.norm(output, dim=1, keepdim=True)
		#	probs = self.classification(output)
			#print(probs)
		#	return output, probs
		
		return output
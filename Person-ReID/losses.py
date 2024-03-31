import torch
import numpy as np
from termcolor import colored

def getValueFromCosineSchedule(t_cur, t_max, n_min=0.0, n_max=1.0):
	nt = n_min + 0.5*(n_max - n_min)*(1 + np.cos(((t_max - t_cur)/t_max)*np.pi))
	return nt

def BatchCenterLoss(batch_fvs, batch_labels, centers, centers_labels, tau=0.1, gpu_index=0):

	# Calculating Similarity
	S = torch.matmul(batch_fvs, centers.T)
	centers_labels = torch.Tensor(centers_labels)
	
	# Calcuating Loss
	batch_loss = torch.tensor(0.0).cuda(gpu_index)

	for si in range(batch_fvs.shape[0]):
		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]

		#print(colored("###====== Jesus ======###", "blue"))
		#print(fvs_similarities)
		
		# Proxy Loss
		positive_similarity = fvs_similarities[centers_labels == pseudo_label][0]
		#print(positive_similarity)

		pos_sim = torch.exp(positive_similarity/tau)
		all_sim = torch.exp(fvs_similarities/tau).sum()
		batch_loss += -torch.log(pos_sim/all_sim)
		#print(-torch.log(pos_sim/all_sim))
		#exit()

	batch_loss = batch_loss/batch_fvs.shape[0]	

	return batch_loss

def BatchWeightedCenterLoss(batch_fvs, batch_labels, samples_distortion, centers, centers_labels, current_epoch, 
																	number_of_epoches, is_clean_training, tau=0.1, gpu_index=0):
	
	w0 = 1.0
	w1 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.8, n_max=1.0)
	w2 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.6, n_max=1.0)
	w3 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.4, n_max=1.0)
	w4 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.2, n_max=1.0)
	w5 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)
	
	distortion_weights = torch.Tensor([w0,w1,w2,w3,w4,w5])
	#samples_distortion = torch.Tensor(samples_distortion).long()
	w = distortion_weights[samples_distortion]
	w = w.reshape(w.shape[0],1).cuda(gpu_index)

	#if is_clean_training:
	#	field_frame_idx = torch.where(samples_distortion == 0)[0]
	#else:
	#	field_frame_idx = torch.where(samples_distortion != 0)[0]

	field_frame_idx = torch.arange(samples_distortion.shape[0])

	# Calculating Similarity
	Sim = torch.matmul(batch_fvs, centers.T)
	S_exp = torch.exp(Sim/tau)
	probs = S_exp/torch.sum(S_exp, dim=1, keepdim=True)
	predicted = centers_labels[torch.argmax(probs, dim=1).cpu().numpy()[field_frame_idx]]

	S = -w*torch.log(probs)
	#S = -torch.log(probs)

	nb = batch_fvs.shape[0]
	nc = centers.shape[0]

	centers_labels = torch.Tensor(centers_labels)
	batch_labels_expanded = batch_labels.repeat(nc,1).T
	centers_labels_expanded = centers_labels.repeat(nb,1)
	mask = (batch_labels_expanded == centers_labels_expanded).int().cuda(gpu_index)
	batch_loss = torch.sum(torch.sum(S*mask, dim=1)[field_frame_idx])/torch.sum(w[field_frame_idx]*torch.sum(mask, dim=1, keepdim=True))

	#batch_loss = torch.sum(S*mask, dim=1)

	#batch_loss = torch.mean(batch_loss)
	#batch_loss = torch.sum(batch_loss)/torch.sum(mask, dim=1).sum()

	samples_with_centroid_idx = torch.where(torch.sum(mask, dim=1) == 1)[0].detach().cpu()
	batch_acc_bal = getACCBal(predicted[samples_with_centroid_idx], batch_labels.numpy()[field_frame_idx][samples_with_centroid_idx])
	average_max_prob = torch.max(probs, dim=1)[0].detach().cpu().numpy()[field_frame_idx].mean()
	
	return batch_loss, batch_acc_bal, average_max_prob

def distortionLoss(batch_fvs, batch_labels, samples_distortion, current_epoch, number_of_epoches, gpu_index=0):

	w0 = 1.0
	w1 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.90, n_max=1.0)
	w2 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.85, n_max=1.0)
	w3 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.80, n_max=1.0)
	w4 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.75, n_max=1.0)
	w5 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.70, n_max=1.0)
	w6 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.6, n_max=1.0)
	w7 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.5, n_max=1.0)
	w8 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.4, n_max=1.0)
	w9 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.3, n_max=1.0)
	w10 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.2, n_max=1.0)
	w11 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)
	w12 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)

	lambda_distortion = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=1e-3, n_max=1e-1)

	distortion_weights = torch.Tensor([w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12])
	
	distortion_loss = torch.tensor(0.0).cuda(gpu_index)
	weights_sum = 0.0

	samples_distortion = torch.Tensor(samples_distortion)

	all_labels = torch.unique(batch_labels)

	for label in all_labels:
		#print(colored("###======== Label %s ========###" % label, "yellow"))
		label_idx = torch.where(batch_labels == label)[0]
		fvs = batch_fvs[label_idx]
		distortions = samples_distortion[label_idx]
		
		all_distortions = torch.unique(distortions)
		assert all_distortions[0] == 0

		clean_fvs = fvs[torch.where(distortions == 0)[0]]
		
		for distortion in all_distortions[1:]:
			#print(colored("#=== Distortion %s ===#" % distortion, "blue"))
			distortion_idx = torch.where(distortions == distortion)[0]
			
			if distortion_idx.shape[0] == clean_fvs.shape[0]:
				selected_idx = torch.arange(distortion_idx.shape[0])
			elif distortion_idx.shape[0] > clean_fvs.shape[0]:
				selected_idx = np.random.choice(distortion_idx.shape[0], size=clean_fvs.shape[0], replace=False)
			else:
				selected_idx = np.random.choice(distortion_idx.shape[0], size=clean_fvs.shape[0], replace=True)

			
			distortion_idx = distortion_idx[selected_idx]
			distortion_fvs = fvs[distortion_idx]
			w = distortion_weights[distortion.long()]
			
			distortion_loss += w*(clean_fvs - distortion_fvs).pow(2).sum(dim=1).mean()
			weights_sum += w

	distortion_loss = distortion_loss/weights_sum
	return distortion_loss



def BatchWeightedCrossEntropyLoss(probs, batch_labels, samples_distortion, labels_dict, current_epoch, number_of_epoches, gpu_index=0):
	
	w0 = 1.0
	w1 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.90, n_max=1.0)
	w2 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.85, n_max=1.0)
	w3 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.80, n_max=1.0)
	w4 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.75, n_max=1.0)
	w5 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.70, n_max=1.0)
	w6 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.6, n_max=1.0)
	w7 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.5, n_max=1.0)
	w8 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.4, n_max=1.0)
	w9 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.3, n_max=1.0)
	w10 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.2, n_max=1.0)
	w11 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)
	w12 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)

	distortion_weights = torch.Tensor([w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12])
	samples_distortion = torch.Tensor(samples_distortion).long()
	#samples_distortion = torch.zeros(len(samples_distortion), dtype=torch.long)
	w = distortion_weights[samples_distortion]
	w = w.reshape(w.shape[0],1).cuda(gpu_index)

	S = -w*torch.log(probs)

	labels = torch.Tensor([labels_dict[id_name.item()] for id_name in batch_labels.int()])
	
	nb, nc = probs.shape
	labels_expanded = labels.repeat(nc,1).T
	gt_expanded = torch.arange(nc).repeat(nb,1)
	mask = (labels_expanded == gt_expanded).int().cuda(gpu_index)
	batch_loss = torch.sum(torch.sum(S*mask, dim=1))/torch.sum(w)

	batch_acc_bal = getACCBal(torch.argmax(probs, dim=1).cpu().numpy(), labels.numpy())
	average_max_prob = torch.max(probs, dim=1)[0].detach().cpu().numpy().mean()

	return batch_loss, batch_acc_bal, average_max_prob


def getACCBal(predicted_labels, gt_labels):

	all_labels = np.union1d(np.unique(predicted_labels), np.unique(gt_labels))
	n = len(all_labels)
	predicted_labels_idx_cm = [np.where(l == all_labels)[0][0] for l in predicted_labels]
	gt_labels_idx_cm = [np.where(l == all_labels)[0][0] for l in gt_labels]
	cm = np.zeros((n,n))

	for lidx in np.arange(len(predicted_labels_idx_cm)):
		cm[gt_labels_idx_cm[lidx]][predicted_labels_idx_cm[lidx]] += 1.0

	cm_relative = cm/(np.sum(cm, axis=1)+1e-7)
	acc_bal = np.trace(cm_relative)/n
	return acc_bal


def BatchL2CenterLoss(batch_fvs, batch_labels, centers, centers_labels, tau=100, gpu_index=0):

	# Calculating Similarity
	eps = 1e-7
	fvs2centers_distances = torch.cdist(batch_fvs, centers, p=2.0).pow(2)
	#print(fvs2centers_distances)
	centers_labels = torch.Tensor(centers_labels)
	
	# Calcuating Loss
	batch_loss = torch.tensor(0.0).cuda(gpu_index)

	for si in range(batch_fvs.shape[0]):
		dists = fvs2centers_distances[si]
		pseudo_label = batch_labels[si]

		#print(colored("###====== Jesus ======###", "blue"))
		#print(dists)
		
		# Proxy Loss
		positive_distance = dists[centers_labels == pseudo_label][0]
		#print(positive_distance)

		#pos_sim = torch.exp(-1*positive_distance/tau)
		#all_sim = torch.exp(-1*dists/tau).sum() + eps
		batch_loss += positive_distance
		#print(-torch.log(pos_sim/all_sim))
		#exit()

	batch_loss = batch_loss/batch_fvs.shape[0]	

	return batch_loss

def BatchProxyLoss(batch_fvs, batch_labels, all_proxies, proxies_labels, top_negs=50, tau=0.1, gpu_index=0):

	# Calculating Similarity
	S = torch.matmul(batch_fvs, all_proxies.T)
	
	# Calcuating Loss
	batch_loss = torch.tensor(0.0).cuda(gpu_index)

	for si in range(batch_fvs.shape[0]):
		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]

		#print(fvs_similarities)
		
		# Proxy Loss
		positive_similarity = fvs_similarities[proxies_labels == pseudo_label]
		#print("Positive Similarities", positive_similarity)

		all_negative_similarity = fvs_similarities[proxies_labels != pseudo_label]
		negative_similarity = torch.topk(all_negative_similarity, k=top_negs, largest=True)[0]
		#print("Negative Similarities", negative_similarity)

		pos_sim = torch.exp(positive_similarity/tau)
		neg_sim = torch.exp(negative_similarity/tau)

		ratio = pos_sim/(pos_sim.sum() + neg_sim.sum())
		batch_loss += -torch.log(ratio).mean()
		#print(-torch.log(ratio).mean())
		#exit()

	batch_loss = batch_loss/batch_fvs.shape[0]	

	return batch_loss


def BatchWeightedProxyLoss(batch_fvs, batch_labels, samples_distortion, all_proxies, proxies_labels, current_epoch, 
																		number_of_epoches, top_negs=50, tau=0.1, gpu_index=0):

	# Calculating Similarity
	S = torch.matmul(batch_fvs, all_proxies.T)

	w0 = 1.0
	w1 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.8, n_max=1.0)
	w2 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.6, n_max=1.0)
	w3 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.4, n_max=1.0)
	w4 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.2, n_max=1.0)
	w5 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)
	
	distortion_weights = torch.Tensor([w0,w1,w2,w3,w4,w5])
	#distortion_weights = torch.ones(13)
	
	proxies_labels = torch.Tensor(proxies_labels)

	# Calcuating Loss
	batch_loss = torch.tensor(0.0).cuda(gpu_index)
	sample_weights = []

	for si in range(batch_fvs.shape[0]):

		#print(colored("#======== Sample %d ========#" % si, "yellow"))
		w = distortion_weights[samples_distortion[si]]
		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]

		#print("Distortion level and weight: %d, %.3f" % (samples_distortion[si], w))
		#print(fvs_similarities)
		
		pos_idx = np.where(proxies_labels == pseudo_label)[0]
		neg_idx = np.where(proxies_labels != pseudo_label)[0]

		# Proxy Loss
		if len(pos_idx) > 0:
			positive_similarities = fvs_similarities[pos_idx]
			negative_similarities = fvs_similarities[neg_idx]
			num_positives = positive_similarities.shape[0]
			#print("Shapes", positive_similarities.shape, negative_similarities.shape)

			#p, pos_idx = torch.topk(positive_similarities, k=num_positives, largest=False)
			#q, neg_idx = torch.topk(negative_similarities, k=num_positives, largest=True)

			#print(p, pos_idx, q, neg_idx)
			selected_negative_similarities = torch.topk(negative_similarities, k=num_positives, largest=True)[0]

			#print("Positive Similarities", positive_similarities)
			#print("Negative Similarities", selected_negative_similarities)

			pos_sim = torch.exp(positive_similarities/tau)
			neg_sim = torch.exp(selected_negative_similarities/tau)

			ratio = pos_sim/(pos_sim.sum() + neg_sim.sum())
			batch_loss += -w*torch.log(ratio).mean()



			#p = torch.exp(p[0]/tau)
			#q = torch.exp(q[0]/tau)
			#batch_loss += -w*torch.log(p/(p+q))
			sample_weights.append(w)
		

	batch_loss = batch_loss/np.sum(sample_weights)
	#print("Final Loss", batch_loss)
	#exit()
	return batch_loss

def BatchSoftmaxTripletLoss(batch_fvs, batch_labels, batch_pids, tau=0.1, gpu_index=0):
  
	S = torch.mm(batch_fvs, batch_fvs.T)
	batch_loss = torch.tensor(0.0).cuda(gpu_index)
	corrects = 0
	total_number_triplets = 0

	for si in np.arange(S.shape[0]):

		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]
		true_label = batch_pids[si] # Only for reference! It is NOT used on optmization!!

		positive_similarities = fvs_similarities[batch_labels == pseudo_label]
		negative_similarities = fvs_similarities[batch_labels != pseudo_label]

		#print(positive_similarities.shape, negative_similarities.shape)

		p, pos_idx = torch.topk(positive_similarities, k=1, largest=False)
		q, neg_idx = torch.topk(negative_similarities, k=1, largest=True)

		#print(p, pos_idx, q, neg_idx)

		p = torch.exp(p[0]/tau)
		q = torch.exp(q[0]/tau)

		sample_loss = -torch.log(p/(p+q))
		batch_loss += sample_loss

		pos_pid = batch_pids[batch_labels == pseudo_label][pos_idx[0]]
		neg_pid = batch_pids[batch_labels != pseudo_label][neg_idx[0]]

		#print(true_label, pos_pid, neg_pid)

		if (true_label == pos_pid) and (true_label != neg_pid):
			corrects += 1
		total_number_triplets += 1

	loss = batch_loss/S.shape[0]
	return loss, corrects, total_number_triplets

def BatchSoftmaxClothesTripletLoss(batch_fvs, batch_labels, batch_clothes, tau=0.1, gpu_index=0):
  
	S = torch.mm(batch_fvs, batch_fvs.T)
	batch_loss = torch.tensor(0.0).cuda(gpu_index)
	number_of_triplets = 0
	
	for si in np.arange(S.shape[0]):

		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]
		cloth_label = batch_clothes[si]

		positive_similarities = fvs_similarities[(batch_labels == pseudo_label) & (batch_clothes != cloth_label)]
		negative_similarities = fvs_similarities[batch_labels != pseudo_label]

		num_positives = positive_similarities.shape[0]
		num_negatives = negative_similarities.shape[0]

		if num_positives > 0 and num_negatives > 0:

			p, pos_idx = torch.topk(positive_similarities, k=1, largest=False)
			q, neg_idx = torch.topk(negative_similarities, k=1, largest=True)

			#print(p, pos_idx, q, neg_idx)

			p = torch.exp(p[0]/tau)
			q = torch.exp(q[0]/tau)

			sample_loss = -torch.log(p/(p+q))
			batch_loss += sample_loss
			number_of_triplets += 1

	loss = batch_loss/number_of_triplets
	return loss

def BatchSoftmaxAllTripletLoss(batch_fvs, batch_labels, samples_distortion, current_epoch, number_of_epoches, tau=0.1, gpu_index=0):
	  
	w0 = 1.0
	w1 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.90, n_max=1.0)
	w2 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.85, n_max=1.0)
	w3 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.80, n_max=1.0)
	w4 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.75, n_max=1.0)
	w5 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.70, n_max=1.0)
	w6 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.6, n_max=1.0)
	w7 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.5, n_max=1.0)
	w8 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.4, n_max=1.0)
	w9 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.3, n_max=1.0)
	w10 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.2, n_max=1.0)
	w11 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)
	w12 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)

	distortion_weights = torch.Tensor([w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12])

	distortion_weights = torch.Tensor([w0,w1,w2,w3,w4,w5])
	samples_distortion = torch.Tensor(samples_distortion).long()
	w = distortion_weights[samples_distortion].cuda(gpu_index)
	
	S = torch.mm(batch_fvs, batch_fvs.T)
	S_exp = torch.exp(S/tau)

	batch_labels = np.array([batch_labels.numpy()])
	nb = batch_fvs.shape[0]
	
	batch_labels_expanded = np.repeat(batch_labels, nb, axis=0)
	batch_labels_expanded_transposed = np.repeat(batch_labels, nb, axis=0).T
	
	pos_mask = torch.Tensor(batch_labels_expanded == batch_labels_expanded_transposed).int().cuda(gpu_index)
	neg_mask = 1 - pos_mask

	pos_sim = S_exp*pos_mask
	neg_sim = S_exp*neg_mask

	neg_sum = torch.sum(neg_sim, dim=1, keepdim=True)
	relative_pos = -torch.log(S_exp/(S_exp+neg_sum))*pos_mask

	#pos_weights = pos_dist/torch.sum(pos_dist, dim=1, keepdim=True)
	#neg_weights = neg_dist/torch.sum(neg_dist, dim=1, keepdim=True)	

	#batch_loss = torch.mean(torch.sum(relative_pos, dim=1)/torch.sum(pos_mask, dim=1))
	batch_loss = torch.sum(w*(torch.sum(relative_pos, dim=1)/torch.sum(pos_mask, dim=1)))/torch.sum(w)
	return batch_loss


def BatchSoftmaxBipatiteLoss(batch_fvs, batch_labels, batch_cameras, batch_clothes, samples_distortion, current_epoch, number_of_epoches, tau=0.1, gpu_index=0):
	  
	w0 = 1.0
	w1 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.90, n_max=1.0)
	w2 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.85, n_max=1.0)
	w3 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.80, n_max=1.0)
	w4 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.75, n_max=1.0)
	w5 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.70, n_max=1.0)
	w6 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.6, n_max=1.0)
	w7 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.5, n_max=1.0)
	w8 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.4, n_max=1.0)
	w9 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.3, n_max=1.0)
	w10 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.2, n_max=1.0)
	w11 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)
	w12 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)

	distortion_weights = torch.Tensor([w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12])
	#distortion_weights = torch.Tensor([1,1,1,1,1,1,1,1,1,1,1,1,1])
	#samples_distortion = torch.Tensor(samples_distortion).long()
	w = distortion_weights[samples_distortion].cuda(gpu_index)
	
	S = torch.mm(batch_fvs, batch_fvs.T)
	S_exp = torch.exp(S/tau)
	
	nb = batch_fvs.shape[0]

	batch_labels = np.array([batch_labels.numpy()])
	batch_cameras = np.array([batch_cameras])
	batch_clothes = np.array([batch_clothes])

	# Labels, cameras and clothes annotation expansions
	batch_labels_expanded = np.repeat(batch_labels, nb, axis=0)
	batch_labels_expanded_transposed = np.repeat(batch_labels, nb, axis=0).T

	batch_cameras_expanded = np.repeat(batch_cameras, nb, axis=0)
	batch_cameras_expanded_transposed = np.repeat(batch_cameras, nb, axis=0).T

	batch_clothes_expanded = np.repeat(batch_clothes, nb, axis=0)
	batch_clothes_expanded_transposed = np.repeat(batch_clothes, nb, axis=0).T
	
	# Masks definition
	labels_mask = torch.Tensor(batch_labels_expanded == batch_labels_expanded_transposed).int().cuda(gpu_index)
	cameras_mask = torch.Tensor(batch_cameras_expanded == batch_cameras_expanded_transposed).int().cuda(gpu_index)
	clothes_mask = torch.Tensor(batch_clothes_expanded == batch_clothes_expanded_transposed).int().cuda(gpu_index)
	
	# Clothing-based partitions
	pos_mask = labels_mask*cameras_mask*(1-clothes_mask)
	pos_sim = S_exp*pos_mask
	neg_sim = S_exp*(1-labels_mask)*cameras_mask

	pos_sum = torch.sum(pos_sim, dim=1, keepdim=True)
	neg_sum = torch.sum(neg_sim, dim=1, keepdim=True)

	eps = 1e-9
	relative_pos = -torch.log((pos_sim+eps)/(pos_sum+neg_sum+eps))*pos_mask

	batch_clothes_loss = torch.sum(w*(torch.sum(relative_pos, dim=1)/(torch.sum(pos_mask, dim=1)+eps)))/torch.sum(w)

	if torch.isnan(batch_clothes_loss):
		a = 0

	# Camera-based partitions
	batch_cameras_expanded = np.repeat(batch_cameras == 'controlled', nb, axis=0)
	batch_cameras_expanded_transposed = np.repeat(batch_cameras != 'controlled', nb, axis=0).T
	cameras_mask = torch.Tensor(batch_cameras_expanded == batch_cameras_expanded_transposed).int().cuda(gpu_index)

	pos_mask = labels_mask*cameras_mask
	pos_sim = S_exp*pos_mask
	neg_sim = S_exp*(1-labels_mask)*cameras_mask

	pos_sum = torch.sum(pos_sim, dim=1, keepdim=True)
	neg_sum = torch.sum(neg_sim, dim=1, keepdim=True)

	relative_pos = -torch.log((pos_sim+eps)/(pos_sum+neg_sum+eps))*pos_mask

	batch_camera_loss = torch.sum(w*(torch.sum(relative_pos, dim=1)/(torch.sum(pos_mask, dim=1)+eps)))/torch.sum(w)

	if torch.isnan(batch_camera_loss):
		a = 0

	return batch_clothes_loss, batch_camera_loss

def BatchWeightedPoseLoss(batch_fvs, batch_labels, batch_cameras, batch_clothes, samples_distortion, current_epoch, number_of_epoches, tau=0.1, gpu_index=0):
  	
	w0 = 1.0
	w1 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.90, n_max=1.0)
	w2 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.85, n_max=1.0)
	w3 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.80, n_max=1.0)
	w4 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.75, n_max=1.0)
	w5 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.70, n_max=1.0)
	w6 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.6, n_max=1.0)
	w7 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.5, n_max=1.0)
	w8 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.4, n_max=1.0)
	w9 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.3, n_max=1.0)
	w10 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.2, n_max=1.0)
	w11 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)
	w12 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)

	distortion_weights = torch.Tensor([w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12])
	w = distortion_weights[samples_distortion].cuda(gpu_index)
	
	S = torch.mm(batch_fvs, batch_fvs.T)
	S_exp = torch.exp(S/tau)
	
	nb = batch_fvs.shape[0]

	batch_labels = np.array([batch_labels.numpy()])
	batch_cameras = np.array([batch_cameras])
	batch_clothes = np.array([batch_clothes])

	# Labels and clothes annotation expansions
	batch_labels_expanded = np.repeat(batch_labels, nb, axis=0)
	batch_labels_expanded_transposed = np.repeat(batch_labels, nb, axis=0).T

	batch_cameras_expanded = np.repeat(batch_cameras, nb, axis=0)
	batch_cameras_expanded_transposed = np.repeat(batch_cameras, nb, axis=0).T

	batch_clothes_expanded = np.repeat(batch_clothes, nb, axis=0)
	batch_clothes_expanded_transposed = np.repeat(batch_clothes, nb, axis=0).T
	
	# Masks definition
	labels_mask = torch.Tensor(batch_labels_expanded == batch_labels_expanded_transposed).int().cuda(gpu_index)
	cameras_mask = torch.Tensor(batch_cameras_expanded == batch_cameras_expanded_transposed).int().cuda(gpu_index)
	clothes_mask = torch.Tensor(batch_clothes_expanded == batch_clothes_expanded_transposed).int().cuda(gpu_index)
	
	# Clothing-based partitions
	pos_mask = labels_mask*clothes_mask*cameras_mask
	pos_sim = S_exp*pos_mask
	neg_sim = S_exp*(1-labels_mask)*cameras_mask

	pos_sum = torch.sum(pos_sim, dim=1, keepdim=True)
	neg_sum = torch.sum(neg_sim, dim=1, keepdim=True)

	eps = 1e-9
	relative_pos = -torch.log((pos_sim+eps)/(pos_sum+neg_sum+eps))*pos_mask

	batch_pose_loss = torch.sum(w*(torch.sum(relative_pos, dim=1)/(torch.sum(pos_mask, dim=1)+eps)))/torch.sum(w)

	return batch_pose_loss

def BatchWeightedSoftmaxTripletLoss(batch_fvs, batch_labels, samples_distortion, current_epoch, number_of_epoches, 
																										tau=0.1, gpu_index=0):
  
	S = torch.mm(batch_fvs, batch_fvs.T)
	batch_loss = torch.tensor(0.0).cuda(gpu_index)
	
	w0 = 1.0
	w1 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.90, n_max=1.0)
	w2 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.85, n_max=1.0)
	w3 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.80, n_max=1.0)
	w4 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.75, n_max=1.0)
	w5 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.70, n_max=1.0)
	w6 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.6, n_max=1.0)
	w7 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.5, n_max=1.0)
	w8 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.4, n_max=1.0)
	w9 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.3, n_max=1.0)
	w10 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.2, n_max=1.0)
	w11 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)
	w12 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)

	distortion_weights = torch.Tensor([w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12])
	
	sample_weights = []

	for si in np.arange(S.shape[0]):
		w = distortion_weights[samples_distortion[si]]
		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]
		
		positive_similarities = fvs_similarities[batch_labels == pseudo_label]
		negative_similarities = fvs_similarities[batch_labels != pseudo_label]

		#print(positive_similarities.shape, negative_similarities.shape)

		p, pos_idx = torch.topk(positive_similarities, k=1, largest=False)
		q, neg_idx = torch.topk(negative_similarities, k=1, largest=True)

		#print(p, pos_idx, q, neg_idx)

		p = torch.exp(p[0]/tau)
		q = torch.exp(q[0]/tau)

		sample_loss = -w*torch.log(p/(p+q))
		sample_weights.append(w)
		batch_loss += sample_loss

	loss = batch_loss/np.sum(sample_weights)
	return loss

def BatchWeightedSoftmaxAllTripletLoss(batch_fvs, batch_labels, samples_distortion, current_epoch, number_of_epoches, 
																										tau=0.1, gpu_index=0):
  
	w0 = 1.0
	w1 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.8, n_max=1.0)
	w2 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.6, n_max=1.0)
	w3 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.4, n_max=1.0)
	w4 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.2, n_max=1.0)
	w5 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)

	distortion_weights = torch.Tensor([w0,w1,w2,w3,w4,w5])
	samples_distortion = torch.Tensor(samples_distortion).long()
	w = distortion_weights[samples_distortion].cuda(gpu_index)
	
	# Calculating Distance
	Dist = 1.0 - (torch.matmul(batch_fvs, batch_fvs.T) + 1.0)/2.0
	
	nb = batch_fvs.shape[0]
	batch_labels_expanded = batch_labels.repeat(nb,1)
	batch_labels_expanded_transposed = batch_labels.repeat(nb,1).T
	
	pos_mask = (batch_labels_expanded == batch_labels_expanded_transposed).int().cuda(gpu_index)
	neg_mask = 1 - pos_mask

	pos_dist = torch.exp(Dist)*pos_mask
	neg_dist = torch.exp(-Dist)*neg_mask

	pos_weights = pos_dist/torch.sum(pos_dist, dim=1, keepdim=True)
	neg_weights = neg_dist/torch.sum(neg_dist, dim=1, keepdim=True)	

	pos_loss = torch.sum(pos_weights*Dist, dim=1)
	neg_loss = torch.sum(neg_weights*Dist, dim=1)

	batch_loss = torch.sum(w*torch.log(1 + torch.exp(pos_loss - neg_loss)))/torch.sum(w)
	
	return batch_loss

def BatchWeightedSoftmaxAllCosineLoss(batch_fvs, batch_labels, samples_distortion, current_epoch, number_of_epoches, 
																										tau=0.1, gpu_index=0):
  
	S = torch.mm(batch_fvs, batch_fvs.T)
	batch_loss = torch.tensor(0.0).cuda(gpu_index)
	
	w0 = 1.0
	w1 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.8, n_max=1.0)
	w2 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.6, n_max=1.0)
	w3 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.4, n_max=1.0)
	w4 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.2, n_max=1.0)
	w5 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)
	distortion_weights = [w0,w1,w2,w3,w4,w5]

	sample_weights = []

	for si in np.arange(S.shape[0]):
		#print(colored("#======== sample %d ========#" % si, "yellow"))
		w = distortion_weights[samples_distortion[si]]
		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]
		
		positive_similarities = fvs_similarities[batch_labels == pseudo_label]
		# It consideres the own sample
		num_pos_instances = positive_similarities.shape[0] - 1
		#print(positive_distances.shape, positive_distances, positive_distances.sum()/num_pos_instances)
		#print(samples_distortion[si], w)

		sample_loss = w*torch.sum(1.0 - positive_similarities)/num_pos_instances
		sample_weights.append(w)
		batch_loss += sample_loss

	loss = batch_loss/np.sum(sample_weights)
	#print("Loss", loss)
	#exit()
	return loss

def BatchControlledCameraHardLoss(batch_fvs, batch_labels, samples_distortion, current_epoch, number_of_epoches, tau=0.1, gpu_index=0):
  
	w0 = 1.0
	w1 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.90, n_max=1.0)
	w2 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.85, n_max=1.0)
	w3 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.80, n_max=1.0)
	w4 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.75, n_max=1.0)
	w5 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.70, n_max=1.0)
	w6 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.6, n_max=1.0)
	w7 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.5, n_max=1.0)
	w8 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.4, n_max=1.0)
	w9 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.3, n_max=1.0)
	w10 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.2, n_max=1.0)
	w11 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)
	w12 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)

	distortion_weights = torch.Tensor([w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12])
	
	S = torch.mm(batch_fvs, batch_fvs.T)
	batch_loss = torch.tensor(0.0).cuda(gpu_index)

	total_number_triplets = 0
	triplets_weights = []

	for si in np.arange(S.shape[0]):
		#print(colored("#======== sample %d ========#" % si, "yellow"))
		if samples_distortion[si] == 0:
			sample_loss = torch.tensor(0.0).cuda(gpu_index)
			#w = distortion_weights[samples_distortion[si]]
			fvs_similarities = S[si]
			pseudo_label = batch_labels[si]

			class_distortions = np.unique(samples_distortion[batch_labels == pseudo_label])
			
			# Sanity checking to make sure there is camera from rand/struct controlled setup
			assert class_distortions[0] == 0

			# Verify if there are images from field cameras
			if len(class_distortions[1:]) > 0:

				negative_similarities = fvs_similarities[(batch_labels != pseudo_label).numpy() & (samples_distortion == samples_distortion[si])]
				negative_similarities = torch.sort(negative_similarities, descending=True)[0]
				
				neg_idx = 0
				for cdistortion in class_distortions[1:]:
					pos_similarities = fvs_similarities[(batch_labels == pseudo_label).numpy() & (samples_distortion == cdistortion)]
					pos_sim = torch.exp(torch.min(pos_similarities)/tau)
					neg_sim = torch.exp(negative_similarities[neg_idx]/tau)
					w = distortion_weights[cdistortion]

					sample_loss += -w*torch.log(pos_sim/(pos_sim + neg_sim))
					neg_idx += 1

					total_number_triplets += 1
					triplets_weights.append(w)

				batch_loss += sample_loss

	#loss = batch_loss/total_number_triplets
	triplets_weights = torch.Tensor(triplets_weights)
	loss = batch_loss/torch.sum(triplets_weights)
	return loss


def BatchCameraHardLoss(batch_fvs, batch_labels, batch_camera_labels, centers_cameras, centers_cameras_labels, samples_distortion, 
																						current_epoch, number_of_epoches, tau=0.1, gpu_index=0):
  
	w0 = 1.0
	w1 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.90, n_max=1.0)
	w2 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.85, n_max=1.0)
	w3 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.80, n_max=1.0)
	w4 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.75, n_max=1.0)
	w5 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.70, n_max=1.0)
	w6 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.6, n_max=1.0)
	w7 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.5, n_max=1.0)
	w8 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.4, n_max=1.0)
	w9 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.3, n_max=1.0)
	w10 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.2, n_max=1.0)
	w11 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)
	w12 = getValueFromCosineSchedule(current_epoch, number_of_epoches, n_min=0.1, n_max=1.0)

	distortion_weights = torch.Tensor([w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12])
	# NSA
	S = torch.mm(batch_fvs, centers_cameras.T)
	batch_loss = torch.tensor(0.0).cuda()
	num_similarities = 0
	
	for si in range(S.shape[0]):

		#print(colored("#======== sample %d ========#" % si, "yellow"))
		
		fvs_similarities = S[si]

		id_label = str(int(batch_labels[si].item()))
		#print("ID Label", id_label)
		camera_label = batch_camera_labels[si].item()
		#print("Camera label", camera_label)

		#print(centers_cameras_labels)

		cameras = centers_cameras_labels[np.where(centers_cameras_labels[:,0] == id_label)[0], 1]
		#print(cameras)

		all_neg_idx = np.where(np.logical_and(centers_cameras_labels[:,0] != id_label, centers_cameras_labels[:,1] == camera_label))[0]

		# This IF here is to make sure there is at least one negative camera-proxy for each sample 
		if len(all_neg_idx) > 0:
			#print(all_neg_idx)
			all_neg_idx_sorted = torch.argsort(fvs_similarities[all_neg_idx], descending=True)
			#print(all_neg_idx_sorted)
			neg_counter = 0

			#print("Neg similarities", fvs_similarities[all_neg_idx][all_neg_idx_sorted])

			for cam in cameras:
				if cam != camera_label:
					#print("cam", cam)
					pos_idx = np.where(np.logical_and(centers_cameras_labels[:,0] == id_label, centers_cameras_labels[:,1] == cam))[0]
					#print("pos_idx", pos_idx)
					neg_idx = all_neg_idx[all_neg_idx_sorted[neg_counter]]
					#print(neg_idx)

					neg_counter += 1

					if neg_counter == len(all_neg_idx_sorted):
						neg_counter = 0

					#print(fvs_similarities[pos_idx[0]], fvs_similarities[neg_idx])
					neg_sim = torch.exp(fvs_similarities[neg_idx]/tau)
					pos_sim = torch.exp(fvs_similarities[pos_idx[0]]/tau)

					sample_loss = -torch.log(pos_sim/(pos_sim+neg_sim))
					batch_loss += sample_loss
					num_similarities += 1
	
	loss = batch_loss/num_similarities
	return loss

def BatchDistortionLoss(batch_fvs, distorted_fvs, gpu_index=0):
		
	num_samples = batch_fvs.shape[0]
	total_similarity = torch.tensor(0.0).cuda(gpu_index)

	for i in range(num_samples):
		start_idx = i*6 + 1   
		end_idx = (i+1)*6     

		#print(colored("Start: %d, End: %d" % (start_idx, end_idx), "yellow"))
		S = (1.0 - torch.mm(batch_fvs[i:(i+1)], distorted_fvs[start_idx:end_idx].T)).mean()
		total_similarity += S
		#print(S)

	loss = total_similarity/num_samples
	return loss

def BatchInstanceLoss(batch_fvs, gpu_index=0):
		
	augmented_fvs01 = batch_fvs[::2]
	augmented_fvs02 = batch_fvs[1::2]

	S = 1.0 - torch.mm(augmented_fvs01, augmented_fvs02.T)
	batch_loss = torch.trace(S)
	loss = batch_loss/S.shape[0]

	return loss

def BatchHardSoftmaxTripletLoss(batch_fvs, batch_labels, centers, centers_labels, tau=0.1, gpu_index=0):
  
	S = torch.mm(batch_fvs, centers.T)
	batch_loss = torch.tensor(0.0).cuda(gpu_index)
	centers_labels = torch.Tensor(centers_labels)

	#print(colored("Inside the loss!", "yellow"))
	for si in np.arange(S.shape[0]):

		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]
		
		positive_similarities = fvs_similarities[centers_labels == pseudo_label]
		negative_similarities = fvs_similarities[centers_labels != pseudo_label]

		#print(positive_similarities.shape, negative_similarities.shape)

		p, pos_idx = torch.topk(positive_similarities, k=1, largest=False)
		q, neg_idx = torch.topk(negative_similarities, k=1, largest=True)

		#print(p, pos_idx, q, neg_idx)

		p = torch.exp(p[0]/tau)
		q = torch.exp(q[0]/tau)

		sample_loss = -torch.log(p/(p+q))
		batch_loss += sample_loss

	loss = batch_loss/S.shape[0]
	return loss

def BatchMedianSoftmaxTripletLoss(batch_fvs, batch_labels, tau=0.1, gpu_index=0):
  
	S = torch.mm(batch_fvs, batch_fvs.T)
	batch_loss = torch.tensor(0.0).cuda(gpu_index)
	corrects = 0
	total_number_triplets = 0

	for si in np.arange(S.shape[0]):

		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]
		true_label = batch_pids[si] # Only for reference! It is NOT used on optmization!!

		positive_similarities = fvs_similarities[batch_labels == pseudo_label]
		negative_similarities = fvs_similarities[batch_labels != pseudo_label]

		#print(positive_similarities.shape, negative_similarities.shape)


		#p, pos_idx = torch.topk(positive_similarities, k=1, largest=False)
		#q, neg_idx = torch.topk(negative_similarities, k=1, largest=True)

		p, pos_idx = torch.median(positive_similarities, dim=0)
		q, neg_idx = torch.median(negative_similarities, dim=0)

		#print(p, pos_idx, q, neg_idx)

		#exit()
		p = torch.exp(p/tau)
		q = torch.exp(q/tau)

		sample_loss = -torch.log(p/(p+q))
		batch_loss += sample_loss

		pos_pid = batch_pids[batch_labels == pseudo_label][pos_idx]
		neg_pid = batch_pids[batch_labels != pseudo_label][neg_idx]

		#print(true_label, pos_pid, neg_pid)

		if (true_label == pos_pid) and (true_label != neg_pid):
			corrects += 1
		total_number_triplets += 1

	loss = batch_loss/S.shape[0]
	return loss, corrects, total_number_triplets
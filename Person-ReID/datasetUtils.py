import os
import numpy as np
from termcolor import colored
from tabulate import tabulate

def load_set_from_market_duke(directory):
	
	images_names = []
	for filename in os.listdir(directory):
		if filename.endswith(".jpg"):
			camid = int(filename.split("_")[1][1])
			pid = int(filename.split("_")[0])
			if(pid != -1):
				img_path = os.path.join(directory, filename)
				images_names.append([img_path, pid, camid, 'person'])
	            
	images_names = np.array(images_names)
	return images_names

def load_set_from_veri(directory):
	
	images_names = []
	for filename in os.listdir(directory):
		if filename.endswith(".jpg"):
			camid = int(filename.split("_")[1][1:])
			pid = int(filename.split("_")[0])
			if(pid != -1):
				img_path = os.path.join(directory, filename)
				images_names.append([img_path, pid, camid, 'object'])
	            
	images_names = np.array(images_names)

	return images_names

def load_set_from_veriWild(PATH, base_name):
	
	images_names = []
	file = open(PATH, "r")
	for line in file.readlines():
		subpath, refid, camid = line.split(" ")

		pid, img_name = subpath.split("/")

		img_path = os.path.join(base_name, subpath)
		pid = int(pid)
		camid = int(camid)

		images_names.append([img_path, pid, camid, 'object'])

	images_names = np.array(images_names)
	return images_names

def load_set_from_MSMT17(PATH, base_name):
	
	images_names = []
	train_file = open(PATH, "r")
	for line in train_file.readlines():
		img_name, pid_name = line.split(" ")

		pid = int(pid_name[:-1])
		camid = img_name.split("_")[2]
		
		img_path = os.path.join(base_name, img_name)
		images_names.append([img_path, pid, camid, 'person'])

	images_names = np.array(images_names)
	return images_names

def load_set_from_DeepChange(base_name, file_path, split_set):
	
	images_names_considering_cameras = []
	images_names_considering_days = []
	images_names_considering_tracklets = []
	
	train_file = open(os.path.join(base_name, file_path), "r")
	for line in train_file.readlines():
		img_name, tracklet_id = line.split(",")

		img_path = os.path.join(base_name, split_set, img_name)
		pid, camid, day_id, hour_id, _, _ = img_name.split("_")

		images_names_considering_cameras.append([img_path, pid[1:], camid[1:], 'person'])
		images_names_considering_days.append([img_path, pid[1:], day_id, 'person'])
		images_names_considering_tracklets.append([img_path, pid[1:], str(int(tracklet_id)), 'person'])

	images_names_considering_cameras = np.array(images_names_considering_cameras)
	images_names_considering_days = np.array(images_names_considering_days)
	images_names_considering_tracklets = np.array(images_names_considering_tracklets)

	#return [images_names_considering_cameras, images_names_considering_days, images_names_considering_tracklets]
	return images_names_considering_tracklets

def load_general_set(PATH):
	
	images_names = []
	file = open(PATH, "r")
	for line in file.readlines():
		full_img_name, pid, camid, kind = line.split(" ")
		images_names.append([full_img_name, pid, camid, kind[:-1]])

	images_names = np.array(images_names)
	return images_names


## Load target dataset
def load_dataset(dataset_name):
	
	if dataset_name == "Market":
	
		train_images = load_set_from_market_duke("/scratch/datasets/ReID_datasets/Market-1501-v15.09.15/bounding_box_train")
		gallery_images = load_set_from_market_duke("/scratch/datasets/ReID_datasets/Market-1501-v15.09.15/bounding_box_test")
		queries_images = load_set_from_market_duke("/scratch/datasets/ReID_datasets/Market-1501-v15.09.15/query")

	elif dataset_name == "Duke":

		train_images = load_set_from_market_duke("/scratch/datasets/ReID_datasets/DukeMTMC-reID/bounding_box_train")
		gallery_images = load_set_from_market_duke("/scratch/datasets/ReID_datasets/DukeMTMC-reID/bounding_box_test")
		queries_images = load_set_from_market_duke("/scratch/datasets/ReID_datasets/DukeMTMC-reID/query")

	elif dataset_name == "MSMT17":

		base_name_train = "/scratch/datasets/ReID_datasets/MSMT17_V2/mask_train_v2"
		#train_images = load_set_from_MSMT17("/scratch/datasets/ReID_datasets/MSMT17_V2/list_train_uda.txt", base_name_train)
		train_images = load_set_from_MSMT17("/scratch/datasets/ReID_datasets/MSMT17_V2/list_train.txt", base_name_train)
		val_images = load_set_from_MSMT17("/scratch/datasets/ReID_datasets/MSMT17_V2/list_val.txt", base_name_train)

		base_name_test = "/scratch/datasets/ReID_datasets/MSMT17_V2/mask_test_v2"
		gallery_images = load_set_from_MSMT17("/scratch/datasets/ReID_datasets/MSMT17_V2/list_gallery.txt", base_name_test)
		queries_images = load_set_from_MSMT17("/scratch/datasets/ReID_datasets/MSMT17_V2/list_query.txt", base_name_test)

		num_train_samples = train_images.shape[0]
		num_train_ids = np.unique(train_images[:,1]).shape[0]
		num_train_cameras = np.unique(train_images[:,2]).shape[0]

		num_val_samples = val_images.shape[0]
		num_val_ids = np.unique(val_images[:,1]).shape[0]
		num_val_cameras = np.unique(val_images[:,2]).shape[0]

		num_queries_samples = queries_images.shape[0]
		num_queries_ids = np.unique(queries_images[:,1]).shape[0]
		num_queries_cameras = np.unique(queries_images[:,2]).shape[0]

		num_gallery_samples = gallery_images.shape[0]
		num_gallery_ids = np.unique(gallery_images[:,1]).shape[0]
		num_gallery_cameras = np.unique(gallery_images[:,2]).shape[0]

		datasets_descriptions = [["MSMT17", num_train_samples, num_train_ids, num_train_cameras,
											num_val_samples, num_val_ids, num_val_cameras,
											num_queries_samples, num_queries_ids, num_queries_cameras,
											num_gallery_samples, num_gallery_ids, num_gallery_cameras]]

		print(tabulate(datasets_descriptions, headers=['Dataset', '#Train Samples', '#Train IDs', '#Train Cameras', 
																	'#Val Samples', '#Val IDs', '#Val Cameras',
																	'#Query Samples', '#Query IDs', '#Query Cameras',
																	'#Gallery Samples', '#Gallery IDs', '#Gallery Cameras']))

		return train_images, val_images, queries_images, gallery_images

	elif dataset_name == "Veri":

		train_images = load_set_from_veri("/home/dejavu/datasets/ReID_Datasets/VeRi/image_train")
		gallery_images = load_set_from_veri("/home/dejavu/datasets/ReID_Datasets/VeRi/image_test")
		queries_images = load_set_from_veri("/home/dejavu/datasets/ReID_Datasets/VeRi/image_query")

	elif dataset_name == "Veri-Wild":

		base_name = "/home/dejavu/datasets/ReID_Datasets/VeRI-Wild/images"

		train_images = load_set_from_veriWild("/home/dejavu/datasets/ReID_Datasets/VeRI-Wild/train_test_split/train_list_start0.txt", base_name)
		gallery_images = load_set_from_veriWild("/home/dejavu/datasets/ReID_Datasets/VeRI-Wild/train_test_split/test_10000_id.txt", base_name)
		queries_images = load_set_from_veriWild("/home/dejavu/datasets/ReID_Datasets/VeRI-Wild/train_test_split/test_10000_id_query.txt", base_name)

	elif dataset_name == "PRCC":

		train_images = load_general_set("/home/dejavu/datasets/ReID_Datasets/prcc/rgb/train.txt")
		val_images = load_general_set("/home/dejavu/datasets/ReID_Datasets/prcc/rgb/val.txt")
		print(colored("There is validation which is not being used!", "yellow"))

		num_splits = 10
		gallery_images = []

		for gal_idx in range(1,num_splits+1):
			gallery_split = load_general_set("/home/dejavu/datasets/ReID_Datasets/prcc/rgb/test/gallery_%d.txt" % gal_idx)
			gallery_images.append(gallery_split)
			
		queries_images = []
		queries_images_B = load_general_set("/home/dejavu/datasets/ReID_Datasets/prcc/rgb/test/query_B.txt")
		queries_images_C = load_general_set("/home/dejavu/datasets/ReID_Datasets/prcc/rgb/test/query_C.txt")
		queries_images.append(queries_images_B)
		queries_images.append(queries_images_C)
		queries_images.append(np.concatenate((queries_images_B, queries_images_C), axis=0))

	elif dataset_name == "VC-Clothes":

		VC_train_images = load_general_set("/home/dejavu/datasets/ReID_Datasets/ClothesChanging/VC-Clothes/train_file.txt")
		VC_gallery_images = load_general_set("/home/dejavu/datasets/ReID_Datasets/ClothesChanging/VC-Clothes/gallery_file.txt")
		VC_query_images = load_general_set("/home/dejavu/datasets/ReID_Datasets/ClothesChanging/VC-Clothes/query_file.txt")

		Real_gallery_images = load_general_set("/home/dejavu/datasets/ReID_Datasets/ClothesChanging/Real28/gallery_file.txt")
		Real_query_images = load_general_set("/home/dejavu/datasets/ReID_Datasets/ClothesChanging/Real28/query_file.txt")

		train_images = VC_train_images
		gallery_images = [VC_gallery_images, Real_gallery_images]
		queries_images = [VC_query_images, Real_query_images]

		print(VC_train_images.shape, VC_gallery_images.shape, VC_query_images.shape, Real_gallery_images.shape, Real_query_images.shape)

	elif dataset_name == "DeepChange":

		base_name = "/scratch/datasets/ReID_datasets/DeepChange"
		train_images = load_set_from_DeepChange(base_name, "train-set-bbox.txt", "train-set")
		#val_images = load_general_set("/home/dejavu/datasets/ReID_Datasets/DeepChange/train-set-bbox.txt")
		#print(colored("There is validation which is not being used!", "yellow"))
		
		# It will return three galleris: first one considering cameras for matching, second one considering days for matching
		# and the last one considering both cameras and days. 
		
		gallery_images = load_set_from_DeepChange(base_name, "test-set-gallery-bbox.txt", "test-set-gallery")
		queries_images = load_set_from_DeepChange(base_name, "test-set-query-bbox.txt", "test-set-query")

		#gallery_images = load_set_from_DeepChange(base_name, "val-set-gallery-bbox.txt", "val-set-gallery")
		#queries_images = load_set_from_DeepChange(base_name, "val-set-query-bbox.txt", "val-set-query")

	elif dataset_name == "Celeb-ReID":

		train_images = load_general_set("/scratch/datasets/ReID_datasets/Celeb-reID/Celeb-reID/train_file.txt")
		gallery_images = load_general_set("/scratch/datasets/ReID_datasets/Celeb-reID/Celeb-reID/gallery_file.txt")
		queries_images = load_general_set("/scratch/datasets/ReID_datasets/Celeb-reID/Celeb-reID/query_file.txt")

	elif dataset_name == "ImageNet":

		train_images = load_general_set("/home/dejavu/datasets/ReID_Datasets/ImageNet/train_file.txt")
		val_images = load_general_set("/home/dejavu/datasets/ReID_Datasets/ImageNet/val_file.txt")
		print("There is validation which is not being used!")

		num_splits = 10
		gallery_images = []
		queries_images = []

		for gal_idx in range(1,num_splits+1):
			gallery_split = load_general_set("/home/dejavu/datasets/ReID_Datasets/ImageNet/gallery_file%d.txt" % gal_idx)
			gallery_images.append(gallery_split)

			query_split = load_general_set("/home/dejavu/datasets/ReID_Datasets/ImageNet/query_file%d.txt" % gal_idx)
			queries_images.append(query_split)
			
	
	return train_images, gallery_images, queries_images

def load_text_dataset(base_dir):

	training_txtfile = open("training_tweets.txt","r")
	query_txtfile = open("query_tweets.txt","r")
	gallery_txtfile = open("gallery_tweets.txt","r")

	train_text = []
	query_text = []
	gallery_text = []

	for sample in training_txtfile.readlines():
		author_id, tweet_id = sample[:-1].split(" ")
		full_path = os.path.join(base_dir, author_id, "tweets.json")
		train_text.append([full_path, author_id, tweet_id])

	train_text = np.array(train_text)

	for sample in query_txtfile.readlines():
		author_id, tweet_id = sample[:-1].split(" ")
		full_path = os.path.join(base_dir, author_id, "tweets.json")
		query_text.append([full_path, author_id, tweet_id])

	for sample in gallery_txtfile.readlines():
		author_id, tweet_id = sample[:-1].split(" ")
		full_path = os.path.join(base_dir, author_id, "tweets.json")
		gallery_text.append([full_path, author_id, tweet_id])

	query_text = np.array(query_text)
	gallery_text = np.array(gallery_text)

	return train_text, gallery_text, query_text


def load_multiple_datasets(targets_names):

	# Note that we will concatenate all training images but NOT the gallery and query images,
	# since we would like to have an unique training set but separated evaluations

	train_images_target = []
	gallery_images_target = []
	queries_images_target = []

	for target in targets_names:

		#print("Dataset %s has:" % target)
		train_images, gallery_images, queries_images = load_dataset(target)
		
		#print(train_images.shape)
		#print(gallery_images.shape)
		#print(queries_images.shape)
		train_images_target.append(train_images)
		gallery_images_target.append(gallery_images)
		queries_images_target.append(queries_images)

	#train_images_target = np.concatenate(train_images_target, axis=0)
	#print("Total train data size:", train_images_target.shape)

	return train_images_target, gallery_images_target, queries_images_target

def get_dataset_samples_and_statistics(targets_names):
	#print(targets_names)
	train_images_target, gallery_images_target, queries_images_target = load_multiple_datasets(targets_names)
	
	datasets_descriptions = []
	for target_idx in np.arange(len(targets_names)):	
		target = targets_names[target_idx]

		# Getting training set information
		num_train_samples = len(train_images_target[target_idx])
		num_train_ids = len(np.unique(train_images_target[target_idx][:,1]))
		num_train_cameras = len(np.unique(train_images_target[target_idx][:,2]))


		# Getting gallery set information
		if type(gallery_images_target[target_idx]) != list:
			num_gallery_samples = len(gallery_images_target[target_idx])
			num_gallery_ids = len(np.unique(gallery_images_target[target_idx][:,1]))
			num_gallery_cameras = len(np.unique(gallery_images_target[target_idx][:,2]))
		else:
			number_of_galleries = len(gallery_images_target[target_idx])
			for gal_idx in range(number_of_galleries):
				num_gallery_samples = len(gallery_images_target[target_idx][gal_idx])
				num_gallery_ids = len(np.unique(gallery_images_target[target_idx][gal_idx][:,1]))
				num_gallery_cameras = len(np.unique(gallery_images_target[target_idx][gal_idx][:,2]))

				datasets_descriptions.append([target + str(gal_idx), num_train_samples, num_train_ids, num_train_cameras, 
												num_gallery_samples, num_gallery_ids, num_gallery_cameras,
												"FI", "FI", "FI"])

		# Getting query (probe) set information
		if type(queries_images_target[target_idx]) != list:
			num_queries_samples = len(queries_images_target[target_idx])
			num_queries_ids = len(np.unique(queries_images_target[target_idx][:,1]))
			num_queries_cameras = len(np.unique(queries_images_target[target_idx][:,2]))

			datasets_descriptions.append([target, num_train_samples, num_train_ids, num_train_cameras, 
													num_gallery_samples, num_gallery_ids, num_gallery_cameras,
													num_queries_samples, num_queries_ids, num_queries_cameras])

		else:
			number_of_queries = len(queries_images_target[target_idx])
			for query_idx in range(number_of_queries):
				num_queries_samples = len(queries_images_target[target_idx][query_idx])
				num_queries_ids = len(np.unique(queries_images_target[target_idx][query_idx][:,1]))
				num_queries_cameras = len(np.unique(queries_images_target[target_idx][query_idx][:,2]))

				datasets_descriptions.append([target + str(query_idx), num_train_samples, num_train_ids, num_train_cameras, 
												"EI", "EI", "EI",
												num_queries_samples, num_queries_ids, num_queries_cameras])

	train_images_target = np.concatenate(train_images_target, axis=0)

	print(tabulate(datasets_descriptions, headers=['Dataset', '#Train Samples', '#Train IDs', '#Train Cameras', 
																'#Gallery Samples', '#Gallery IDs', '#Gallery Cameras',
																'#Query Samples', '#Query IDs', '#Query Cameras']))	

	return train_images_target, gallery_images_target, queries_images_target


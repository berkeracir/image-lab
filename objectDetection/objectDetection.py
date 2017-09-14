# Some parts of this code is adapted from Priyanka Dwivedi
# https://github.com/priya-dwivedi/Deep-Learning/blob/master/Object_Detection_Tensorflow_API.ipynb

import os
import inspect
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
import math
import copy
from matplotlib import pyplot as plt

from PIL import Image
import download_and_extract as dae
import frame_extract as fe

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def model_name(model):
	if model == "mobilenet":
		return dae.URL_mobilenet.split('/')[-1]
	elif model == "inception":
		return dae.URL_inception.split('/')[-1]
	elif model == "rfcn_resnet":
		return dae.URL_rfcn_resnet.split('/')[-1]
	elif model == "rcnn_resnet":
		return dae.URL_rcnn_resnet.split('/')[-1]
	elif model == "rcnn_inception":
		return dae.URL_rcnn_inception.split('/')[-1]
	else:
		print "Error: Unexpected Model"
		sys.exit()

CURR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
INPUT_DIR = os.path.join(CURR_PATH,"frames")
annotation_dir = os.path.join(CURR_PATH,"annotations")


# Choose the model: "mobilenet", "inception", "rfcn_resnet", "rcnn_resnet", "rcnn_inception"
MODEL = "mobilenet"

# Download and extract the model
dae.download_and_extract(MODEL)

# Extract the frames of the videos
fe.frame_extract(CURR_PATH)

MODEL_NAME = model_name(MODEL).split('.')[0]
PATH_TO_CKPT = os.path.join(CURR_PATH,"data", MODEL_NAME, "frozen_inference_graph.pb") 
PATH_TO_LABELS = os.path.join(CURR_PATH, "data", MODEL_NAME, "graph.pbtxt") 

NUM_CLASSES = 90

label_map = label_map_util.load_labelmap("/path/to/tensorflow/models/object_detection/data/mscoco_label_map.pbtxt")
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# cmpr function takes the necessary arguments of a previous object and compares with the new detected object
# to decide if they are the same or not

def cmpr(className,sqBox,annotation,im_width, im_height):

	if(className != annotation[9]):
		return False

	else:
		x_margin = math.ceil(math.log(annotation[3]-annotation[1]))
		y_margin = math.ceil(math.log(annotation[4]-annotation[2]))

		if abs(int(sqBox[1]*im_width) - annotation[1]) > x_margin and abs(int(sqBox[0]*im_height) - annotation[2]) > y_margin and abs(int(sqBox[3]*im_width) - annotation[3]) > x_margin:
			return False

		elif abs(int(sqBox[1]*im_width) - annotation[1]) > x_margin and abs(int(sqBox[0]*im_height) - annotation[2]) > y_margin and abs(int(sqBox[2]*im_height) - annotation[4]) > y_margin:
			return False

		elif abs(int(sqBox[1]*im_width) - annotation[1]) > x_margin and abs(int(sqBox[2]*im_height) - annotation[4]) > y_margin and abs(int(sqBox[3]*im_width) - annotation[3]) > x_margin:
			return False

		elif abs(int(sqBox[2]*im_height) - annotation[4]) > y_margin and abs(int(sqBox[0]*im_height) - annotation[2]) > y_margin and abs(int(sqBox[3]*im_width) - annotation[3]) > x_margin:
			return False

		else:
			return True

def detect_objects(image_np, sess, detection_graph, image_path,videoName):
	global annotations
	global ID
	global annotation_dir
	new_annotations = []


	image_np_expanded = np.expand_dims(image_np, axis=0)
	image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
	boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
	scores = detection_graph.get_tensor_by_name("detection_scores:0")
	classes = detection_graph.get_tensor_by_name("detection_classes:0")
	num_detections = detection_graph.get_tensor_by_name("num_detections:0")

	(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor:image_np_expanded})

	vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),
		np.squeeze(classes).astype(np.int32),    
		np.squeeze(scores),    
		category_index, 
		use_normalized_coordinates=True, 
		line_thickness=8)


	frame = image_path.split('/')[-1].split('.')[0]
	im_width = image_np.shape[1]
	im_height = image_np.shape[0]
	sqBoxes = np.squeeze(boxes)
	sqScores = np.squeeze(scores)

	if not os.path.exists(annotation_dir):
		os.makedirs(annotation_dir)

	for i in range(0,(sqBoxes).shape[0]):

		if sqScores[i] > 0.5:   # scores threshold
			sqClasses = np.squeeze(classes).astype(np.int32)
			if sqClasses.all() in category_index.keys():

				# Below, the detected object is written to the annotation file with the desired format compatible with vatic

				sqBox = sqBoxes[i]
				annotation_dir = os.path.join(CURR_PATH,"annotations")
				file = open(os.path.join(annotation_dir,videoName+".txt"),"a")
				written = False
				if annotations == []:                        # Add first frame's detected objects
					file.write(str(ID) + ' ')
					file.write(str(int(sqBox[1]*im_width)) + ' ' + str(int(sqBox[0]*im_height)) +' ')
					file.write(str(int(sqBox[3]*im_width)) + ' ' + str(int(sqBox[2]*im_height)) + ' ')
					file.write(frame + ' 0 0 0 ')
					file.write(category_index[sqClasses[i]]['name'] + '\n')
					new_annotations.append([ID,int(sqBox[1]*im_width),int(sqBox[0]*im_height),
						int(sqBox[3]*im_width),int(sqBox[2]*im_height),frame,0,0,0,category_index[sqClasses[i]]['name']])
					ID = ID + 1

				else:
					for m in range(len(annotations)):        # Check whether the object exists in previous frame or not by comparing w/ all objects in previous frame
						j = annotations[m]
						className = category_index[sqClasses[i]]['name']
						sameObject = cmpr(className,sqBox,j,im_width,im_height)

						if sameObject:
							file.write(str(j[0]) + ' ')
							file.write(str(int(sqBox[1]*im_width)) + ' ' + str(int(sqBox[0]*im_height)) +' ')
							file.write(str(int(sqBox[3]*im_width)) + ' ' + str(int(sqBox[2]*im_height)) + ' ')
							file.write(frame)
							file.write(' 0 0 0 ')
							file.write(str(j[9]) + '\n')
							new_annotations.append([j[0],int(sqBox[1]*im_width),int(sqBox[0]*im_height),
								int(sqBox[3]*im_width),int(sqBox[2]*im_height),frame,0,0,0,j[9]])
							annotations.remove(j)
							written = True
							break

					if (not written):						# If the object doesn't exist, add with a new ID
						file.write(str(ID) + ' ')
						file.write(str(int(sqBox[1]*im_width)) + ' ' + str(int(sqBox[0]*im_height)) +' ')
						file.write(str(int(sqBox[3]*im_width)) + ' ' + str(int(sqBox[2]*im_height)) + ' ')
						file.write(frame + ' 0 0 0 ')
						file.write('' + category_index[sqClasses[i]]['name'] + '\n')
						new_annotations.append([ID,int(sqBox[1]*im_width),int(sqBox[0]*im_height),
							int(sqBox[3]*im_width),int(sqBox[2]*im_height),frame,0,0,0,category_index[sqClasses[i]]['name']])
						ID = ID + 1
						written = True
				file.close()

	while(not annotations == []):
		annotations.remove(annotations[0])

	annotations = copy.deepcopy(new_annotations)

	while(not new_annotations == []):
		new_annotations.remove(new_annotations[0])

	return image_np 

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

detection_graph = tf.Graph()

with detection_graph.as_default():
	od_graph_def = tf.GraphDef()

	with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def,name='')


# Detect objects in videos frame by frame

with detection_graph.as_default():

	with tf.Session(graph=detection_graph) as sess:
		dirs = os.listdir(INPUT_DIR)
		for FRAME_DIR in dirs:
			annotations = [] # annotations from prev frame
			ID = 0 # number of the objects
			videoName = FRAME_DIR
			l = len([file for file in os.listdir(os.path.join(INPUT_DIR,FRAME_DIR))])
			FRAME_PATHS = [os.path.join(INPUT_DIR, FRAME_DIR, "{}.jpg".format(i)) for i in range(0,l)]

			for image_path in FRAME_PATHS:
				image = Image.open(image_path)
				image_np = load_image_into_numpy_array(image)
				image_process = detect_objects(image_np, sess, detection_graph,image_path,videoName)

			if (os.path.isfile(os.path.join(CURR_PATH,"annotations",videoName+".txt"))):
				f = open(os.path.join(CURR_PATH,"annotations",videoName+".txt"),"r")
				lines = f.readlines()
				f.close()
				f = open(os.path.join(CURR_PATH,"annotations",videoName+".txt"),"w")

				for i in range(ID):
					for line in lines:
						if line.split(' ')[0] == str(i):
						  f.write(line)
				f.close()


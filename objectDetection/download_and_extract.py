import os
import inspect
import sys
import tarfile

from six.moves import urllib

CURR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

URL_mobilenet = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz"
URL_inception = "http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz"
URL_rfcn_resnet = "http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_11_06_2017.tar.gz"
URL_rcnn_resnet = "http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz"
URL_rcnn_inception = "http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz"

def download_and_extract(model):
	"""Download and extract model tar file."""
	if model == "mobilenet":
		URL = URL_mobilenet
	elif model == "inception":
		URL = URL_inception
	elif model == "rfcn_resnet":
		URL = URL_rfcn_resnet
	elif model == "rcnn_resnet":
		URL = URL_rcnn_resnet
	elif model == "rcnn_inception":
		URL = URL_rcnn_inception
	else:
		print "Error:Unexpected Model"
		sys.exit()
	DEST_PATH = os.path.join(CURR_PATH,"data")
	if not os.path.exists(DEST_PATH):
		os.makedirs(DEST_PATH)
	filename = URL.split('/')[-1]
	filepath = os.path.join(DEST_PATH, filename)
	if not os.path.exists(filepath):
		def download_progress(count, block_size, total_size):
			sys.stdout.write("\r>> Downloading: %s | %.1f%%" % (filename,float(count*block_size)/float(total_size)*100))
			sys.stdout.flush()
		filepath, _ = urllib.request.urlretrieve(URL, filepath, download_progress)
		print()
		statinfo = os.stat(filepath)
		print "Successfully downloaded!", filename, statinfo.st_size, "bytes."
	tarfile.open(filepath, "r:*").extractall(path=DEST_PATH)

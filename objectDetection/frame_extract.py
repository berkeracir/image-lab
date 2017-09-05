import cv2
import os
import inspect

def frame_extract(CURR_PATH):
	path, dirs, files = os.walk(os.path.join(CURR_PATH, "videos")).next()
	for i in files:
		captured_video = cv2.VideoCapture(os.path.join(path, i))
		success, image = captured_video.read()
		count = 0
		directory = os.path.join(CURR_PATH, "frames", i.split('.')[0])
		if not os.path.exists(directory):
			os.makedirs(directory)	
		while success:
			cv2.imwrite(directory + '/%d.jpg' % count, image)
			success, image = captured_video.read()
			count += 1

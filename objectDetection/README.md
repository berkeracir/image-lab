# Object Detection with Google Tensorflow Object Detection API and Annotation

1. Follow the installation process on [Tensorflow's Installation Guide](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md) until Protobuf Compilation(included).
2. Check where the tensorflow packages are installed and adapt the path in "objectDetection.py", line 57 (`label_map = label_map_util.load_labelmap("/path/to/tensorflow/models/object_detection/data/mscoco_label_map.pbtxt")`) to your tensorflow directory.
3. Decide which detection model to use, these models are:
  * Single Shot Multibox Detector (SSD) with MobileNet,
  * SSD with Inception V2,
  * Region-Based Fully Convolutional Networks (R-FCN) with Resnet 101,
  * Faster RCNN with Resnet 101,
  * Faster RCNN with Inception Resnet v2.

	More information about speed/accuracy trade-offs for modern convolutional object detectors can be found [here](https://arxiv.org/pdf/1611.10012v3.pdf).
	Note: Default model is *Single Shot Multibox Detector (SSD) with MobileNet*, it can be changed in "objectDetection.py", line 42-43.

4. Add Libraries to PYTONPATH, this command needs to run from every new terminal you start:
`export PYTHONPATH=$PYTHONPATH:/path/to/tensorflow/models/`
5. Run objectDetection.py
6. Annotations of each video will be in `../objectDetection/annotations/` with the name of video.

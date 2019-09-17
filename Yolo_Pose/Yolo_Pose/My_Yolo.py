from darkflow.net.build import TFNet
import cv2
import numpy as np


class Yolo:
	def __init__(self):
		self.Yolo_options ={"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
		self.tfnet = TFNet(self.Yolo_options)
		self.Yolo_class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
              'dog', 'horse', 'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor']
		self.Yolo_num_class = len(self.Yolo_class_names)
		self.Yolo_class_colors = []

		for i in range(0,self.Yolo_num_class):
			self.hue = 255*i/self.Yolo_num_class
			self.col = np.zeros((1,1,3)).astype("uint8")
			self.col[0][0][0] =self.hue
			self.col[0][0][1] =128
			self.col[0][0][2] =255
			self.cvcol = cv2.cvtColor(self.col, cv2.COLOR_HSV2BGR)
			self.col = (int(self.cvcol[0][0][0]), int(self.cvcol[0][0][1]), int(self.cvcol[0][0][2]))
			self.Yolo_class_colors.append(self.col) 

	def Yolo(self):
		self.Yolo_options ={"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
		self.tfnet = TFNet(self.Yolo_options)
		self.Yolo_class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
              'dog', 'horse', 'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor']
		self.Yolo_num_class = len(self.Yolo_class_names)
		self.Yolo_class_colors = []

		for i in range(0,self.Yolo_num_class):
			self.hue = 255*i/self.Yolo_num_class
			self.col = np.zeros((1,1,3)).astype("uint8")
			self.col[0][0][0] =self.hue
			self.col[0][0][1] =128
			self.col[0][0][2] =255
			self.cvcol = cv2.cvtColor(self.col, cv2.COLOR_HSV2BGR)
			self.col = (int(self.cvcol[0][0][0]), int(self.cvcol[0][0][1]), int(self.cvcol[0][0][2]))
			self.Yolo_class_colors.append(self.col) 
			self.Yolo_class_colors.append(self.col) 

		return self.Yolo_options,self.tfnet,self.Yolo_class_names,self.Yolo_num_class,self.Yolo_class_colors

	


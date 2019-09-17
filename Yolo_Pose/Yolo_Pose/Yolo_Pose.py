import pyopenpose as op
from darkflow.net.build import TFNet
import cv2
import numpy as np
import sys
from sys import platform
import argparse
import os
import pprint


class My_Yolo_Pose():
	
	def __init__(self):
		#######Yoloのパラメータ初期化####################
		self.Yolo_options ={"model": "cfg/yolo.cfg", "load": "../../bin/yolo.weights", "threshold": 0.1}
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
		###################################
		########OpenPose###################
		self.Now_path = os.path.dirname(os.path.realpath(__file__))
		#self.dir_path = os.path.dirname("C://tools//openpose//examples//tutorial_api_python//openpose.py")
		self.dir_path = os.path.dirname("C:\tools\openpose\examples\tutorial_api_python\openpose.py")
		self.model_path = "C:/tools/openpose/models/"

		try:
			if platform =="win32":
				sys.path.append(self.dir_path + '/../../python/openpose/Release');
				
				os.environ['PATH']  = os.environ['PATH'] + ';' + self.dir_path + '/../../x64/Release;' +  self.dir_path + '/../../bin;'
				import pyopenpose as op
				
			else:
				sys.path.append('../../python');
				from openpose import pyopenpose as op
		except ImportError as e:
			print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
			raise e

		# Flags
		self.parser = argparse.ArgumentParser()
		#self.parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
		self.parser.add_argument("--image_path", default="C:\tools\openpose\examples\media\COCO_val2014_000000000564.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
		self.args = self.parser.parse_known_args()

		# Custom Params (refer to include/openpose/flags.hpp for more parameters)
		self.params = dict()
		self.params["model_folder"] = self.model_path
		#############################################################
		


	def start_openpose(self,cap,opWrapper,images):
		#############OpenPose処理###################
		datum = op.Datum()
		datum.cvInputData = images
		opWrapper.emplaceAndPop([datum])

		cv2.putText(datum.cvOutputData,"OpenPose using Python-OpenCV",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
		cv2.imshow('Human Pose Estimation',datum.cvOutputData)

			##############################################

	def start_yolo(self,cap,image):
		###############Yolo処理#######################
		result = self.tfnet.return_predict(image)
		for item in result:
			tlx = item['topleft']['x']
			tly = item['topleft']['y']
			brx = item['bottomright']['x']
			bry = item['bottomright']['y']
			label = item['label']
			conf = item['confidence']

			if conf > 0.6:
				for i in self.Yolo_class_names:
					if label == id:
						class_num = self.Yolo_class_names.index(i)
						break

				#cv2.rectangle(image, (tlx, tly), (brx, bry), self.Yolo_class_colors[class_num], 2)
				text = label + " " + ('%.2f' % conf)  
				#cv2.rectangle(image, (tlx, tly - 15), (tlx + 100, tly + 5), self.Yolo_class_colors[class_num], -1)
				cv2.putText(image, text, (tlx, tly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
		cv2.imshow("Show FLAME Image", image)

	def start(self):
		opWrapper = op.WrapperPython()
		opWrapper.configure(self.params)
		opWrapper.start()
		cap = cv2.VideoCapture(0)
		#cap = cv2.VideoCapture("sample2.mp4")
		print(self.Yolo_options)
		while True:
			ret,image = cap.read()
			images = image.copy()

			self.start_openpose(cap,opWrapper,image)
			self.start_yolo(cap,images)		
			
			key = cv2.waitKey(1)
			if key==ord('q'):
				break
		stream.release()
		cv2.destroyAllWindows()



M = My_Yolo_Pose()
M.start()

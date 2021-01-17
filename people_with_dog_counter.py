from monitoring.monitorCenterObj import MonitorCenterObj, MonitorObject
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import getMetadata as meta
import datetime

#parse the arguments:
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True,
                help="base path to YOLO directory")
ap.add_argument("-i", "--input", type=str,
                help="path to input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to output video file")
ap.add_argument("-s", "--skip-frames", type=int, default=100,
                help="# of skip frames between detections")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applyong non-maxima suppression")
ap.add_argument("-st", "--startTime", type=str,
                help="start video time, format= Y-m-d_H:M:S")
ap.add_argument("-d", "--countDogs", type=int, default=0,
                help="# calculate how many dogs are walking from 8 to 10 in the morning")

args = vars(ap.parse_args())



#get labels from coco dataset:
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
'''
Категории, которые имеет COCO daset (80 classes):
['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
  'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
   'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

'''

# YOLO weights and model configuration:
weightsPath = os.path.sep.join([args["yolo"], "yolov4.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov4.cfg"])

# load  YOLO v4
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layers_names = net.getLayerNames()
layers_names = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

dur_ms=meta.get_duration_ms(args["input"])
print("[INFO] time (ms): ", dur_ms)

# instantiate centroid tracker
mc = MonitorCenterObj()

# initialize a list to store dlib correlation trackers
trackers = []

# map each unique object ID to a TrackableObject
trackableObjects = {}
boxes_for_traker=[]
totalFrames = 0
totalDown = 0
totalUp = 0

if args["countDogs"]:
	mcDog = MonitorCenterObj()
	trackersDog = []
	trackableObjectsDog = {}
	trackersPerson = []
	trackableObjectsPerson = {}
	total = 0
	flagDog = set()
	totalFlagDog = set()
	flagPerson = set()
	totalFlagPerson = set()
	dogID = LABELS.index('dog')
	personID = LABELS.index('person')
	tmp_by_dog = []

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
	countFrame = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(countFrame))
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	countFrame = -1

currentTime=datetime.datetime.strptime(args["startTime"], "%Y-%m-%d_%H:%M:%S")
speed=datetime.timedelta(milliseconds=int(countFrame/dur_ms))


while True:
	(grabbed, frame) = vs.read()
	if frame is None:
		break
	# check view
	frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
	# convert the frame from BGR to RGB for dlib
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

	if args["countDogs"]: rectsDog = []
	rects = []
	if totalFrames % args["skip_frames"] == 0:
		trackers = []
		trackersDog = []
		blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
		net.setInput(blob)
		layerOutputs = net.forward(layers_names)
		boxes = []
		confidences = []
		classIDs = []

		for output in layerOutputs:
			for detection in output:
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				if confidence > args["confidence"]:
					if args["countDogs"]:
						if LABELS[classID] != "person" and LABELS[classID] != "dog":
							continue
					else:
						if LABELS[classID] != "person":
							continue
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
		boxes_for_traker = []
		tmp_by_dog = []
		if len(idxs) > 0:
			if args["countDogs"] and ( 8 <= currentTime.hour <10):
				for i in idxs.flatten():
					# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					if classIDs[i] == dogID:
						tmp_by_dog.append((x, y, x + w, y + h))
					if classIDs[i] == personID:
						boxes_for_traker.append((x, y, x + w, y + h))

			else:
				for i in idxs.flatten():
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					boxes_for_traker.append((x, y, x + w, y + h))

		bt = boxes_for_traker
		for counter in bt:
			tracker = dlib.correlation_tracker()
			rect = dlib.rectangle(counter[0], counter[1], counter[2], counter[3])
			tracker.start_track(rgb, rect)
			trackers.append(tracker)
		if args["countDogs"]:
			by = tmp_by_dog
			for by_counter in by:
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(by_counter[0], by_counter[1], by_counter[2], by_counter[3])
				tracker.start_track(rgb, rect)
				trackersDog.append(tracker)
	else:
		for tracker in trackers:
			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()
			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())
			rects.append((startX, startY, endX, endY))
		if args["countDogs"]:
			for tracker in trackersDog:
				tracker.update(rgb)
				pos = tracker.get_position()
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())
				rectsDog.append((startX, startY, endX, endY))

	objects = mc.update(rects)
	if args["countDogs"]: dogs = mcDog.update(rectsDog)
	if args["countDogs"]:
		for (objectID, centroid) in dogs.items():
			to = trackableObjectsDog.get(objectID, None)
			if to is None:
				to = MonitorObject(objectID, centroid)
			else:
				flagDog.add(objectID)

			trackableObjectsDog[objectID] = to

			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
	for (objectID, centroid) in objects.items():
		to = trackableObjects.get(objectID, None)
		# if there is no existing trackable object, create one
		if to is None:
			to = MonitorObject(objectID, centroid)
			if args["countDogs"]:
				totalFlagDog.update(flagDog)
				flagDog = set()
				flagPerson = set()
		else:
			if args["countDogs"]:
				flagPerson.add(objectID)
				totalFlagPerson.add(objectID)
				total += len(flagDog.difference(totalFlagDog))
				totalFlagDog.update(flagDog)
			y = [c[1] for c in to.centroids]
			#print(np.mean(y),centroid[1],centroid)
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)
			if not to.counted:
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					to.counted = True
				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					to.counted = True
		trackableObjects[objectID] = to
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
	result = [
		("Up", totalUp),
		("Down", totalDown),
		("Dogs", total if args["countDogs"] else None)
	]

	for (i, (k, v)) in enumerate(result):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	if writer is not None:
		writer.write(frame)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	totalFrames += 1

if writer is not None:
	writer.release()


if not args.get("input", False):
	vs.stop()
else:
	vs.release()
cv2.destroyAllWindows()


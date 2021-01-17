import argparse
import dlib
import cv2
import os
import numpy as np
from monitoring.monitorCenterObj import MonitorCenterObj, MonitorObject


ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-i", "--input", type=str, help="path to input video file")
ap.add_argument("-o", "--output", type=str, help="path to output video file")
ap.add_argument("-s", "--skip-frames", type=int, default=100, help="# of skip frames between detections")
ap.add_argument("-c", "--confidence", type=float, default=0.8, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
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
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (W, H), True)

    rects = []
    if totalFrames % args["skip_frames"] == 0:
        trackers = []
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
        if len(idxs) > 0:
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

    objects = mc.update(rects)
    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)
        # if there is no existing trackable object, create one
        if to is None:
            to = MonitorObject(objectID, centroid)
        else:
            y = [c[1] for c in to.centroids]
            #print(np.mean(y), centroid[0], centroid)
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    result = [
        ("Up", totalUp),
        ("Down", totalDown)
    ]

    for (i, (k, v)) in enumerate(result):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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

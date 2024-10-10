# implementation of centroid tracker
#and correlation object tracking counter in dlib

import numpy as np
import time
import cv2
import dlib
import argparse
import torch
import torchvision

# for centroid Tracking
from scipy.spatial import distance as dist
from collections import OrderedDict

# not sure if I will use
from imutils.video import VideoStream
from imutils.video import FPS

# Storing object ID, previous centroids location, and boolean for counted objects
class trackedobject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False

#module for centroid Tracking
class centroidtracker:
    def __init__(self, maxdisappear = 50, maxdistance = 50):
        #initialize next objectID
        self.nextobjectID = 0 #initialize objectID as zero from runtime
        #initialize Ordered dictionary
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        #initalize input variables
        self.maxdistance = maxdistance
        self.maxdisappear = maxdisappear
    def register(self, centroid):

        self.objects[self.nextobjectID] = centroid # use keys with centroid position
        self.disappeared[self.nextobjectID] = 0 # use keys with boolean 0,1
        self.nextobjectID += 1
    def deregister(self, objectID):

        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # loop for deregistering objects that was not found for the duration of the number of loops run
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1 #addition from zero on every loop if not found and zeroed
                if self.disappeared[objectID] > self.maxdisappear: # number of frame loops addition
                    self.deregister(objectID)
            return self.objects #??? why return early?
        # getting centroid position by calculating
        centroids = np.zeros((len(rects), 2), dtype = "int") # zero 4x2 matrix for what?
        for (i, (x, y, xmax, ymax)) in enumerate(rects):
            xcenter = int((x + xmax)/2.0)
            ycenter = int((y + ymax)/2.0)
            centroids[i] = (xcenter, ycenter) # register it into the 4x2 zero matrix, does this mean it only accepts 4 centroids coordinates?
        #registering the then calculated centroid when the dictionary registry is None (initialization), this uses the range loop of the zero matrix shape created earlier
        if len(self.objects) ==  0:
            for i in range(0, len(centroids)):
                self.register(centroids[i])
        else: #if there is objects in the registry
            #seperating two list of values from the dictionary registry
            registeredID = list(self.objects.keys())
            registered_centroids = list(self.objects.values())

            # calculating euclidean distance
            euclid_distance = dist.cdist(np.array(registered_centroids), centroids)

            # used for performing matching the previous centroid to the present centroid
            rows = euclid_distance.min(axis = 1).argsort() # sort the row indexes based on their minimum distance values
            cols = euclid_distance.argmin(axis=1)[rows] # sorting the smallest distance value in each column using the previously computed index list

            #keep track of examined rows and columns
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: # ignoring examined rows or columns
                    continue
                if euclid_distance[row, col] > self.maxdistance: # cancel matching since object is too far away
                    continue

                # match he old centroid to the new one
                objectID = registeredID[row] # align objectID to array of centroids
                self.objects[objectID] = centroids[col] # update the centroid with the lowest distance value in that row
                self.disappeared[objectID] = 0

                # examined column and row indexes
                usedRows.add(row)
                usedCols.add(col)

            # separate unused rows and columns from the used ones
            unusedRows = set(range(0, euclid_distance.shape[0])).difference(usedRows)
            unusedCols = set(range(0, euclid_distance.shape[1])).difference(usedCols)

            # if the number of registered centroids is equal or greater than the number of new centroids, need to check if centroids disappeared
            if euclid_distance.shape[0] >= euclid_distance.shape[1]:
                for row in unusedRows:

                    objectID = registeredID[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxdisappear:
                        self.deregister(objectID)
            # if input centroids is larger than the existing centroids, register them
            else:
                for col in unusedCols:
                    self.register(centroids[col])

        return self.objects



def cumulative_count(input_frame, boxes, classes, scores):
    centroid_tracker = centroidtracker(maxdisappear = 40, maxdistance = 50)
    trackers = []
    trackableobjects = {}
    skip_frames = 30
    totalFrames = 0
    fps = FPS().start()

    while True:
        if totalFrames % skip_frames == 0:
            tracker = dlib.correlation_tracker()
            (x, y, xmax, ymax) = boxes
            rect = dlib.rectangle(x, y, xmax, ymax)
            tracker.start_track(input_frame, rect)
            trackers.append(tracker)
    else:
        for tracker in trackers:
            tracker.update(input_frame)
            pos =  tracker.get_position()
            x = int(pos.left())
            y = int(pos.top())
            xmax = int(pos.right())
            ymax = int(pos.bottom())

            rects.append((x, y, xmax, ymax))

    objects  = centroid_tracker.update(rects)

    for (objectID, centroid) in objects.items():
        to = trackableobjects.get(objectID, None)

        if to is None:
            to = trackableobject(objectID, centroid)

        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            if not to.counted:
                total += 1
                to.counted = True
        trackableobject[objectID] = to
    info = [
        ("total", total)
    ]
    for (i, (k,v)) in enumerate(info):
        text = "{}: {}".format(k,v)
        cv.putText(input_frame, txt, (416, 416), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    totalFrames += 1
    fps.update()
    return total

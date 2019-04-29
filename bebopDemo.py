#!/usr/bin/python3
# import logging
import libwscDrone
import numpy as np
import cv2 as cv
import threading
import time

# global drone

STREAM_WIDTH = 856
STREAM_HEIGHT = 480

def thread0(name):

    print("Thread 0: starting", name)
    for i in range(50000):
      drone.changeCount(0, i)
      # time.sleep(0.1)
    print("Thread 0: finishing", name)

def thread1(name):

  print("Thread 1: starting", name)
  for i in range(50000):
   drone.changeCount(1, i)
   # time.sleep(0.1)
  print("Thread 1: finishing", name)

if __name__ == "__main__":
   print("starting")

   drone = libwscDrone.PyBebop(1)

   print("***************************************")
   frame = np.asarray(drone.getFrameBuffer()).reshape(STREAM_HEIGHT, STREAM_WIDTH, 3)

   for i in range(10000):
      print(frame.shape)
      cv.imshow('frame', frame)
      cv.waitKey(30)

   cv.destroyAllWindows()
   # drone.takeoffDrone()
   # drone.landDrone()
   # drone.landDrone()

   # t0 = threading.Thread(target=thread0, args=(1,))
   # t1 = threading.Thread(target=thread1, args=(1,))
   # t0.start()
   # t1.start()


   print("exited")
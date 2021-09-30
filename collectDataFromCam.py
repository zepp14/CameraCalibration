
import argparse
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
import threading
import time
import socket,os,struct
import cv2
import pickle
from scipy.stats import norm
#commited to Git

import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32
#from tf.transformations import euler_from_quaternion
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import numpy as np


class PositionReader(object):
    def __init__(self, stringName):
        self.PoseTopic = '/vrpn_client_node/'+ stringName+'/pose'
        self._Pose_msg = PoseStamped()
        self.sub_Pose = rospy.Subscriber(self.PoseTopic, PoseStamped, self._pose_callback)
        self.PositionVector = 0
        
    def _pose_callback(self, msg):
        self._Pose_msg = msg

    def get_Pos(self):
        x = self._Pose_msg.pose.position.x
        y = self._Pose_msg.pose.position.y
        z = self._Pose_msg.pose.position.z
        PosVec = [x,y,z]

        return PosVec

    def get_Quaternion(self):
        x = self._Pose_msg.pose.orientation.x
        y = self._Pose_msg.pose.orientation.y
        z = self._Pose_msg.pose.orientation.z
        w = self._Pose_msg.pose.orientation.w
        QVec = [x,y,z,w]

        return QVec

    def get_Euler(self):
        try:
            Q = self.get_Quaternion()
            r = R.from_quat(Q)
            return r.as_euler()
        except:
             Q = [0,0,0,1]
             r = R.from_quat(Q)
             return r.as_euler()
            

    def get_DCM(self):
        try:
            Q = self.get_Quaternion()
            r = R.from_quat(Q)
            return r.as_matrix()
        except:
             Q = [0,0,0,1]
             r = R.from_quat(Q)
             return r.as_matrix()

class SaveFileObj(object):

    def __init__(self):
        self.InitialTime = time.time()
        self.Time_Vector = []
        self.Time_Vector.append(time.time() - self.InitialTime)
        self.ImageData = np.zeros((244, 324, 1))
        self.Position = np.zeros((1,3))
        self.Attitude = np.zeros((1,4))


    def add_Entry(self, TimeVal, Img, X, Q):
        self.Time_Vector.append(TimeVal- self.InitialTime)
        self.ImageData = np.append(self.ImageData, np.reshape(Img,(244, 324, 1)), axis=2)
        self.Position = np.vstack((self.Position,X))
        self.Attitude = np.vstack((self.Attitude,Q))

    def saveObj(self,nameStr):
        print(self.__dict__)
        filehandler = open(nameStr, 'wb')
        pickle.dump(self.__dict__, filehandler)

if __name__ == '__main__':
    rospy.init_node("Position Logger", disable_signals=True)
    PosReader = PositionReader('cf0')
    sv1 = SaveFileObj()
    img_white = 100*np.ones((244, 324))
    try:
        while not rospy.is_shutdown():

            Pos = PosReader.get_Pos()
            Q = PosReader.get_Quaternion()
            
            DCM = PosReader.get_DCM()
            print(DCM )
            sv1.add_Entry(time.time(), img_white, Pos, Q)
            time.sleep(0.5)


            

    except KeyboardInterrupt:
        print(np.shape(sv1.ImageData))
        sv1.saveObj("dataTestSet.pkl")
        cv2.imshow('frameO',sv1.ImageData[:,:,5])
        cv2.waitKey(0)
        pass

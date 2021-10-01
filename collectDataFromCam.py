
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
import lzma

deck_ip = None
deck_port = None

class ImgThread(threading.Thread):
    def __init__(self, callback):
        threading.Thread.__init__(self, daemon=True)
        self._callback = callback

    def run(self):
        print("Connecting to socket on {}:{}...".format(deck_ip, deck_port))
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((deck_ip, deck_port))
        print("Socket connected")

        imgdata = None
        data_buffer = bytearray()

        while(1):
            # Reveive image data from the AI-deck
            data_buffer.extend(client_socket.recv(512))

            # Look for start-of-frame and end-of-frame
            start_idx = data_buffer.find(b"\xff\xd8")
            end_idx = data_buffer.find(b"\xff\xd9")

            # At startup we might get an end before we get the first start, if
            # that is the case then throw away the data before start
            if end_idx > -1 and end_idx < start_idx:
                data_buffer = data_buffer[start_idx:]

            # We have a start and an end of the image in the buffer now
            if start_idx > -1 and end_idx > -1 and end_idx > start_idx:
                # Pick out the image to render ...
                imgdata = data_buffer[start_idx:end_idx + 2]
                passon = data_buffer.copy()
                # .. and remove it from the buffer
                data_buffer = data_buffer[end_idx + 2 :]
                try:
                    self._callback(imgdata)
                    # print(data_buffer)
                    # imstr = np.fromstring(str(passon[start_idx:end_idx + 2]), np.uint8)
                    # img_np = cv2.imdecode(imstr, cv2.IMREAD_UNCHANGED)
                    
                    # RGB_VEC = np.reshape(imstr,(79056,3))
                    

                    # Ch1 = np.reshape(RGB_VEC[:,0],(244, 324))
                    # Ch2 = np.reshape(RGB_VEC[:,1],(244, 324))
                    # Ch3 = np.reshape(RGB_VEC[:,2],(244, 324))

                    # color_img = cv2.merge((Ch1,Ch2,Ch3))
                    # gray_img =  Ch1
        
                    # cv2.imshow('frame1', gray_img)
                except gi.repository.GLib.Error:
                    print("Error rendering image")




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
        self.MarkerPosition = np.zeros((1,3))
        self.MarkerAttitude = np.zeros((1,4))


    def add_Entry(self, TimeVal, Img, X, Q, Xm = [0, 0, 0], Qm = [0,0,0,1]):
        self.Time_Vector.append(TimeVal- self.InitialTime)
        self.ImageData = np.append(self.ImageData, np.reshape(Img,(244, 324, 1)), axis=2)
        self.Position = np.vstack((self.Position,X))
        self.Attitude = np.vstack((self.Attitude,Q))
        self.MarkerPosition = np.vstack((self.MarkerPosition,Xm))
        self.MarkerAttitude = np.vstack((self.MarkerAttitude ,Qm))        

    def saveObj(self,nameStr):
        
        #filehandler = lzma.open(nameStr, 'wb')
        filehandler = lzma.open(nameStr, 'wb')
        pickle.dump(self.__dict__, filehandler)


class FrameViewer(Gtk.Window):

    def __init__(self,sv1, PosReader, MarkerPosReader):
        super(FrameViewer, self).__init__()
        self.frame = None
        self.init_ui()
        self._start = None
        self.set_default_size(374, 294)
        #at state zero 0 the frame is white, sets to black when observation is made
        self.state_flag = 0
        self.startTime = 0
        self.readTime = 0
        self.deltaT = []
        self.num_samples = 0
        self.MAX_samp = 200
        self.grayImg = 0 
        self.sv1=sv1
        self.PosReader=PosReader
        self.MarkerPosReader = MarkerPosReader

    def init_ui(self):
        self.override_background_color(Gtk.StateType.NORMAL, Gdk.RGBA(0, 0, 0, 1))
        self.set_border_width(20)
        self.set_title("Connecting...")
        self.frame = Gtk.Image()
        
        f = Gtk.Fixed()
        f.put(self.frame, 10, 10)
        self.add(f)
        self.connect("destroy", Gtk.main_quit)
        self._thread = ImgThread(self._showframe)
        self._thread.start()





    def _update_image(self, pix):
        self.frame.set_from_pixbuf(pix)
        imstr = np.fromstring(pix.get_pixels(), np.uint8)
        img_np = cv2.imdecode(imstr, cv2.IMREAD_UNCHANGED)
        
        RGB_VEC = np.reshape(imstr,(79056,3))
        

        Ch1 = np.reshape(RGB_VEC[:,0],(244, 324))
        Ch2 = np.reshape(RGB_VEC[:,1],(244, 324))
        Ch3 = np.reshape(RGB_VEC[:,2],(244, 324))

        color_img = cv2.merge((Ch1,Ch2,Ch3))
        gray_img =  Ch1
        self.grayImg  = gray_img 

        #print(meanVal )
        #cv2.imshow('frame', gray_img)

    def _stateFlipFlop(self):
        img_white = 100*np.ones((1000,1000))
        img_black = 0*np.ones((1000,1000))
        if self.num_samples <= self.MAX_samp:
            if(self.state_flag == 0):
                cv2.imshow('frameI', img_white)
                self.startTime = time.time()
            elif(self.state_flag == 1):
                
                
                #self.state_flag = 0
                cv2.imshow('frameI', img_black)
                self.num_samples += 1 
                self.deltaT.append(self.readTime -self.startTime)
                print(self.num_samples,': ',self.readTime -self.startTime )
                

            else:
                pass
        else:
            Gtk.main_quit()

            
    def _runRos(self):
        Pos = self.PosReader.get_Pos()
        Pos_m = self.MarkerPosReader.get_Pos()
        Q = self.PosReader.get_Quaternion()
        Q_mkr = self.MarkerPosReader.get_Quaternion()
            
        DCM = self.PosReader.get_DCM()
        print(DCM )
        self.sv1.add_Entry(time.time(), self.grayImg, Pos, Q, Xm = Pos_m, Qm = Q_mkr)
        cv2.imshow('frame', self.grayImg)
        


    def _showframe(self, imgdata):
        # Add FPS/img size to window title
        if (self._start != None):
            fps = 1 / (time.time() - self._start)
            GLib.idle_add(self.set_title, "{:.1f} fps / {:.1f} kb".format(fps, len(imgdata)/1000))
        self._start = time.time()

        # Try to decode JPEG from the data sent from the stream
        try:
            img_loader = GdkPixbuf.PixbufLoader()
            img_loader.write(imgdata)
            #print(GdkPixbuf.PixbufLoader().shape())
            #cv2.imshow('frame', GdkPixbuf.PixbufLoader())
            img_loader.close()
            pix = img_loader.get_pixbuf()
            
            
            GLib.idle_add(self._update_image, pix)
            GLib.idle_add(self._runRos )
        except gi.repository.GLib.Error:
            print("Could not set image!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Connect to AI-deck JPEG streamer example')
    #parser.add_argument("-n",  default="192.168.4.1", metavar="ip", help="AI-deck IP")
    parser.add_argument("-n",  default="192.168.2.29", metavar="ip", help="AI-deck IP")
    parser.add_argument("-p", type=int, default='5000', metavar="port", help="AI-deck port")
    args = parser.parse_args()

    deck_port = args.p
    deck_ip = args.n

    rospy.init_node("Position Logger", disable_signals=True)
    PosReader = PositionReader('cf0')
    MarkerPosReader = PositionReader('Marker')
    sv1 = SaveFileObj()

    fw = FrameViewer(sv1, PosReader, MarkerPosReader )
    fw.show_all()




    
    try:
        Gtk.main()
        sv1.saveObj("dataTestSet.xz")
        # while not rospy.is_shutdown():

        #     Pos = PosReader.get_Pos()
        #     Q = PosReader.get_Quaternion()
            
        #     DCM = PosReader.get_DCM()
        #     print(DCM )
        #     sv1.add_Entry(time.time(), img_white, Pos, Q)
        #     time.sleep(0.5)


            

    except KeyboardInterrupt:
        Gtk.main_quit()
        #print(np.shape(sv1.ImageData))
        #sv1.saveObj("dataTestSet.xz")
        #cv2.imshow('frameO',sv1.ImageData[:,:,5])
        #cv2.waitKey(0)
        pass

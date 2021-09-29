import argparse
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
import threading
import time
import socket,os,struct
import cv2
#commited to Git


import numpy as np
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

# UI for showing frames from AI-deck example
class FrameViewer(Gtk.Window):

    def __init__(self):
        super(FrameViewer, self).__init__()
        self.frame = None
        self.init_ui()
        self._start = None
        self.set_default_size(374, 294)
        #at state zero 0 the frame is white, sets to black when observation is made
        self.state_flag = 0
        self.startTime = 0
        self.readTime = 0
        self.deltaT = 0

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
        meanVal = np.mean(Ch1)
        if(meanVal > 40):
            #self.readTime = time.time()
            self.state_flag = 1
        elif(meanVal < 40):
            pass
            self.state_flag = 0
        #print(meanVal )
        cv2.imshow('frame', gray_img)

    def _stateFlipFlop(self):
        img_white = 100*np.ones((1000,1000))
        img_black = 0*np.ones((1000,1000))

        if(self.state_flag == 0):
            cv2.imshow('frameI', img_white)
            self.startTime = time.time()
        elif(self.state_flag == 1):
            self.readTime = time.time()
            
            #self.state_flag = 0
            cv2.imshow('frameI', img_black)
            time.sleep(0.5)
            print(self.readTime -self.startTime )
        else:
            pass

        

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
            GLib.idle_add(self._stateFlipFlop)
        except gi.repository.GLib.Error:
            print("Could not set image!")

# Args for setting IP/port of AI-deck. Default settings are for when
# AI-deck is in AP mode.
parser = argparse.ArgumentParser(description='Connect to AI-deck JPEG streamer example')
#parser.add_argument("-n",  default="192.168.4.1", metavar="ip", help="AI-deck IP")
parser.add_argument("-n",  default="192.168.2.29", metavar="ip", help="AI-deck IP")
parser.add_argument("-p", type=int, default='5000', metavar="port", help="AI-deck port")
args = parser.parse_args()

deck_port = args.p
deck_ip = args.n

fw = FrameViewer()
fw.show_all()
try:
    Gtk.main()
except KeyboardInterrupt:
    Gtk.main_quit()
import numpy as np
import pickle
import lzma
import cv2
import time
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize
fig = plt.figure()
ax = plt.axes(projection='3d')

fg, ax2 = plt.subplots()


def MarkerDetection(Img):

    dst = cv2.GaussianBlur(Img,(5,5),cv2.BORDER_DEFAULT)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(dst)
    #y = 244, x = 324
    return [maxLoc[1],maxLoc[0]]

def BCDistoModel(x_p,y_p):
    #u -> Pyc   v -> Pxc
    x_p_c, y_p_c = 0., 0.

    r = np.sqrt(x_p**2) + np.sqrt(y_p**2)

    

    return x_p_c, y_p_c

def RadialDistoModel(P_p, K):
    #u -> Pyc   v -> Pxc
    x_p,y_p = P_p[0], P_p[1]
    K1 = K[0]
    K2 = K[1]
    K3 = K[2]
    x_p_c, y_p_c = 0., 0.

    r = np.sqrt(x_p**2) + np.sqrt(y_p**2)
    x_p_c = x_p / (1 + K1*r**2 + K2*r**4 + K3*r**6)
    y_p_c = y_p / (1 + K1*r**2 + K2*r**4 + K3*r**6)

    return x_p_c, y_p_c


def InvRadialDistoModel(P_p, K):
    #u -> Pyc   v -> Pxc
    x_p,y_p = P_p[0], P_p[1]
    K1 = K[0]
    K2 = K[1]
    K3 = K[2]
    x_p_c, y_p_c = 0., 0.

    r = np.sqrt(x_p**2) + np.sqrt(y_p**2)
    x_p_c = x_p * (1 + K1*r**2 + K2*r**4 + K3*r**6)
    y_p_c = y_p * (1 + K1*r**2 + K2*r**4 + K3*r**6)

    return [x_p_c[0,0], y_p_c[0,0]]

def invCamMatrixFunc(P_p, fx, fy, cx, cy):
    mtx = np.matrix([[fx, 0., cx],[0., fy, cy],[0., 0., 1.0]])
    invMtx = np.linalg.inv(mtx)
    PixelVec = np.matrix([P_p[0],P_p[1],1]).transpose()
    return invMtx @ PixelVec

def CamMatrixFunc(P_w, fx, fy, cx, cy):
    mtx = np.matrix([[fx, 0., cx],[0., fy, cy],[0., 0., 1.0]])
    #invMtx = np.linalg.inv(mtx)
    PixelVec = np.matrix([P_w[0]/(P_w[2]+1e-10),P_w[1]/(P_w[2]+1e-10),1]).transpose()
    #PixelVec = np.matrix([P_w[0],P_w[1],1]).transpose()
    return mtx @ PixelVec

DataDict = pickle.load( lzma.open( "dataTestSetWorking2.xz", "rb" ) )


Images = DataDict[ 'ImageData' ] 
shapeIm =  np.shape(Images)
markerMotion = DataDict[ 'MarkerPosition' ] 
droneMotion = DataDict[ 'Position' ]

#compute Camera Vect
droneQ =  DataDict[ 'Attitude' ] 
droneDCM = np.zeros((3,3, shapeIm[2]))
a = np.matrix([1,0,0]).transpose()
CameraVector = np.zeros((shapeIm[2],3))
for i in range(0,shapeIm[2]):
    try:
        Q = droneQ[i,:]
        r = R.from_quat(Q)
        droneDCM[:,:,i] = r.as_matrix()
    except:
        Q = [0,0,0,1]
        r = R.from_quat(Q)
        droneDCM[:,:,i] = r.as_matrix()
    
    temp = r.as_matrix() @ a
    CameraVector[i,:] = temp.transpose()
##a = np.matrix([1,0,0]).transpose()
start = 200
start = 211
end = shapeIm[2]-160
end = 600
DCM = np.mean(droneDCM, axis=2 )
ViewVector = np.zeros((shapeIm[2],3))
for i in range(0,shapeIm[2]):
    #temp = droneDCM[:,:,i].transpose() @ (np.asmatrix(markerMotion[i,:] - droneMotion[i,:])).transpose() + np.asmatrix(droneMotion[i,:]).transpose() 
    mtx = np.matrix([[0., 0., 1],[0., 1, 0],[-1.0, 0., 0]])
    mtx2 = np.matrix([[1, 0, 0],[0., -1, 0],[0, 0., 1]])
    MTX = mtx2 @ mtx
    temp =  MTX.transpose() @  DCM.transpose() @ (np.asmatrix(markerMotion[i,:] - droneMotion[i,:])).transpose() 
    ViewVector[i,:] = temp.transpose() 

#ViewVector = markerMotion - droneMotion
# #CameraVector = 1
# tranlated = CameraVector + droneMotion
# print(CameraVector[10,:])

ax.plot3D(markerMotion[start:end,0],markerMotion[start:end,1],markerMotion[start:end,2])
ax.plot3D(droneMotion[start:end,0],droneMotion[start:end,1],droneMotion[start:end,2])
ax.plot3D(ViewVector [start:end,0],ViewVector[start:end,1],ViewVector[start:end,2])


# #print(markerMotion)

DotLocation = []

for i in range(0,shapeIm[2]):
    dst = cv2.GaussianBlur(0.01*Images[40:,:,i],(5,5),cv2.BORDER_DEFAULT)
    #maxima = cv2.dilate(dst, None, iterations=3)
    MaxLoc = MarkerDetection(dst)
    DotLocation.append(MaxLoc)
    #image = cv2.circle(dst, (MaxLoc,),10, 255 )
    #print(MaxLoc)
print(np.shape(DotLocation))
DotLocation = np.array(DotLocation)


#attempt Transform with guess fx,fy
P_p = []
P_w = []
fx = 175
fy = 175
cx = 324/2
cy = 244/2
#cx = 0
#cy = 0
K = [0, 0, 0]
for i in range(0,shapeIm[2]):
    a = CamMatrixFunc(ViewVector[i,:] , fx, fy, cx, cy).transpose()
    b = invCamMatrixFunc(DotLocation[i,:] , fx, fy, cx, cy)
    mtx = np.matrix([[0., 0., 1],[0., 1, 0],[-1.0, 0., 0]])
    mtx2 = np.matrix([[1, 0, 0],[0., -1, 0],[0, 0., 1]])
    b = mtx2 @ mtx @ b
    b = b.transpose()
    print(np.shape(a))
    P_p.append([a[0,0],a[0,1]])
    P_w.append([b[0,0],b[0,1], b[0,2]])



P_p = np.array(P_p)
P_w = np.array(P_w) 
print(np.shape(P_p))
ax.plot3D(P_w[start:end,0],P_w[start:end,1],P_w[start:end,2])
#PixelTransform #y = 244, x = 324
ax2.plot( DotLocation[start:end,1], DotLocation[start:end,0])
ax2.plot(P_p[start:end,1], P_p[start:end,0])

ax2.invert_yaxis()
plt.show()

for i in range(0,shapeIm[2]):
    dst = cv2.GaussianBlur(0.01*Images[40:,:,i],(5,5),cv2.BORDER_DEFAULT)
    #maxima = cv2.dilate(dst, None, iterations=3)
    MaxLoc = MarkerDetection(dst)
    image = cv2.circle(dst, (MaxLoc[1],MaxLoc[0] ),10, 255 )
    print(MaxLoc)
    cv2.imshow("Vid frame", image)
    time.sleep(1/45)
    print(i)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
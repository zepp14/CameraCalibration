from functools import lru_cache
import numpy as np
import pickle
import lzma
import cv2
import time
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
from scipy.stats import chi2,norm

fig = plt.figure()
ax = plt.axes(projection='3d')

fg, ax2 = plt.subplots()
fg3, ax3 = plt.subplots(1,3)
fg4, ax4 = plt.subplots(4,1)

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
    K2 = K[1] / 1e4
    K3 = K[2] / 1e6
    x_p_c, y_p_c = 0., 0.

    r = np.sqrt(x_p**2 + y_p**2)
    x_p_c = x_p / (1. + (K1*r**2 + K2*r**4 + K3*r**6)/1e11)
    y_p_c = y_p / (1. + (K1*r**2 + K2*r**4 + K3*r**6)/1e11)

    return x_p_c, y_p_c


def InvRadialDistoModel(P_p, K):
    #u -> Pyc   v -> Pxc
    x_p,y_p = P_p[0], P_p[1]
    K1 = K[0]
    K2 = K[1] / 1e4
    K3 = K[2] / 1e6
    x_p_c, y_p_c = 0., 0.

    r = np.sqrt(x_p**2) + np.sqrt(y_p**2)
    x_p_c = x_p + x_p * (K1*r**2 + K2*r**4 + K3*r**6) / 1e11
    y_p_c = y_p + y_p * (K1*r**2 + K2*r**4 + K3*r**6) /  1e11

    return [x_p_c, y_p_c]

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

def CostFunction(P_w, P_p, param):
    J_array = []
    #parameters = [fx, fy, K1, K2, K3, ang1, ang2]
    fx = param[0]
    fy = param[1]
    cx = 324/2
    cy = 244/2
    K = [param[2], param[3], param[4]]
    ang1_set = param[5]
    ang2_set = param[6]

    P_pix = []
    

    shapePw = np.shape(P_w)
    for i in range(0,shapePw[0]):

        mtx = np.matrix([[0., 0., 1],[0., 1, 0],[-1.0, 0., 0]])
        mtx2 = np.matrix([[1, 0, 0],[0., -1, 0],[0, 0., 1]])
        ang1 = np.deg2rad(ang1_set)
        ang2 = np.deg2rad(ang2_set)
        yAdj = np.matrix([[np.cos(ang1), 0, np.sin(ang1)],[0., 1., 0],[-np.sin(ang1), 0., np.cos(ang1)]])
        zAdj = np.matrix([[np.cos(ang2), -np.sin(ang2), 0],[np.sin(ang2), np.cos(ang2),0.],[0.,0.,1.]])

        MTX = yAdj @ zAdj @ mtx2 @ mtx  
        temp =  MTX.transpose() @  DCM.transpose() @ (np.asmatrix(P_w[i,:])).transpose() 

        ViewVector[i,:] = temp.transpose()
        CamFrame1 = CamMatrixFunc(ViewVector[i,:] , fx, fy, cx, cy).transpose()

        G_P_p = InvRadialDistoModel([CamFrame1[0,0],CamFrame1[0,1]], K)
        #fliped axis
        G_mat = np.matrix([G_P_p[0], G_P_p[1]]).transpose()
        Xmeas = np.asmatrix(P_p[i,:]).transpose()
        J = 1/2 * (Xmeas - G_mat).transpose() @  (Xmeas - G_mat) + 1e-3 * (G_mat).transpose() @  (G_mat) 
        J_array.append(J)

        P_pix.append([G_P_p[0],G_P_p[1]])
       
    P_pix = np.array(P_pix)
    return np.mean(J_array), P_pix

def GradCostFunction(P_w, P_p, param, h=1e-6):
    shParam = np.shape(param)
    gradient = []

    for i in range(0, len(param)):
        X = np.array(param)
        
        H = np.zeros((1,len(param)))
        H[0,i] = h
     
        XpH = X + H
        XmH = X - H
        
        Jph,_ = CostFunction(P_w, P_p, XpH[0])
        Jmh,_ = CostFunction(P_w, P_p, XmH[0])

        grad = (1/(2*h))*(Jph - Jmh)
        gradient.append(grad)





    return(gradient)

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]


DataDict = pickle.load( lzma.open( "dataTestSetWorking2.xz", "rb" ) )


Images = DataDict[ 'ImageData' ] 
shapeIm =  np.shape(Images)
markerMotion = DataDict[ 'MarkerPosition' ] 
droneMotion = DataDict[ 'Position' ]
TimeVect = DataDict[ 'Time_Vector' ]

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
#start = 211
end = shapeIm[2]-220
#end = 600
DCM = np.mean(droneDCM[:,:,start:end], axis=2 )
ViewVector = np.zeros((shapeIm[2],3))
for i in range(0,shapeIm[2]):
    #temp = droneDCM[:,:,i].transpose() @ (np.asmatrix(markerMotion[i,:] - droneMotion[i,:])).transpose() + np.asmatrix(droneMotion[i,:]).transpose() 
    mtx = np.matrix([[0., 0., 1],[0., 1, 0],[-1.0, 0., 0]])
    mtx2 = np.matrix([[1, 0, 0],[0., -1, 0],[0, 0., 1]])
    ang = np.deg2rad(15)
    yAdj = np.matrix([[np.cos(ang), 0, np.sin(ang)],[0., 1., 0],[-np.sin(ang), 0., np.cos(ang)]])

    MTX = yAdj @ mtx2 @ mtx 
    temp =  MTX.transpose() @  DCM.transpose() @ (np.asmatrix(markerMotion[i,:] - droneMotion[i,:])).transpose() 

    ViewVector[i,:] = temp.transpose() 

#ViewVector = markerMotion - droneMotion
# #CameraVector = 1
# tranlated = CameraVector + droneMotion
# print(CameraVector[10,:])

ax.plot3D(markerMotion[start:end,0],markerMotion[start:end,1],markerMotion[start:end,2])
ax.plot3D(droneMotion[start:end,0],droneMotion[start:end,1],droneMotion[start:end,2])
#ax.plot3D(ViewVector[start:end,0], ViewVector[start:end,1],ViewVector[start:end,2])


# #print(markerMotion)

DotLocation = []
offset = 40
for i in range(0,shapeIm[2]):
    dst = cv2.GaussianBlur(0.01*Images[offset:,:,i],(5,5),cv2.BORDER_DEFAULT)
    #maxima = cv2.dilate(dst, None, iterations=3)
    MaxLoc = MarkerDetection(dst)
    MaxLoc[0] = MaxLoc[0] +offset
    DotLocation.append(MaxLoc)
    #image = cv2.circle(dst, (MaxLoc,),10, 255 )
    #print(MaxLoc)
print(np.shape(DotLocation))
DotLocation = np.array(DotLocation)


#attempt Transform with guess fx,fy



ang1 = 0
ang2 = 0
PW = markerMotion- droneMotion
params =  [1.63398284e+02, 1.65892853e+02, 1.10086469e-03, 3.73184715e-03, 1.18046238e-01, 4.79938573e+00, 1.65922751e+01]
fx = params[0]
fy = params[1]
cx = 324/2
cy = 244/2
#cx = 0
#cy = 0
K1 = params[2]
K2 = params[3]
K3 = params[4]
K = [K1, K2, K3]
ang1 = params[5]
ang2 = params[6]
ang1_set = ang1
ang2_set = ang2
PW = markerMotion- droneMotion
J,P_pix2 = CostFunction(PW[start:end,:], DotLocation[start:end,:], [fx, fy, K1, K2, K3, ang1, ang2])



#ProjectedPoints = [invCamMatrixFunc(x, fx, fy, cx, cy) for x in DotLocation]
ProjectedPoints = [invCamMatrixFunc(RadialDistoModel(x, K), fx, fy, cx, cy) for x in DotLocation]
PP = np.asarray(ProjectedPoints)
PP = PP[:,:,0]
sz = np.shape(PP)
VeiwedPoint  = np.zeros((shapeIm[2],3))

TrueViewPosition =  markerMotion - droneMotion

for i in range(0, shapeIm[2]):
    mtx = np.matrix([[0., 0., 1],[0., 1, 0],[-1.0, 0., 0]])
    mtx2 = np.matrix([[1, 0, 0],[0., -1, 0],[0, 0., 1]])
    ang1 = np.deg2rad(ang1_set)
    ang2 = np.deg2rad(ang2_set)
    yAdj = np.matrix([[np.cos(ang1), 0, np.sin(ang1)],[0., 1., 0],[-np.sin(ang1), 0., np.cos(ang1)]])
    zAdj = np.matrix([[np.cos(ang2), -np.sin(ang2), 0],[np.sin(ang2), np.cos(ang2),0.],[0.,0.,1.]])

    MTX = yAdj @ zAdj @ mtx2 @ mtx  
    temp =  DCM @ MTX  @ (np.asmatrix(PP[i,:])).transpose() 

    VeiwedPoint [i,:] =   (temp.transpose() / np.linalg.norm(temp)) 

error = VeiwedPoint[start:end,:] - normalize(TrueViewPosition[start:end,:])


sqError = np.sqrt(np.linalg.norm(error, axis=1))

#angular error
CamView = cartesian_to_spherical( VeiwedPoint[start:end,0], VeiwedPoint[start:end,1],VeiwedPoint[start:end,2])
LatCam = CamView[1][:].value
LonCam = CamView[2][:].value


TrueView = cartesian_to_spherical( TrueViewPosition[start:end,0], TrueViewPosition[start:end,1],TrueViewPosition[start:end,2])
LatTrueView = TrueView[1][:].value
LonTrueView= TrueView[2][:].value
LatError = reject_outliers(LatCam - LatTrueView, m = 3.)
Latmu, Latstd = norm.fit(LatError)

print(Latmu)

LonError = reject_outliers(LonCam - LonTrueView, m = 3.)
Lonmu, Lonstd = norm.fit(LonError)

ax.plot3D(VeiwedPoint[:,0],VeiwedPoint[:,1],VeiwedPoint[:,2])
#PixelTransform #y = 244, x = 324
ax2.plot( DotLocation[start:end,1], DotLocation[start:end,0])
ax2.plot(P_pix2[:,1], P_pix2[:,0],'r--')
ax2.set_xlim([0,324])
ax2.set_ylim([0,244])

ax2.invert_yaxis()



ax3[0].hist((LatError),density=False, bins=40 )
#ax3[0].set_xlim((-0.05,0.05))
ax3[1].hist((LonError),density=False, bins=40 )
#ax3[1].set_xlim((-0.05,0.05))
#ax3[2].hist(error[:,2],density=True, bins=25 )
#ax3[2].set_xlim((-0.05,0.05))

normViewed = VeiwedPoint
normView = normalize(TrueViewPosition)

ax4[0].plot(TimeVect[start:end],normViewed[start:end,0])
ax4[1].plot(TimeVect[start:end],normViewed[start:end,1])
ax4[2].plot(TimeVect[start:end],normViewed[start:end,2])

ax4[0].plot(TimeVect[start:end],normView[start:end,0])
ax4[1].plot(TimeVect[start:end],normView[start:end,1])
ax4[2].plot(TimeVect[start:end],normView[start:end,2])

plt.show()

# for i in range(0,shapeIm[2]):
#     dst = cv2.GaussianBlur(0.01*Images[40:,:,i],(5,5),cv2.BORDER_DEFAULT)
#     #maxima = cv2.dilate(dst, None, iterations=3)
#     MaxLoc = MarkerDetection(dst)
#     image = cv2.circle(dst, (MaxLoc[1],MaxLoc[0] ),10, 255 )
#     print(MaxLoc)
#     cv2.imshow("Vid frame", image)
#     time.sleep(1/45)
#     print(i)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
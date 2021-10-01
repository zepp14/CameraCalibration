
import numpy as np
import pickle
import lzma
import cv2
import time
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
fig = plt.figure()
ax = plt.axes(projection='3d')

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
start = 194
end = shapeIm[2]-160

#CameraVector = 1
tranlated = CameraVector + droneMotion
print(CameraVector[10,:])

ax.plot3D(markerMotion[start:end,0],markerMotion[start:end,1],markerMotion[start:end,2])
ax.plot3D(droneMotion[start:end,0],droneMotion[start:end,1],droneMotion[start:end,2])
#ax.plot3D(tranlated [:,0],tranlated [:,1],tranlated [:,2])


#print(markerMotion)
plt.show()
# for i in range(0,shapeIm[2]):
#      cv2.imshow("Vid frame", 0.01*Images[:,:,i])
#      time.sleep(1/45)
#      print(i)
#      if cv2.waitKey(25) & 0xFF == ord('q'):
#          break
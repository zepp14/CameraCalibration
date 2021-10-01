
import numpy as np
import pickle
import lzma
import cv2
import time
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')

DataDict = pickle.load( lzma.open( "dataTestSet.xz", "rb" ) )


Images = DataDict[ 'ImageData' ] 
markerMotion = DataDict[ 'MarkerPosition' ] 
droneMotion = DataDict[ 'Position' ] 

start = 0
end = 0

ax.plot3D(markerMotion[:,0],markerMotion[:,1],markerMotion[:,2])
ax.plot3D(droneMotion[:,0],droneMotion[:,1],droneMotion[:,2])


shapeIm =  np.shape(Images)
print(markerMotion)
plt.show()
# for i in range(0,shapeIm[2]):
#     cv2.imshow("Vid frame", 0.01*Images[:,:,i])
#     time.sleep(1/30)
#     print(0.01*Images[:,:,i])
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
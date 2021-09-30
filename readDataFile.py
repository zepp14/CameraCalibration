
import numpy as np
import pickle
   
favorite_color = pickle.load( open( "dataTestSet.pkl", "rb" ) )

print(favorite_color['Time_Vector'])
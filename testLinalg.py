import numpy as np
T = 0.15
A = np.matrix([[1, T ],[0,1]])
B = np.matrix([[T],[0]])

KK =   np.linalg.inv(B.transpose() @ B) @ B.transpose()

Ad = A-(B @ KK @ A)
print(np.linalg.eigvals(0.99*Ad))
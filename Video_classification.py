import cv2 
import numpy as np 
import os
import matplotlib.pyplot as plt
import torch

map = {
    "zero":0,
    "one":1,
    "two":2,
    "three":3,
    "four":4,
    "five":5,
    "six":6,
    "seven":7,
    "eight":8,
    "nine":9
}

path_train = "Dataset/MNIST/train"
imgs_train = os.listdir(path_train)
n_classes = 10
starts = []
images = []
M = 200
N = 784

phi = np.random.randn(M, N)

# phi = torch.load("models/phi_batch_32/phi_200.pt")
# phi.requires_grad = False
# phi = phi.detach().numpy()


for i in range(n_classes):
    id = 0
    for j in range(len(imgs_train)):
        if(id == 0):
            starts.append(len(images))
            id = 1
        if map[imgs_train[j].split("_")[0]] == i:
            images.append(np.array(plt.imread(path_train + "/" + imgs_train[j])).flatten())

images = np.array(images)


centers = np.zeros((n_classes, M))
for i in range(n_classes - 1):
    centers[i,:] = np.mean((phi @ images[starts[i]:starts[i + 1]].T).T, axis=0)

centers[9,:] = np.mean((phi @ images[starts[9]:len(images)].T).T, axis=0)

cap = cv2.VideoCapture('Video/Images_trans.mp4') 

if (cap.isOpened()== False): 
	print("Error opening video file") 

while(cap.isOpened()): 

    ret, frame = cap.read()

    frame1 = np.mean(frame, axis=2)

    frame1 = cv2.resize(frame1, (28, 28))

    meas = phi @ frame1.flatten()

    y_pred = np.argmin(np.linalg.norm(meas - centers, axis=1))

    if ret == True: 

        cv2.putText(frame, "Predicted Digit : " + str(y_pred), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)
        cv2.imshow('Frame', frame) 
        
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break

    else: 
        break

cap.release() 

cv2.destroyAllWindows() 

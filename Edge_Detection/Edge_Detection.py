import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import cv2

"""Reading the Image using OpenCV"""

img=cv2.imread('house.tif',0)
plt.imshow(img,cmap='gray')

"""Horizontal Edge Detection using inbuilt convolve, user built convolve and sobel function"""

[row,col]=img.shape
imgs=np.zeros((row,col),dtype=np.int)
mask=np.array([[-2,-1,-2],[0,0,0],[2,1,2]])
for i in range(1,row-1):
  for j in range(1,col-1):
    imgs[i,j]=int(np.sum(img[i-1:i+2,j-1:j+2]*mask))
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
inby=signal.convolve2d(img,mask,mode='same')
plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(imgs,cmap='gray',vmin=0,vmax=255)
plt.title('Horizontal Edge Detected')
plt.subplot(1,3,2)
plt.imshow(sobely,cmap='gray',vmin=0,vmax=255)
plt.title('Horizontal Edge Detected with sobel')
plt.subplot(1,3,3)
plt.imshow(inby,cmap='gray',vmin=0,vmax=255)
plt.title('Horizontal Edge Detected with inbuilt convolve')

"""Vertical Edge Detection using inbuilt convolve, user built convolve and sobel function"""

[row,col]=img.shape
imgsv=np.zeros((row,col),dtype=np.int)
mask=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
for i in range(1,row-1):
  for j in range(1,col-1):
    imgsv[i,j]=int(np.sum(img[i-1:i+2,j-1:j+2]*mask))
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
inbx=signal.convolve2d(img,mask,mode='same')
plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(imgsv,cmap='gray',vmin=0,vmax=255)
plt.title('Vertical Edge Detected')
plt.subplot(1,3,2)
plt.imshow(sobelx,cmap='gray',vmin=0,vmax=255)
plt.title('Vertical Edge Detected with sobel')
plt.subplot(1,3,3)
plt.imshow(inbx,cmap='gray',vmin=0,vmax=255)
plt.title('Vertical Edge Detected with inbuilt convolve')

"""Diagonal Edge Detection using inbuilt convolve, user built convolve and sobel function"""

[row,col]=img.shape
imgss=np.zeros((row,col),dtype=np.int)
mask=np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
for i in range(1,row-1):
  for j in range(1,col-1):
    imgs[i,j]=int(np.sum(img[i-1:i+2,j-1:j+2]*mask))
inb=signal.convolve2d(img,mask,mode='same')
plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(imgs,cmap='gray',vmin=0,vmax=255)
plt.title('Diagonal Edge Detected')
sobel=sobelx+sobely
plt.subplot(1,3,2)
plt.imshow(sobel,cmap='gray',vmin=0,vmax=255)
plt.title('Diagonal Edge Detected with adding sobel images')
plt.subplot(1,3,3)
plt.imshow(inb,cmap='gray',vmin=0,vmax=255)
plt.title('Diagonal Edge Detected with inbuilt convolve')


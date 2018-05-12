# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 10:44:30 2018

@author: Joe
"""

import numpy as np
import scipy as sp
from PIL import Image


def EigenfaceCore(T):
    
    #average face image
    m = np.sum(T, axis = 1)
    (rows, cols) = T.shape
    m = m/cols
    
    #subtracting image from average
    
    a = np.zeros(shape=(rows,cols))
    for i in range(0,cols-1):
        a[:,i] = T[:,i] - m
        
        
    temp = np.transpose(a) #change matrix multiplication MxM
    cov = np.matmul(temp,a) #calculate the covariance of T
    [d,v] = np.linalg.eig(cov) #calculate M eigenvectors
    
    (vrow,vcol) = np.shape(v)
    L_eig_vec = np.zeros(shape=(vrow,1))
    transfer = np.zeros(shape=(vrow,1))

    for j in range(0,vcol-1):
        if d[j] > 1:
            transfer =(v[:,j])
            transfer = transfer.reshape(vrow,1)
            if L_eig_vec[0,0] == 0:
                L_eig_vec = v[:,j]
                L_eig_vec = L_eig_vec.reshape(vrow,1)
            L_eig_vec = np.hstack((L_eig_vec,transfer))
            
    
    eigenfaces = np.matmul(a,L_eig_vec)
    
    return [m,a,eigenfaces]


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def Recognition(TestImage, m, A, Eigenfaces):
    
    
    (vrow,Train_Number)=np.shape(Eigenfaces)
    ProjectedImages = np.zeros(shape = (Train_Number,1))
    TransposeE = np.transpose(Eigenfaces)
    
    temp= np.matmul(TransposeE,A[:,0])
    temp_rows = np.shape(temp)
    mod = np.zeros(shape=(19,1))
    mod = temp
    mod = mod.reshape(19,1)
    ProjectedImages = mod
    
    
    for i in range(1,(Train_Number-1)):
        #Projection of centered images into facespace 
        temp= np.matmul(TransposeE,A[:,i]) 
        mod = temp
        mod = mod.reshape(19,1)
        ProjectedImages = np.hstack((ProjectedImages,mod))
        
        
    
    [imager,imagec] = np.shape(TestImage)    
    m_mod = np.zeros(shape=(imager,imagec))
    m_mod = m
    m_mod = m_mod.reshape(imager,imagec)
    Difference = (TestImage) - m_mod 						#Centered test image """
    ProjectedTestImage = np.matmul(TransposeE,Difference)
    
    Euc_dist = np.zeros(shape=(Train_Number))
    
    for i in range(0,(Train_Number-1)):
        q = ProjectedImages[:,i]
        q = q.reshape(19,1)
        
        step1 = ProjectedTestImage - q
        step2 = np.linalg.norm(step1)
        tempval = np.square(step2)
        Euc_dist[i] = tempval
        #print(tempval)
        #wont work for image 1 but it will do for now
        if np.min(Euc_dist) == tempval:
            Output = i+1 #images start at 1 but array index starts at 0
    
    
    
    return Output



def CreateDatabase():


#   loading images and converting them to grayscale
    #reshaping the matracies to columns
    
    face1 = Image.open("1.jpg")
    data1 = np.array(face1)
    graydata1 = rgb2gray(data1)
    (drows,dcols) = graydata1.shape
    length = drows*dcols
    graydata1 = graydata1.reshape(length,1)
    
    face2 = Image.open("2.jpg")
    data2 = np.array(face2)
    graydata2 = rgb2gray(data2)
    (drows,dcols) = graydata2.shape
    length = drows*dcols
    graydata2 = graydata2.reshape(length,1)
    
    face3 = Image.open("3.jpg")
    data3 = np.array(face3)
    graydata3 = rgb2gray(data3)
    (drows,dcols) = graydata3.shape
    length = drows*dcols
    graydata3 = graydata3.reshape(length,1)
    
    face4 = Image.open("4.jpg")
    data4 = np.array(face4)
    graydata4 = rgb2gray(data4)
    (drows,dcols) = graydata4.shape
    length = drows*dcols
    graydata4 = graydata4.reshape(length,1)
    
    face5 = Image.open("5.jpg")
    data5 = np.array(face5)
    graydata5 = rgb2gray(data5)
    (drows,dcols) = graydata5.shape
    length = drows*dcols
    graydata5 = graydata5.reshape(length,1)
    
    face6 = Image.open("6.jpg")
    data6 = np.array(face6)
    graydata6 = rgb2gray(data6)
    (drows,dcols) = graydata6.shape
    length = drows*dcols
    graydata6 = graydata6.reshape(length,1)
    
    face6 = Image.open("6.jpg")
    data6 = np.array(face6)
    graydata6 = rgb2gray(data6)
    (drows,dcols) = graydata6.shape
    length = drows*dcols
    graydata6 = graydata6.reshape(length,1)
    
    face7 = Image.open("7.jpg")
    data7 = np.array(face7)
    graydata7 = rgb2gray(data7)
    (drows,dcols) = graydata7.shape
    length = drows*dcols
    graydata7 = graydata7.reshape(length,1)
    
    face8 = Image.open("8.jpg")
    data8 = np.array(face8)
    graydata8 = rgb2gray(data8)
    (drows,dcols) = graydata8.shape
    length = drows*dcols
    graydata8 = graydata8.reshape(length,1)
    
    face9 = Image.open("9.jpg")
    data9 = np.array(face9)
    graydata9 = rgb2gray(data9)
    (drows,dcols) = graydata9.shape
    length = drows*dcols
    graydata9 = graydata9.reshape(length,1)
    
    face10 = Image.open("10.jpg")
    data10 = np.array(face10)
    graydata10 = rgb2gray(data10)
    (drows,dcols) = graydata10.shape
    length = drows*dcols
    graydata10 = graydata10.reshape(length,1)
    
    face11 = Image.open("12.jpg")
    data11 = np.array(face11)
    graydata11 = rgb2gray(data11)
    (drows,dcols) = graydata11.shape
    length = drows*dcols
    graydata11 = graydata11.reshape(length,1)
    
    face12 = Image.open("12.jpg")
    data12 = np.array(face12)
    graydata12 = rgb2gray(data12)
    (drows,dcols) = graydata12.shape
    length = drows*dcols
    graydata12 = graydata12.reshape(length,1)
    
    face13 = Image.open("13.jpg")
    data13 = np.array(face13)
    graydata13 = rgb2gray(data13)
    (drows,dcols) = graydata13.shape
    length = drows*dcols
    graydata13 = graydata13.reshape(length,1)
    
    face14 = Image.open("14.jpg")
    data14 = np.array(face14)
    graydata14 = rgb2gray(data14)
    (drows,dcols) = graydata14.shape
    length = drows*dcols
    graydata14 = graydata14.reshape(length,1)
    
    face15 = Image.open("15.jpg")
    data15 = np.array(face15)
    graydata15 = rgb2gray(data15)
    (drows,dcols) = graydata15.shape
    length = drows*dcols
    graydata15 = graydata15.reshape(length,1)
    
    face16 = Image.open("16.jpg")
    data16 = np.array(face16)
    graydata16 = rgb2gray(data16)
    (drows,dcols) = graydata16.shape
    length = drows*dcols
    graydata16 = graydata16.reshape(length,1)
    
    face17 = Image.open("17.jpg")
    data17 = np.array(face17)
    graydata17 = rgb2gray(data17)
    (drows,dcols) = graydata17.shape
    length = drows*dcols
    graydata17 = graydata17.reshape(length,1)
    
    face18 = Image.open("18.jpg")
    data18 = np.array(face18)
    graydata18 = rgb2gray(data18)
    (drows,dcols) = graydata18.shape
    length = drows*dcols
    graydata18 = graydata18.reshape(length,1)
    
    face19 = Image.open("19.jpg")
    data19 = np.array(face19)
    graydata19 = rgb2gray(data19)
    (drows,dcols) = graydata19.shape
    length = drows*dcols
    graydata19 = graydata19.reshape(length,1)
    
    face20 = Image.open("20.jpg")
    data20 = np.array(face20)
    graydata20 = rgb2gray(data20)
    (drows,dcols) = graydata20.shape
    length = drows*dcols
    graydata20 = graydata20.reshape(length,1)
    
    #creation of the T matrix 
    T = np.zeros(shape = (length,1))
    
    #filling the T matrix
    T = graydata1
    T = np.hstack((T,graydata2))
    T = np.hstack((T,graydata3))
    T = np.hstack((T,graydata4))
    T = np.hstack((T,graydata5))
    T = np.hstack((T,graydata6))
    T = np.hstack((T,graydata7))
    T = np.hstack((T,graydata8))
    T = np.hstack((T,graydata9))
    T = np.hstack((T,graydata10))
    T = np.hstack((T,graydata11))
    T = np.hstack((T,graydata12))
    T = np.hstack((T,graydata13))
    T = np.hstack((T,graydata14))
    T = np.hstack((T,graydata15))
    T = np.hstack((T,graydata16))
    T = np.hstack((T,graydata17))
    T = np.hstack((T,graydata18))
    T = np.hstack((T,graydata19))
    T = np.hstack((T,graydata20))
    


    return T


test_face = Image.open("10.jpg")
test_data = np.array(test_face)
graydata_test = rgb2gray(test_data)
(trows,tcols) = np.shape(graydata_test)
length = trows*tcols
gray_test = graydata_test.reshape(length,1)


#function calls and evaluation
t = CreateDatabase()
(m,a,eigenfaces) = EigenfaceCore(t)
output = Recognition(gray_test, m, a, eigenfaces)

print(output)

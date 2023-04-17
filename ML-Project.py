import os
import cv2 as cv
import numpy as np
from skimage.feature import canny
from skimage.feature import hog
from sklearn.neural_network import MLPClassifier
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage import exposure
from numpy import asarray

#   Reading Multiple Images From File
#   Using os.walk module, to give us roots , directories and files in our selected path.

def Read_IMG_Path(Path):
    Total_Images = []
    No_Of_Imgs = []
    for roots, dir, files in os.walk(Path):
        No_Of_Imgs.append(len(files))
        for file in files:
            p = os.path.join(roots, file)
            Total_Images.append(p)
        
    return Total_Images, No_Of_Imgs

#   Our Data of Training
Data_Train, Train_Count = Read_IMG_Path('data/train')

#   Our Data of Validation
Data_Validation, Validation_Count = Read_IMG_Path("data/val")

#   DS_Store file removing
Train_Count.pop(0)
Validation_Count.pop(0)
Data_Train.pop(0)
Data_Validation.pop(0)
# print(Data_Train)
# print(Data_Validation)

# Path = 'data/train'
# print(Total_Images)
# print(No_Of_Imgs)

# for roots, dir, files in os.walk(Path):
    # print(files)
    # print(dir)
    # print(roots)
# Files = os.listdir(Path)
# print(Files)

# for File in Files:
#     imgpath = os.path.join(Path, File)
#     img = cv.imread(imgpath, 0)
#     cv.imshow("img",img)
#     cv.waitKey(0)
#     break
# print(Files)



#   -------------------------   FEATURES ---------------------------------------

def Hog(Data):
    FeatureExtract = []
    for image in Data:
        imageRead = cv.imread(image, 0)
        
        # cv.imshow("img",img)
        # cv.waitKey(0)

        #creating hog features
        fd, hog_image = hog(imageRead, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, channel_axis=None)
        # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range = (0,5))
        # hog_image = asarray(hog_image)
        FeatureExtract.append(hog_image)
    return FeatureExtract
    #     mg = imread(image)
    #     # plt.axis("off")
    #     # plt.imshow(img)
    #     # print(img.shape)

    #     #resizing image
    #     resized_img = np.resize(mg, (128*4, 64*4))
    #     # plt.axis("off")
    #     # plt.imshow(resized_img)
    #     # plt.show()
    #     # print(resized_img.shape)

    #     #creating hog features
    #     fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
    #                         cells_per_block=(2, 2),channel_axis=None, visualize=True, feature_vector=True)
    #     FeatureExtract.append(hog_image)
    # return FeatureExtract
    #     # print(fd.shape)
    #     # print(hog_image.shape)
    #     # plt.axis("off")
    #     # plt.imshow(hog_image, cmap="gray")
    #     # plt.show()
        # imageRead = cv.imread(image,0)
        # print(hog_image_rescaled)

#   Black Pixels Count
def Black_Pixels(Data):
    FeatureExtract = []
    for image in Data:
        imageRead = cv.imread(image, 0)
        imageRead = np.round(imageRead)
        count = np.count_nonzero(imageRead == 0)
        FeatureExtract.append(count)
    return FeatureExtract

#   Edge-Detection Feature
def Edge_Detection(Data):
    Feature_Extract = []
    for image in Data:
        imageRead = cv.imread(image, 0)
        # imageRead = maImg.imread(image)
        # print(imageRead)
        count = 0
        img = canny(imageRead)
        count = np.count_nonzero(imageRead == 0)
        Feature_Extract.append(count)
    return Feature_Extract

#   Diagonal Feature
def Diagonal(Data):
    FeatureExtract = []
    for image in Data:
        imageRead = cv.imread(image, 0)
        imageRead = np.resize(imageRead, (300, 300))
        diag = np.diag(imageRead)
        FeatureExtract.append(diag)
    return FeatureExtract


#   Centroid Feature
def Centroid(Data):
    Feature_Extract = []
    for image in Data:
        imageRead = cv.imread(image, 0)
        mid_pixel = len(imageRead)//2
        new_img = imageRead[mid_pixel-5:mid_pixel+5]
        train_img = []
        for i in new_img:
            train_img.append(i[(len(i)//2)-50:(len(i)//2)+50])
        train_img = np.array(train_img)
        train_img = np.reshape(train_img, (1, 1000))
        # print(train_img)
        Feature_Extract.append(train_img)
    return Feature_Extract


train_x = Black_Pixels(Data_Train)
# train_x = Diagonal(Data_Train)
# train_x = Hog(Data_Train)
# train_x = Edge_Detection(Data_Train)
# train_x = Centroid(Data_Train)
train_x = np.array(train_x)
# print(featureExtract)
train_x = train_x.reshape(len(train_x), -1)
# print(train_x)


#   Setting Up Data which will be helpful for mapping our train data towards specific value from "i" to "x"
train_y = []
digit = 1
for i in Train_Count:
    for j in range(i):
        train_y.append(digit)
    digit += 1
# print(train_y)
train_y = np.array(train_y)
#   Shaping our train_y same to our train_x data.
train_y = train_y.reshape(len(train_y), -1)
# print(train_y)
train_y = train_y.ravel()
# print(train_y)


#   Validation Data working on our feature

Test_x = Black_Pixels(Data_Validation)
# Test_x = Diagonal(Data_Validation)
# Test_x = Hog(Data_Validation)
# Test_x = Edge_Detection(Data_Validation)
# Test_x = Centroid(Data_Validation)
Test_x = np.array(Test_x)
Test_x = Test_x.reshape(len(Test_x), -1)
# print(Test_x)

Test_y = []
digit = 1
for i in Validation_Count:
    for j in range(i):
        Test_y.append(digit)
    digit += 1
# print(Test_y)
Test_y = np.array(Test_y)
#   Shaping our test_y same to our test_x data.
Test_y = Test_y.reshape(len(Test_y), -1)
# print(Test_y)
Test_y = Test_y.ravel()
# print(Test_y)


# ----------*********************---------------- Multi Layer Perceptron CLASSIFIER -----------------******************************-------------
classification = MLPClassifier(hidden_layer_sizes=(50,100,50), activation="logistic", solver="sgd",
                    learning_rate_init=0.1, random_state=1, verbose=True, max_iter=800, n_iter_no_change=200).fit(train_x, train_y)
res = classification.predict(train_x)
# print(res)
res = np.array(res)
# print(res)

Train_Score = round(classification.score(train_x, train_y), 2)*100
print("Training Score: ",Train_Score)

Validation_Score = round(classification.score(Test_x, Test_y), 2)*100
print("Validation Score: ",Validation_Score)


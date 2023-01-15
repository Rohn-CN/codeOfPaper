import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import *
def create_detector():
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create()

    return sift,surf,orb
def display(image):
    plt.imshow(image[...,::-1])

def detect_and_match(detector,image1,image2):
    keypoints1,descriptor1 = detector.detectAndCompute(image1,None)
    keypoints2,descriptor2 = detector.detectAndCompute(image2,None)
    matcher = cv2.DescriptorMatcher.create("BruteForce")
    matches = matcher.knnMatch(descriptor1,descriptor2,1)
    dst_pts = np.float32([keypoints1[match[0].queryIdx].pt for match in matches])
    src_pts = np.float32([keypoints2[match[0].trainIdx].pt for match in matches])
    match_image = cv2.drawMatchesKnn(image1,keypoints1,image2,keypoints2,matches,None,flags=2)
    display(match_image)
    return dst_pts, src_pts

def image_resize(image,ratio):
    return cv2.resize(image, None, fx=ratio, fy=ratio)
def noise_image(image):
    # add gaussian noise
    row, col, ch= image.shape
    mean = 0
    var = 10/255
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image / 255 + gauss
    noisy *= 255
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)
def noise_image2(image,sp_ratio = 0.01):
    # add salt and pepper noise
    row, col, ch= image.shape
    out = image.copy()
    s = int(sp_ratio * image.size) // ch
    change = np.concatenate((np.random.randint(0,row,size=(s,1)),np.random.randint(0,col,size=(s,1))),axis=1)
    for i in range(s):
        r = np.random.randint(0,2)
        for j in range(ch):
            out[change[i,0], change[i,1],j] = 255 * r
    # Salt mode
    return out
def light_image(image):
    # raise the light of random patch of image
    row, col, ch= image.shape
    mask = np.zeros((row,col,ch),np.uint8)
    image_origin = image.copy()
    for i in range(3):
        image_temp = image_origin.copy().astype(np.int)
        x = np.random.randint(0, row)
        y = np.random.randint(0, col)
        w = np.random.randint(0, row - x)
        h = np.random.randint(0, col - y)
        image_temp[x:x+w, y:y+h] = np.clip(image_temp[x:x+w, y:y+h] * np.random.randint(8, 15) / 10, 0, 255)
        image[np.where(mask == 0)] = image_temp[np.where(mask == 0)]
        mask[x:x+w, y:y+h] = 255
    return image
def rotate_image(image,angle):
    # rotate the image without crop
    row, col, ch= image.shape
    new_height = int(row * fabs(cos(radians(angle))) + col * fabs(sin(radians(angle))))
    new_width = int(row * fabs(sin(radians(angle))) + col * fabs(cos(radians(angle))))
    mat_rotation = cv2.getRotationMatrix2D((col / 2, row / 2), angle, 1)
    mat_rotation[0, 2] += (new_width - col) / 2
    mat_rotation[1, 2] += (new_height - row) / 2
    out = cv2.warpAffine(image, mat_rotation, (new_width, new_height), borderValue=(0,0,0))
    return mat_rotation, out

if __name__ == "__main__":
    image1 = cv2.imread(r"/Users/ronghao/code/stitch/pystitch/test_data/0146.jpg")
    image2 = cv2.imread(r"/Users/ronghao/code/stitch/pystitch/test_data/0148.jpg")
    sift,surf,orb = create_detector()
    detect_and_match(sift,image1,image2)
    detect_and_match(surf,image1,image2)
    detect_and_match(orb,image1,image2)
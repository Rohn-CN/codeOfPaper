import cv2
import numpy as np
def create_detector():
    sift = cv2.xfeatures2d.SIFT_create()

    surf = cv2.xfeatures2d.SURF_create()

    orb = cv2.ORB_create()
    return sift,surf,orb
def detect_and_match(detector,image1,image2):
    keypoints1,descriptor1 = detector.detectAndCompute(image1,None)
    keypoints2,descriptor2 = detector.detectAndCompute(image2,None)
    matcher = cv2.DescriptorMatcher.create("BruteForce")
    matches = matcher.knnMatch(descriptor1,descriptor2,2)
    goods = []
    for match in matches:
        if len(match) == 2 and match[0].distance < match[1].distance * 0.8:
            goods.append(match[0])
            dst_pts = np.float32([keypoints1[match.queryIdx].pt for match in goods])
            src_pts = np.float32([keypoints2[match.trainIdx].pt for match in goods])
    
    print(len(matches))
    print(len(goods))

if __name__ == "__main__":
    image1 = cv2.imread(r"/Users/ronghao/code/stitch/pystitch/test_data/0146.jpg")
    image2 = cv2.imread(r"/Users/ronghao/code/stitch/pystitch/test_data/0148.jpg")
    sift,surf,orb = create_detector()
    detect_and_match(sift,image1,image2)
    detect_and_match(surf,image1,image2)
    detect_and_match(orb,image1,image2)
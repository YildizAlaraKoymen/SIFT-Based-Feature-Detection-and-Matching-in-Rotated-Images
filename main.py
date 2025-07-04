
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

size = range(0, 2)

def readOriginalImages(folderName):
    images = []
    for i in size:
        image_name = "original_" + str(i+1) + ".jpg"
        images.append((cv2.imread(folderName + "\\" + image_name), image_name))

    return images

def showImages(images):
    for image in images:
        cv2.imshow(image[1], image[0])
    cv2.waitKey()

def saveImages(folderName,images):
    for image in images:
        cv2.imwrite(folderName + "\\" + image[1], image[0])

def rotateOriginalImages(degree, images):
    #Rotation is easier in PIL, so I will convert to PIL format
    PIL_Images = [Image.fromarray(images[0][0]), Image.fromarray(images[1][0])]

    rotated_Images = []
    for i, image in enumerate(PIL_Images):
        image_name = "rotated_" + str(i+1) + ".jpg"
        rotated_Images.append((np.asarray(image.rotate(360 - degree, expand = True)), image_name))

    return rotated_Images

def findKeyPoints(images):
    keyPoints = []
    for image in images:
        sift = cv2.SIFT.create()
        empty_mask = np.zeros(image[0].shape[:2], dtype=np.uint8)
        kp, des = sift.detectAndCompute(image[0], None)
        keyPoints.append((kp, des))

    return keyPoints

def matchKeyPoints(images):
    kp = findKeyPoints(images)
    img1 = images[0][0] #original image 1
    img2 = images[1][0] #original image 2
    img3 = images[2][0] #rotated image 1
    img4 = images[3][0] #rotated image 2

    kp1 = kp[0][0]  # original image 1 keypoint
    kp2 = kp[1][0]  # original image 2 keypoint
    kp3 = kp[2][0]  # rotated image 1 keypoint
    kp4 = kp[3][0]  # rotated image 2 keypoint

    des1 = kp[0][1] #original image 1 descriptor
    des2 = kp[1][1] #original image 2 descriptor
    des3 = kp[2][1] #rotated image 1 descriptor
    des4 = kp[3][1] #rotated image 2 descriptor

    #BFMatcher object with default parameters

    bf = cv2.BFMatcher()
    matches_image1 = bf.knnMatch(des1, des3, k = 2)
    matches_image2 = bf.knnMatch(des2, des4, k = 2)

    # Apply Lowe's ratio test
    good_1 = []
    for m, n in matches_image1:
        if m.distance < 0.75 * n.distance:
            good_1.append(m)  # not [m], just m

    good_2 = []
    for m, n in matches_image2:
        if m.distance < 0.75 * n.distance:
            good_2.append(m)
    #Sort them in the order of distance
    matches_image1 = sorted(good_1, key = lambda x:x.distance)
    matches_image2 = sorted(good_2, key = lambda x:x.distance)

    #Draw first 5 matches
    # Draw first 5 matches
    match_image1 = cv2.drawMatches(img1, kp1, img3, kp3, matches_image1[:5], None, flags=cv2.DrawMatchesFlags_DEFAULT, matchesThickness=3)
    match_image2 = cv2.drawMatches(img2, kp2, img4, kp4, matches_image2[:5], None, flags=cv2.DrawMatchesFlags_DEFAULT, matchesThickness=3)

    # Convert BGR to RGB for displaying with matplotlib
    match_image1 = cv2.cvtColor(match_image1, cv2.COLOR_BGR2RGB)
    match_image2 = cv2.cvtColor(match_image2, cv2.COLOR_BGR2RGB)

    plt.imshow(match_image1),plt.savefig("Dataset\\image1_match.jpg"),plt.show()
    plt.imshow(match_image2),plt.savefig("Dataset\\image2_match.jpg"),plt.show()



if __name__ == "__main__":
    originalImages = readOriginalImages("Dataset")
    rotatedImages = rotateOriginalImages(45, originalImages)
    matchKeyPoints(originalImages + rotatedImages)
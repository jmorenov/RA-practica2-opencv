import cv2
import numpy as np

def get_hist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return hist

def get_hists_of_images_node(node, frames_per_node):
    directory = 'Frames/Nodo_' + str(node)
    hists = []

    for i in range(1, frames_per_node + 1):
        imagePath = directory + '/imagen' + str(i) + '.jpg'
        image = read(imagePath)
        hist = get_hist(image)
        hists.append(hist)

    return hists

def load_frames(number_of_nodes, frames_per_node):
    responses = []
    data = []

    for i in range(1, number_of_nodes + 1):
        print 'Loading images: Node ' + str(i) + '...'
        hists = get_hists_of_images_node(i, frames_per_node)
        responses = np.concatenate((responses, [i] * frames_per_node))
        data.extend(hists)

    data = np.float32(data)
    responses = np.float32(responses)

    return data, responses

def read(image_path):
    return cv2.imread(image_path, 0)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated
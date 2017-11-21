import numpy as np
import cv2
from sklearn.model_selection import LeaveOneOut

FRAMES_PER_NODE = 3
NUMBER_OF_NODES = 7

def get_hist(img):
    # img = cv2.imread('escilum.tif',0)
    # imagen histograma
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    # funcion distributiva acumulativa
    cdf = hist.cumsum()

    # Aplica mascara a un array
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

    # Cambio tipo mascara para aplicar a la imagen
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    hist = np.array(cdf[img]).flatten()

    return hist

def get_hists_of_images_node(node):
    directory = 'Frames/Nodo_' + str(node)
    hists = []

    for i in range(1, FRAMES_PER_NODE + 1):
        imagePath = directory + '/imagen' + str(i) + '.jpg'
        image = cv2.imread(imagePath, 0)
        hist = get_hist(image)
        hists.append(hist)

    return hists

def count_success(Y_Pred, Y):
    return np.sum(Y_Pred == Y)

print 'Loading images...'

responses = []
trainData = []

for i in range(1, NUMBER_OF_NODES + 1):
    hists = get_hists_of_images_node(i)
    responses = np.concatenate((responses, [i] * FRAMES_PER_NODE))

    for j in range(0, len(hists)):
        trainData.append(hists[j])

trainData = np.float32(trainData)
responses = np.float32(responses)

print 'Train KNN...'
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)

print 'Cross validation...'

loo = LeaveOneOut()
iterations = loo.get_n_splits(trainData)

k = 3
success = 0
i = 0
for train_index, test_index in loo.split(trainData):
    X_train, X_test = trainData[train_index], trainData[test_index]
    y_train, y_test = responses[train_index], responses[test_index]
    ret, results, neighbours, dist = knn.findNearest(X_train, k)
    success += count_success(results.transpose(), y_train)
    i += 1
    print str(i) + '/' + str(iterations)

print 'Success rate: ', (success/(((FRAMES_PER_NODE * NUMBER_OF_NODES) - 1) * iterations)) * 100, '%'
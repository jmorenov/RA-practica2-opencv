import numpy as np
import scipy.misc
import cv2

def equalizacion(img):
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

    img2 = cdf[img]

    #scipy.misc.imsave('Equalize_img.jpg', img2)
    #histograma_equalize = plt.hist(img2)
    #plt.savefig("Equalize_hist.png")

    return img2

def get_hist(image):
    return equalizacion(image)

def get_hists_of_images_node(node):
    directory = 'Frames/Nodo_' + str(node)
    hists = []

    for i in range(1, 2):
        imagePath = directory + '/imagen' + str(i) + '.jpg'
        image = cv2.imread(imagePath, 0)
        hist = get_hist(image)
        hists.append(hist)

    return hists

print 'Loading images...'

responses = []
trainData = []

for i in range(1, 2):
    hists = get_hists_of_images_node(i)
    responses = np.concatenate((responses, [i]))
    print hists
    trainData.append(hists)
    print trainData
    trainData = np.concatenate((trainData, hists))

trainData = np.float32(trainData)
responses = np.float32(responses)

print 'Train KNN...'
knn = cv2.ml.KNearest_create()

'''trainData = np.float32([
    [[1,3,4], [2,5,6], [3,3,2], [4,5,4]],
    [[4,2,3], [5,4,5], [7,5,6], [8,2,3]]
])'''

knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)

print 'Finding...'
#ret, results, neighbours ,dist = knn.find_nearest(trainData[0][0], 3)

#print "result: ", results,"\n"
#print "neighbours: ", neighbours,"\n"
#print "distance: ", dist
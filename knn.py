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

    for i in range(1, 5):
        imagePath = directory + '/imagen' + str(i) + '.jpg'
        image = cv2.imread(imagePath, 0)
        hist = get_hist(image)
        hists.append(hist)

    return hists

print 'Loading images...'

responses = []
trainData = []

for i in range(1, 8):
    hists = get_hists_of_images_node(i)
    responses.append([i, i, i, i, i])
    trainData.append(hists)

responses = np.array(responses)
trainData = np.array(trainData)

print 'Train KNN...'
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)

print 'Finding...'
ret, results, neighbours ,dist = knn.find_nearest(trainData[0][0], 3)

print "result: ", results,"\n"
print "neighbours: ", neighbours,"\n"
print "distance: ", dist
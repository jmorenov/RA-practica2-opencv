import numpy as np
import scipy.misc

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

    scipy.misc.imsave('Equalize_img.jpg', img2)
    #histograma_equalize = plt.hist(img2)
    #plt.savefig("Equalize_hist.png")
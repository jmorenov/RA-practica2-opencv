import images
import knn
import numpy as np

FRAMES_PER_NODE = 115
NUMBER_OF_NODES = 7
SAMPLE_IMAGE = 'Frames/Nodo_2/imagen6.jpg'
TEST_IMAGE = 'Test/Lenna.png'

print 'Loading images...'
data, responses = images.load_frames(NUMBER_OF_NODES, FRAMES_PER_NODE)

print 'Train KNN...'
knn_model = knn.train(data, responses)

print 'Select sample image...'
img = images.read(SAMPLE_IMAGE)
hist = images.get_hist(img)
sample = np.array([hist])

ret, results, neighbours, dist = knn.predict(knn_model, sample, 9, max_distance=6)
print results

print 'Select test image...'
img = images.read(TEST_IMAGE)
hist = images.get_hist(img)
sample = np.array([hist])

ret, results, neighbours, dist = knn.predict(knn_model, sample, 9, max_distance=6)
print results
import images
import knn
import crossvalidation
import timeit

start = timeit.default_timer()

FRAMES_PER_NODE = 115
NUMBER_OF_NODES = 7

print 'Loading images...'
data, responses = images.load_frames(NUMBER_OF_NODES, FRAMES_PER_NODE)

print 'Train KNN...'
knn_model = knn.train(data, responses)

print 'Cross validation...'
crossvalidation.loo(knn_model, data, responses, 3)

stop = timeit.default_timer()

print 'Execution time: ' + str(stop - start) + 's'
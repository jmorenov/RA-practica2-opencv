import cv2

def train(data, responses):
    knn = cv2.ml.KNearest_create()
    knn.train(data, cv2.ml.ROW_SAMPLE, responses)

    return knn

def predict(knn_model, sample, k, max_distance = -1):
    ret, results, neighbours, dist = knn_model.findNearest(sample, k)

    if max_distance != -1 and min(dist[0]) >= max_distance:
        results[0][0] = -1

    return ret, results, neighbours, dist
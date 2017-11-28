import images
import knn
import videos
import cv2
import numpy as np

def get_named_nodes(actual_node, next_node):
    actual_node_name = 'Actual Nodo ' + str(int(actual_node))
    next_node_name = 'Siguiente Nodo ' + str(int(next_node))

    return actual_node_name, next_node_name

def compute_node_detected(node_detected, result, times_detected):
    if result == -1:
        node_detected = -1
        times_detected = 0
    elif node_detected == -1:
        node_detected = result
        times_detected = 1
    elif node_detected == result:
        times_detected += 1

    return node_detected, times_detected

FRAMES_PER_NODE = 115
NUMBER_OF_NODES = 7
TEST_VIDEO = 'Videos/RecorridoCompleto_1.MOV'

print 'Loading images...'
data, responses = images.load_frames(NUMBER_OF_NODES, FRAMES_PER_NODE)

print 'Train KNN...'
knn_model = knn.train(data, responses)

print 'Predict...'
cap = videos.load_video(TEST_VIDEO)
actual_node = 1
next_node = actual_node + 1
node_detected = -1
times_detected = 0

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = images.get_hist(gray)
    sample = np.array([hist])
    retu, results, neighbours, dist = knn.predict(knn_model, sample, 1, max_distance=3.5)
    node_detected, times_detected = compute_node_detected(node_detected, results[0][0], times_detected)
    frame_rotated = images.rotate_image(frame, -90)

    if node_detected != -1:
        #print 'Nodo predecido: ' + str(int(node_detected))

        if node_detected == next_node or (node_detected > actual_node and times_detected >= 7):
            actual_node = node_detected
            next_node = actual_node + 1

    actual_node_name, next_node_name = get_named_nodes(actual_node, next_node)

    cv2.putText(frame_rotated, actual_node_name, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (200, 255, 155), 13, cv2.LINE_AA)
    cv2.putText(frame_rotated, next_node_name, (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 3, (200, 255, 155), 13, cv2.LINE_AA)
    cv2.namedWindow('Test Dynamic', cv2.WINDOW_FULLSCREEN)
    #cv2.resizeWindow('Test Dynamic', 300, 300)
    cv2.imshow('Test Dynamic', frame_rotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

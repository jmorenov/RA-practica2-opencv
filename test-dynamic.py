import images
import knn
import videos
import cv2
import numpy as np

def get_named_node(result):
    name = ''
    if results[0][0] != -1:
        name = 'Nodo ' + str(int(results[0][0]))

    return name

FRAMES_PER_NODE = 115
NUMBER_OF_NODES = 7
TEST_VIDEO = 'Videos/RecorridoCompleto_1.MOV'

print 'Loading images...'
data, responses = images.load_frames(NUMBER_OF_NODES, FRAMES_PER_NODE)

print 'Train KNN...'
knn_model = knn.train(data, responses)

print 'Predict...'
cap = videos.load_video(TEST_VIDEO)

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = images.get_hist(gray)
    sample = np.array([hist])
    retu, results, neighbours, dist = knn.predict(knn_model, sample, 9, max_distance=6)
    frame_rotated = images.rotate_image(frame, -90)
    named_node = get_named_node(results[0][0])

    cv2.putText(frame_rotated, named_node, (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 6, (200, 255, 155), 13, cv2.LINE_AA)
    cv2.namedWindow('Test Dynamic', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Test Dynamic', 600, 600)
    cv2.imshow('Test Dynamic', frame_rotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

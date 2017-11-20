import cv2
import glob
import os

def load_videos_filename(directory, filePattern):
    return glob.glob(directory + filePattern)

def load_video(file):
    return cv2.VideoCapture(file)

VideosDirectory = "Videos/"
FilePattern = "Nodo_*.MOV"
ListOfVideos = load_videos_filename(VideosDirectory, FilePattern)

for i in range(len(ListOfVideos)):
    cap = load_video(ListOfVideos[i])
    nframes = 0

    while(cap.isOpened() and nframes <= 114):
        ret, frame = cap.read()

        if ret == False:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_name = 'imagen' + str(nframes+1) + '.jpg'
        directory_to_save = 'Frames/Nodo_' + str(i+1) + '/'

        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)

        cv2.imwrite(directory_to_save + frame_name, gray)

        nframes += 1
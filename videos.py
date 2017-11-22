import cv2
import glob

def load_videos_filename(directory, filePattern):
    return glob.glob(directory + filePattern)

def load_video(file):
    return cv2.VideoCapture(file)
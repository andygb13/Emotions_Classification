import cv2
import numpy as np
import torch
from tensorflow import keras

# Constants for model file paths
CLASSIFIER_PATH = './model_files/best_model40'
OBJECT_DETECTION_PATH = './model_files/best.pt'
VIDEO_NAME = 'sample1'
VIDEO_PATH = './video_in/' + VIDEO_NAME + '.mp4'

# Load Emotions classification model
cl_model = keras.models.load_model(CLASSIFIER_PATH)

# Load Object detection model
od_model = torch.hub.load('ultralytics/yolov5', 'custom', path=OBJECT_DETECTION_PATH)

def normalize_data(data):
    """Normalize image data to range (0, 1)."""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def process_img(sample):
    """Process input image for the classifier."""
    if len(sample.shape) > 2:
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    img = normalize_data(sample)
    img = cv2.resize(img, (300, 120))
    img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)
    return img

def get_eyes(img):
    """Detect eyes and return bounding box coordinates."""
    results = od_model(img).pandas().xyxy[0]
    if len(results) > 0:
        x1, y1, x2, y2 = map(int, results.iloc[0, :4])
        flag = True
    else:
        x1 = y1 = x2 = y2 = 0
        flag = False
    return x1, x2, y1, y2, img[y1:y2, x1:x2], flag

def find_emotion(eyes):
    """Find emotion using the classifier."""
    eyes = process_img(eyes)
    classes = np.array(['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
    results = cl_model.predict(eyes)
    prediction = classes[np.argmax(results)]
    return prediction

def main():
    """Main function to process the video and detect emotions."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('./video_out/' + VIDEO_NAME + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

    while(True):
        ret, frame = cap.read()
 
        if ret == True: 
            x1,x2,y1,y2,eyes,detection = get_eyes(frame)
            if detection:
                pred = find_emotion(eyes)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame,pred,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(36,255,12),3)
            out.write(frame)
        # Break the loop
        else:
            break 

    cap.release()
    out.release()

if __name__ == "__main__":
    main()

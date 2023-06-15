import cv2
import numpy as np


def extract_face(image_path):
    # Load the pre-trained deep learning face detector
    model_path = 'deploy.prototxt'
    weights_path = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
    net = cv2.dnn.readNetFromCaffe(model_path, weights_path)

    # Read the image
    image = cv2.imread(image_path)

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Set the blob as input to the network
    net.setInput(blob)

    # Forward pass through the network to detect faces
    detections = net.forward()

    # Extract the first detected face (assuming only one face is present)
    if detections.shape[2] > 0:
        box = detections[0, 0, 0, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        (x, y, w, h) = box.astype(int)
        face = image[y:y+h, x:x+w]
        return face

    return None

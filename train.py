import cv2

import numpy as np


# Your dataset: list of face images, where each image is read using cv2.imread() or similar

# Make sure the file paths are correct and the images exist

dataset = [cv2.imread("/content/Irfan.jpeg", cv2.IMREAD_GRAYSCALE),
           
           cv2.imread("/content/alfiya.jpeg",cv2.IMREAD_GRAYSCALE)]

# Check if images were loaded correctly

for i, img in enumerate(dataset):
   
	if img is None:
       
		print(f"Failed to load image {i+1}")


# Corresponding labels: list of labels, where each label is an integer

labels = [1, 2]


# Initialize the face recognizer

recognizer = cv2.face.LBPHFaceRecognizer_create()


# Train the model if all images were loaded successfully
if all(img is not None for img in dataset):
    recognizer.train(dataset, np.array(labels))

    # Save the trained model
    recognizer.save('model.yml')
else:
    print("Model training skipped due to image loading errors.")

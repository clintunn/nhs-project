import cv2
import os
import numpy as np

# Load the Haarcascade classifier for license plate detection
cascade_path = r'C:\Users\User\Desktop\NHS\haarcascade_licence_plate_rus_16stages.xml'
plate_cascade = cv2.CascadeClassifier(cascade_path)

# Set up video capture
cap = cv2.VideoCapture(r'C:\Users\User\Desktop\NHS\motor.mp4')

# Create a folder to save snapshots
output_folder = 'snapshots'
os.makedirs(output_folder, exist_ok=True)

# Function to detect license plates
def detect_license_plates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, 1.1, 10)
    return plates

# Function to compute similarity between two images
def compute_similarity(img1, img2):
    return np.mean(np.abs(img1 - img2))

# Initialize variables
frame_count = 0
plates_count = 0
saved_plates = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    # Detect license plates in the frame
    plates = detect_license_plates(frame)

    if len(plates) > 0:
        plates_count += 1
        print(f"Frame {frame_count}: {len(plates)} license plate(s) detected")
    
    frame_count += 1

    # Check for user input
    key = input("Press 's' to save, 'q' to quit, or any other key to continue: ")
    
    if key.lower() == 'q':
        print("Quitting...")
        break
    elif key.lower() == 's':
        for (x, y, w, h) in plates:
            plate_img = frame[y:y + h, x:x + w]
            
            # Check for similarity with previously saved plates
            is_duplicate = False
            for saved_plate in saved_plates:
                if compute_similarity(cv2.resize(plate_img, (100, 30)), saved_plate) < 0.1:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                snapshot_path = os.path.join(output_folder, f'plate_{plates_count}.jpg')
                cv2.imwrite(snapshot_path, plate_img)
                print(f"Saved snapshot: {snapshot_path}")
                saved_plates.append(cv2.resize(plate_img, (100, 30)))
            else:
                print("Duplicate plate detected, not saving.")

# Release the video capture
cap.release()

print(f"Video processing completed. Total frames: {frame_count}, Frames with plates: {plates_count}")
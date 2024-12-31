# Import required libraries
import os  # For creating directories and handling file paths
import cv2  # OpenCV library for video capture and image processing

# Define the directory where data (images) will be stored
DATA_DIR = './data'

# Check if the DATA_DIR exists; if not, create it
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Parameters for dataset collection
number_of_classes = 5  # Number of classes to collect images for
dataset_size = 100     # Number of images to collect per class

# Initialize the webcam (index 0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Loop through the number of classes
for j in range(number_of_classes):
    # Create a directory for each class inside DATA_DIR
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))  # Notify user of the current class

    # Wait for user input to start collecting images
    done = False
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Failed to capture image")
            continue  # Skip this loop iteration if frame capture fails

        # Display a message on the video feed to prompt user
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Show the live webcam feed

        # Wait for the user to press 'q' to proceed
        if cv2.waitKey(25) == ord('q'):
            break

    # Start capturing images for the current class
    counter = 0  # Initialize the image counter
    while counter < dataset_size:
        ret, frame = cap.read()  # Capture a new frame
        if not ret:
            print("Failed to capture image")
            continue

        cv2.imshow('frame', frame)  # Show the live feed for confirmation

        # Save the captured frame to the class directory
        img_path = os.path.join(class_dir, '{}.jpg'.format(counter))
        cv2.imwrite(img_path, frame)

        counter += 1  # Increment the counter
        cv2.waitKey(25)  # Small delay to control capture speed

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

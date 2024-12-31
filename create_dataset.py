# Import required libraries
import os         # For working with file and directory paths
import pickle     # For saving data to a file
import mediapipe as mp  # Mediapipe for hand landmarks detection
import cv2        # OpenCV for image reading and processing

# Initialize Mediapipe Hands solution
mp_hands = mp.solutions.hands  # Mediapipe Hands module
hands = mp_hands.Hands(
    static_image_mode=True,            # Process static images (not video streams)
    min_detection_confidence=0.3       # Minimum confidence for hand detection
)

# Directory where image data is stored
DATA_DIR = './data'

# Lists to store processed data and their respective labels
data = []   # Processed hand landmark coordinates
labels = [] # Corresponding class labels (directory names)

# Loop through each directory inside DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)  # Full path of the current directory

    # Skip if the current file is not a directory
    if not os.path.isdir(dir_path):
        print(f"Skipping non-directory file: {dir_path}")
        continue

    # Loop through each image file in the current directory
    for img_path in os.listdir(dir_path):
        data_aux = []  # Temporary list to hold normalized hand landmarks
        x_ = []        # List to store x-coordinates of landmarks
        y_ = []        # List to store y-coordinates of landmarks

        file_path = os.path.join(dir_path, img_path)  # Full path of the image file

        # Read the image using OpenCV
        img = cv2.imread(file_path)

        # Check if the image was loaded successfully
        if img is None:
            print(f"Warning: Unable to read file '{file_path}'. Skipping.")
            continue

        # Convert the image from BGR to RGB (required by Mediapipe)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe to detect hand landmarks
        results = hands.process(img_rgb)

        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            # Loop through detected hands (one or more)
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract x and y coordinates of each landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # Normalized x-coordinate
                    y = hand_landmarks.landmark[i].y  # Normalized y-coordinate

                    x_.append(x)  # Store x-coordinates
                    y_.append(y)  # Store y-coordinates

                # Normalize landmarks relative to the smallest x and y values
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normalize x
                    data_aux.append(y - min(y_))  # Normalize y

            # Append the processed landmarks and label to their respective lists
            data.append(data_aux)  # Store normalized hand landmarks
            labels.append(dir_)    # Store the class label (directory name)

# Save the processed data and labels to a file using pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset creation complete. Data saved to 'data.pickle'.")

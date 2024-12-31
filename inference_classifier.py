# Import necessary libraries
import pickle  # For loading the pre-trained model
import cv2  # OpenCV library for computer vision tasks
import mediapipe as mp  # Mediapipe for hand tracking
import numpy as np  # Numpy for numerical operations

# Load the trained model from a pickle file
model_dict = pickle.load(open('./model.p', 'rb'))  # Load model dictionary from file
model = model_dict['model']  # Extract the trained model

# Initialize video capture to access the webcam
cap = cv2.VideoCapture(0)  # 0 indicates the default webcam

# Initialize Mediapipe components for hand detection and drawing utilities
mp_hands = mp.solutions.hands  # Hands detection module
mp_drawing = mp.solutions.drawing_utils  # For drawing hand landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # Predefined styles for landmarks and connections

# Initialize the Hands model for static images with a minimum confidence threshold
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define a dictionary to map prediction outputs to corresponding labels (characters)
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

# Infinite loop to continuously process video frames from the webcam
while True:
    # Initialize auxiliary variables for storing hand landmarks
    data_aux = []  # List to store normalized landmark coordinates
    x_ = []  # List to store x-coordinates of landmarks
    y_ = []  # List to store y-coordinates of landmarks

    # Read a frame from the webcam
    ret, frame = cap.read()  # ret: success flag, frame: current video frame

    # Get the frame dimensions (Height, Width, and Channels)
    H, W, _ = frame.shape

    # Convert the BGR image to RGB (required by Mediapipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands to detect hand landmarks
    results = hands.process(frame_rgb)

    # Check if any hand landmarks are detected
    if results.multi_hand_landmarks:
        # Iterate through detected hands and draw landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks and hand connections on the frame
            mp_drawing.draw_landmarks(
                frame,  # Image to draw on
                hand_landmarks,  # Detected hand landmarks
                mp_hands.HAND_CONNECTIONS,  # Connections between landmarks
                mp_drawing_styles.get_default_hand_landmarks_style(),  # Landmark style
                mp_drawing_styles.get_default_hand_connections_style())  # Connection style

        # Collect landmark coordinates for further processing
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x  # Normalized x-coordinate
                y = hand_landmarks.landmark[i].y  # Normalized y-coordinate

                x_.append(x)  # Append x-coordinate to list
                y_.append(y)  # Append y-coordinate to list

            # Normalize the coordinates relative to the minimum x and y values
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # x normalized relative to min(x_)
                data_aux.append(y - min(y_))  # y normalized relative to min(y_)

        # Calculate bounding box around the detected hand
        x1 = int(min(x_) * W) - 10  # Top-left x-coordinate (scaled to frame width)
        y1 = int(min(y_) * H) - 10  # Top-left y-coordinate (scaled to frame height)
        x2 = int(max(x_) * W) - 10  # Bottom-right x-coordinate
        y2 = int(max(y_) * H) - 10  # Bottom-right y-coordinate

        # Predict the character using the trained model
        prediction = model.predict([np.asarray(data_aux)])  # Input normalized coordinates to the model

        # Retrieve the predicted character from the labels dictionary
        predicted_character = labels_dict[int(prediction[0])]

        # Draw the bounding box and display the predicted character on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  # Draw rectangle around the hand
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)  # Display predicted label

    # Display the processed frame
    cv2.imshow('frame', frame)

    # Wait for a short duration and check if a key is pressed to exit
    cv2.waitKey(1)

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

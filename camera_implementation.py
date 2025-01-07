import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('pretrained_V5.keras')

# Define 12 class labels
class_labels = [
    "apple",
    "banana",
    "carrot",
    "grapes",
    "kiwi",
    "lemon",
    "mango",
    "noodles",
    "orange",
    "pear",
    "powder",
    "strawberry"
]

# Discriptions for each feature item
item_info = {
    "apple": {"calories": 70, "price": "$1.00"},
    "banana": {"calories": 80, "price": "$0.50"},
    "carrot": {"calories": 25, "price": "$0.30"},
    "grapes": {"calories": 84, "price": "$2.00"},
    "kiwi": {"calories": 30, "price": "$0.75"},
    "lemon": {"calories": 10, "price": "$0.60"},
    "mango": {"calories": 110, "price": "$1.50"},
    "noodles": {"calories": 140, "price": "$2.98"},
    "orange": {"calories": 62, "price": "$0.80"},
    "pear": {"calories": 101, "price": "$1.20"},
    "powder": {"calories": 170, "price": "$25.99"},
    "strawberry": {"calories": 4, "price": "$0.10"}
}

# Confidence threshold for classification
CONFIDENCE_THRESHOLD = 0.75

# Open the webcam feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for model prediction
    preprocessed_frame = cv2.resize(frame, (224, 224))  # Resize to 224x224 for MobileNetV2
    preprocessed_frame = preprocessed_frame / 255.0  # Normalize
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension

    # Model prediction 
    class_pred = model.predict(preprocessed_frame)[0]

    # Print raw prediction values for each class
    print("Raw prediction values:", class_pred)

    # Process prediction
    predicted_class_index = np.argmax(class_pred)
    predicted_class_confidence = class_pred[predicted_class_index]

    # Get the class label from the class_labels list
    predicted_class_label = class_labels[predicted_class_index]

    # Print the predicted class and confidence
    print(f"Predicted class: {predicted_class_label} with confidence: {predicted_class_confidence:.2f}")

    if predicted_class_confidence >= CONFIDENCE_THRESHOLD:
        # Get item info
        calories = item_info[predicted_class_label]["calories"]
        price = item_info[predicted_class_label]["price"]

        # Display the information on the frame
        label = f"{predicted_class_label} ({predicted_class_confidence:.2f})"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Calories: {calories}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Price: {price}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam Object Detection', frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

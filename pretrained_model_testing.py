import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report

# Parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224  # Match input size of the saved model
BATCH_SIZE = 16  # Batch size for evaluation
NUM_CLASSES = 12  # Update if your model uses a different number of classes
MODEL_PATH = "pretrained_V5.keras"  # Path to your saved model
TEST_DIR = "fv_data/test"  # Path to the test dataset directory

# Load the saved model
model = tf.keras.models.load_model(MODEL_PATH)
print(model.summary())

# Prepare the test data generator
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)  # Normalize test data
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",  # Matches the saved model's training configuration
    shuffle=True  # Shuffle the test data for random predictions
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict on the test dataset
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Reset the generator to shuffle again
test_generator.reset()

# Print a classification report
print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Display predictions with true labels
def display_predictions_with_labels(generator, model, class_labels, num_images=16):
    """
    Display test images with their predicted and true labels.
    """
    # Shuffle the generator and retrieve one batch
    generator.reset()  # Reset generator to shuffle
    images, labels = next(generator)  # Get a new batch of shuffled images and labels
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)

    # Display images
    plt.figure(figsize=(20, 20))
    for i in range(min(num_images, len(images))):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        pred_label = class_labels[predicted_classes[i]]
        true_label = class_labels[true_classes[i]]
        plt.title(f"P: {pred_label} T: {true_label}")
    plt.show()

# Show predictions on shuffled test data
display_predictions_with_labels(test_generator, model, class_labels, num_images=16)

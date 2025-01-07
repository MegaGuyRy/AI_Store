
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Paths to datasets directories
train_dir = "fv_data/train"
val_dir = "fv_data/validation"
test_dir = "fv_data/test"

# Parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224  # Match input size expected of pretrained model MobileNetV2
BATCH_SIZE = 16 # Number of images evaluated before updating
NUM_CLASSES = 12  # Number of classes in the dataset


# Use ImageDataGenerator for data augmentation and preprocessing
train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    # Feature Normalization
    featurewise_center=False, # Computes mean of the entire dataset and subtracts it from all images
    samplewise_center=False, # Subtracts mean pixel value of each image individually
    featurewise_std_normalization=False, # Scales the entire dataset to a standard deviation of 1
    samplewise_std_normalization=False, # Scales each image to have a standard deviation of 1 individually
    zca_whitening=False, # Decorrelates the pixels of each image and emphasizes edges and textures while removing redundancy in pixel values
    zca_epsilon=1e-06, # Small constant used in the covariance matrix to avoid numerical instability
    # Geometric Transformations
    rotation_range=30, # Rotates the image within the specified degree
    width_shift_range=0.2, # Shifts the images horizontally by a fraction of the total image width
    height_shift_range=0.2, # Shifts the images vertically by a fraction of the total image height
    brightness_range=None, # Tuple specifying the range for random brightness adjustments Ex:(0.8, 1.2)
    shear_range=0.2, # Slants the image by a fraction of the total image size
    zoom_range=0.2, # Zooms into image with in a specified range
    channel_shift_range=0.0, # Range for random shifts in pixel values across color channels
    fill_mode="nearest", # Specifies how to fill in newly created pixels after transformations above Ex:(constant, nearest, refelct, wrap)
    cval=0.0, # Constant value when fill_mode=constant 
    # Augmentation 
    horizontal_flip=True, 
    vertical_flip=True,
    # Preprocessing
    rescale=1.0 / 255, # Normalize pixel values to [0, 1]
    #preprocessing_function=gaussian_blur_preprocessing, # Allows a custom function to be applyed to each image after augmentation
    data_format=None, # Specifies the image data format Ex: (height, width, channels)
    validation_split=0.0, # Split data for validation (Already done above)
    interpolation_order=1, # Mathematical technique used to estimate and assign pixel values when resizing an image 1:Bilinear, 3:Bicubic
    dtype=None # Data type of the output images 
    )

# Only need normalize for validation
val_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
    #preprocessing_function=gaussian_blur_preprocessing # Allows a custom function to be applyed to each image after augmentation

    )  

# Generate training and validation datasets
train_generator = train_data_gen.flow_from_directory(
    train_dir, # Directory path
    target_size=(IMG_HEIGHT, IMG_WIDTH), # Target size of images
    color_mode="rgb", # if images have color Ex: (RGB or grayscale)
    classes=None, # Manually assign class names
    class_mode="categorical", # Categorical for multi-class classification 
    batch_size=BATCH_SIZE, # Number of images in each batch
    shuffle=True, # Shuffles images and labels at the start of epoch
    seed=8, # Seed for shuffling 
    save_to_dir="aug_images", # Saves aumented images 
    save_prefix="aug_", # Prefex for saved augmented files
    save_format="png", # File save type 
    follow_links=False, # Whether to follow symbolic links in subdirectories
    subset=None, # Specifies whether to load a subset training or val data when using validation_split in ImageDataGenerator
    interpolation="nearest", # Method for resizing (nearest, bilinear, bicubic)
    keep_aspect_ratio=False # Whether to preserve the original aspect ratio of images when resizing
)

val_generator = val_data_gen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH), # Target size of images
    color_mode="rgb", # if images have color Ex: (RGB or grayscale)
    classes=None, # Manually assign class names
    class_mode="categorical", # Categorical for multi-class classification 
    batch_size=BATCH_SIZE, # Number of images in each batch
    shuffle=True, # Shuffles images and labels at the start of epoch
    seed=None, # Seed for shuffling 
    save_to_dir=None, # Saves aumented images 
    save_prefix="", # Prefex for saved augmented files
    save_format="png", # File save type 
    follow_links=False, # Whether to follow symbolic links in subdirectories
    subset=None, # Specifies whether to load a subset training or val data when using validation_split in ImageDataGenerator
    interpolation="nearest", # Method for resizing (nearest, bilinear, bicubic)
    keep_aspect_ratio=False # Whether to preserve the original aspect ratio of images when resizing
)

# Load the pre-trained model MobileNetV2 
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), # Shape of the images
    alpha=1.0,  # The number of filters in each convolution layer
    include_top=False,  # Excludes the classification layers
    weights="imagenet",  # Loads pre-trained weights trained on the ImageNet dataset
    input_tensor=None, # Specifies an optional Keras tensor as the input to the model
    pooling=None, # Defines the type of pooling to apply after the convolutional layers
    classes=1000, # Number of output classes (when include_top=False, replace classes)
    classifier_activation='softmax' # Activation function (when include_top=False, replaced)
)
print(base_model.summary())

# Prevent weights of pre-trained model from being updated
base_model.trainable = False 

# Add new layers for classification 
model = models.Sequential([
    base_model, # Load base weights
    layers.GlobalAveragePooling2D(), # Converts the feature maps from the base_model into a single feature vector by image
    layers.Dense(128, activation='relu'), # Adds a fully connected layer with ReLU activation function
    layers.Dropout(0.5), # Randomly sets a % of the neurons in the previous layer to zero during training
    layers.Dense(NUM_CLASSES, activation='softmax')  # Final classification layer = to num of classes
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Use Adam optimization function and define learning rate for weight updating
    loss="categorical_crossentropy", # Loss formula 
    loss_weights=None,  # Allows assignment of different weights to the loss functions
    metrics=["accuracy"], # Metrics that are monitored 
    weighted_metrics=None, # Allows specification of metrics that take into account sample weight
    run_eagerly=False, # Specifies to execute the model in eager execution mode or with TensorFlow’s graph execution
    steps_per_execution=1, # How many batches are processed before updating
    jit_compile='auto', #  XLA (Accelerated Linear Algebra) compilation for better performance
    auto_scale_loss=True #  Automatically scales the loss to account for mixed precision training
)

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='loss',  # What metric is being monitored for imporvment (loss, val_loss, accuracy, val_accuracy)
    min_delta=0, # The minimum change in metric to be considered improvment
    patience=5,  # Stop training after not seeing improvement over epochs 
    verbose=1, # Indicates if training ended early
    mode='auto', # How to determine if the monitored metric is improving (auto, max, min)
    baseline=None, # Min metric value for training to be eligable for early stopping
    restore_best_weights=True, # Restore the best weights when stopping
    start_from_epoch=8 # What epoch early stopping starts monitoring from
)

# Train the new top layers
history = model.fit(
    train_generator, # Supply model with training data 
    validation_data=val_generator, # Supply model with data for validation
    epochs=20, # Num or epochs the top layers are trained for 
    steps_per_epoch=train_generator.samples // BATCH_SIZE, # The number of batches to process in each epoch
    validation_steps=val_generator.samples // BATCH_SIZE, # Num of validation batches processed after each epoch
    callbacks=[early_stopping]  # List of callback function that can be executed during training
)

# Unfreeze the base model layers for updating
base_model.trainable = True

# TensorFlow requires recompilation whenever the trainability of layers changes
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    
)

# Fit the whole model together for training 
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,  # Allow for more epochs now that whole model it unfrozen
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping]  # Early stopping
)

# Save the retrained model
model.save("pretrained_V4.keras")

# Evaluate the model on the validation dataset
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Test the model on the test dataset
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)  # Normalize test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.2f}")

# --- Matplotlib Section: Plot Training vs Validation Accuracy ---
# Use only fine-tuning metrics for plotting
# Ensure epochs match the length of the recorded fine-tuning metrics
fine_epochs = range(1, len(history_fine.history['accuracy']) + 1)

# Extract fine-tuning metrics
fine_acc = history_fine.history['accuracy']
fine_val_acc = history_fine.history['val_accuracy']
fine_loss = history_fine.history['loss']
fine_val_loss = history_fine.history['val_loss']

# Plot the accuracies (Fine-tuning only)
plt.figure(figsize=(8, 6))
plt.plot(fine_epochs, fine_acc, 'bo-', label='Training Accuracy (Fine-tuning)')
plt.plot(fine_epochs, fine_val_acc, 'ro-', label='Validation Accuracy (Fine-tuning)')
plt.title('Fine-Tuning: Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot the losses (Fine-tuning only)
plt.figure(figsize=(8, 6))
plt.plot(fine_epochs, fine_loss, 'bo-', label='Training Loss (Fine-tuning)')
plt.plot(fine_epochs, fine_val_loss, 'ro-', label='Validation Loss (Fine-tuning)')
plt.title('Fine-Tuning: Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

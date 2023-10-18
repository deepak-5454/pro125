import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the augmentation parameters
datagen = ImageDataGenerator(
    zoom_range=0.2,        
    horizontal_flip=True   
)

# Load and preprocess your dataset
image_directory = '/content/drive/your_dataset_folder'  
data_generator = datagen.flow_from_directory(
    image_directory,
    target_size=(150, 150), 
    batch_size=32,          
    class_mode='binary'     
)

# Generate augmented images
images, labels = next(data_generator)

# Visualize the augmented images
plt.figure(figsize=(12, 8))
for i in range(0, len(images)):
    plt.subplot(4, 8, i + 1)
    plt.imshow(images[i])
    plt.axis('off')

plt.show()

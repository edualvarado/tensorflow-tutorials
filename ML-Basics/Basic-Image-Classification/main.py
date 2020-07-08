# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from plot import plot_image, plot_value_array

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print("==========================")
print("TensorFlow version: ", tf.__version__)
print("==========================")

print("\nImporting MNIST Dataset...")
print("==========================")
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Training set size: ", train_images.shape)
print("Number of training labels: ", len(train_labels))
print("Training Labels: ", train_labels)
print("Test set size: ", test_images.shape)
print("Number of test labels: ", len(test_labels))
print("Test labels: ", test_labels)

# Normalize the values of each pixel from 0-255 scale to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# To plot one image
'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''

# To plot a grid with sample images
'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

print("\nBuilding model...")
print("==========================")
# Transforms from 2D-array to 1D-array (784 pixels) -> FCN
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

print("\nCompiling model...")
print("==========================")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("\nTraining model...")
print("==========================")
model.fit(train_images, train_labels, epochs=1)

print("\nEvaluating model...")
print("==========================")
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

print("\nAdding softmax layers...")
print("==========================")
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

print("\nPlotting predictions...")
print("==========================")

'''
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
'''

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

print("\nOne single prediction...")
print("==========================")
# Make prediction about single image
img = test_images[1]
print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)

# Predict for certain image
predictions_single = probability_model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
print("Prediction: ", np.argmax(predictions_single[0]))
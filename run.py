from pkgutil import extend_path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, load_img
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.activations import relu, sigmoid
from matplotlib import pyplot as plt
import numpy as np
import os

valid_formats = ['.jpg', '.png', '.jpeg']
path = "SMILEs"

def image_paths(root):
    image_paths = []
    # dirpath is the path to the directory, dirnames is a list of the subdirectories, and filenames is a list of the files in the directory
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            extension = os.path.splitext(filename)[1].lower()
            if extension in valid_formats:
                image_path = os.path.join(dirpath, filename)
                image_paths.append(image_path)
    return image_paths

imagen_size = [32, 32]
images_paths = image_paths(path)

def load_dataset(image_paths, target_size=imagen_size):
    images = []
    labels = []
    for image_path in image_paths:
        image = load_img(image_path, target_size=target_size,
                         color_mode="grayscale")
        image = img_to_array(image)
        images.append(image)
        label = image_path.split(os.path.sep)[-3]
        label = 1 if label == 'positives' else 0
        labels.append(label)
    normalize_images = np.array(images) / 255.0

    return np.array(normalize_images), np.array(labels)

# load the dataset  images as numpy arrays and labels of any images
images_dataset, labels = load_dataset(images_paths)


def build_model(input_shape :list[int]= imagen_size + [1]):
    model = Sequential()
    # model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(32, (3, 3), padding="same"))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(64))
    # model.add(Activation("relu"))
    # model.add(Dense(classes))
    # model.add(Activation("softmax"))
    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation=relu,
        padding="same",
        input_shape=input_shape
    ))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        padding="same"
    ))
    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation=relu,
        padding="same",
        input_shape=input_shape
    ))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
       padding="same"
    ))
    model.add(Flatten())
    model.add(Dense(256, activation=relu))
    model.add(Dense(1, activation=sigmoid))

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model


# calculate the counts for each unque label
label, counts = np.unique(labels, return_counts=True)
# counts is a list with len(images) and len(labels)

counts = max(counts) / counts
class_weight = dict(zip(label, counts))

(x_train, x_test, y_train, y_test) = train_test_split(images_dataset, labels,
                                                      test_size=0.2,
                                                      stratify=labels,
                                                      random_state=42
                                                      )
(x_train, x_valid, y_train, y_valid) = train_test_split(images_dataset, labels,
                                                        test_size=0.2,
                                                        stratify=labels,
                                                        random_state=42
                                                        )
model = build_model()
epoch = 20
history = model.fit(
    x_train, y_train,
    validation_data=(x_valid, y_valid),
    epochs=epoch,
    batch_size=64,
    class_weight=class_weight
)
model.save("model")
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(range(epoch), acc, "b", label="Training Accuracy")
plt.plot(range(epoch), val_acc, "r", label="Validation Accuracy")
plt.legend()

plt.figure()
plt.savefig("training acurracy.jpg")
plt.plot(range(epoch), loss, "g", label="Training Loss")
plt.plot(range(epoch), val_loss, "orange", label="Validation Loss")
plt.legend()


plt.savefig("training loss.jpg")
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("test loss: ", test_loss)
print("test accuracy: ", test_accuracy)
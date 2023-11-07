########### test img load
from keras.preprocessing import image
from keras.models import load_model  # TensorFlow is required for Keras to work

import numpy as np
import cv2


def genger_prediction(img_name):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("./model/keras_model.h5", compile=False)
    print(model)
    # Replace this with the path to your image
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image

    img = image.load_img(
        f"./img/{img_name}.jpg",
        target_size=(224, 224),
    )
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)

    prediction_class = prediction.argmax(axis=-1)

    if prediction_class == 0:
        return "남성"
    elif prediction_class == 1:
        return "여성"


def genderV2(img):
    image = cv2.imread(
        f"./img/{img}.jpg",
    )
    model = load_model("./model/gender_v2.h5", compile=False)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("./input/haarcascade_frontalface_default_2.xml")
    faces = cascade.detectMultiScale(gray, 1.1, 7)  # 1.1

    for x, y, w, h in faces:
        face = image[y : y + h, x : x + w]
        face = cv2.resize(face, (150, 150))
        img_scaled = face / 255.0
        reshape = np.reshape(img_scaled, (1, 150, 150, 3))
        img = np.vstack([reshape])
        result = model.predict(img)
        print(result)
        index = np.argmax(result)
        prediction_class = result.argmax(axis=-1)
        if prediction_class == 0:
            return "여성"
        elif prediction_class == 1:
            return "남성"

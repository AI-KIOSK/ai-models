from keras.preprocessing import image
from keras.models import load_model  # TensorFlow is required for Keras to work

import numpy as np


def eastern_age_prediction(img_name):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    model = load_model("./model/easternAgeClassification.h5", compile=False)

    # Replace this with the path to your image
    img = image.load_img(f"./img/{img_name}.jpg", target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # 이미지를 모델에 입력하여 예측 수행
    predictions = model.predict(img_array)
    print(predictions)

    prediction_class = predictions.argmax(axis=-1)
    # ------------

    # print(prediction_class)
    # ------------class: 0(1-20) / 1(21-35) / 2(36-60) / 3(61 ~)
    if prediction_class == 0 or prediction_class == 1:
        return "청년"
    elif prediction_class == 2:
        return "중년"
    elif prediction_class == 3:
        return "노년"


def western_age_prediction(img_name):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    model = load_model("./model/westernAgeClassificationModel.h5", compile=False)

    # Replace this with the path to your image
    img = image.load_img(f"./img/{img_name}.jpg", target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # 이미지를 모델에 입력하여 예측 수행
    predictions = model.predict(img_array)
    print(predictions)

    prediction_class = predictions.argmax(axis=-1)
    # ------------
    # ------------class: 0(1-20) / 1(21-35) / 2(36-60) / 3(61 ~)
    if prediction_class == 0 or prediction_class == 1:
        return "청년"
    elif prediction_class == 2:
        return "중년"
    elif prediction_class == 3:
        return "노년"


def total_age_prediction(img_name):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    model = load_model("./model/ageClassification.h5", compile=False)

    # Replace this with the path to your image
    img = image.load_img(f"./img/{img_name}.jpg", target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # 이미지를 모델에 입력하여 예측 수행
    predictions = model.predict(img_array)
    print(predictions)

    prediction_class = predictions.argmax(axis=-1)
    # ------------
    # ------------class: 0(1-20) / 1(21-35) / 2(36-60) / 3(61 ~)
    if prediction_class == 0 or prediction_class == 1:
        return "청년"
    elif prediction_class == 2:
        return "중년"
    elif prediction_class == 3:
        return "노년"

from keras.preprocessing import image
from keras.models import load_model  # TensorFlow is required for Keras to work

import numpy as np


def nationality_prediction(img_name):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    model = load_model("./model/nationalityClassificationModel.h5", compile=False)

    # Replace this with the path to your image
    img = image.load_img(f"./img/{img_name}.jpg", target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # 이미지를 모델에 입력하여 예측 수행
    predictions = model.predict(img_array)
    print(predictions)

    index = np.argmax(predictions)
    print(index)
    if index == 0:
        return "EASTERN"
    elif index == 1:
        return "WESTERN"

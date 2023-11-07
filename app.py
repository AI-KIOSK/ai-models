from flask import Flask, request, jsonify
from io import BytesIO
from datetime import datetime
import base64
from PIL import Image
from prediction.age import (
    eastern_age_prediction,
    total_age_prediction,
    western_age_prediction,
)

from prediction.gender import genderV2, genger_prediction
from prediction.nationality import nationality_prediction


app = Flask(__name__)


def save_base64_to_img(base64_img: str, path):
    img = Image.open(BytesIO(base64.b64decode(base64_img)))
    img.save(f"./img/{path}.jpg")


@app.route("/", methods=["POST"])
def hello():
    params = request.get_json()

    request_img = params.get("img")
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_base64_to_img(request_img, time_stamp)

    # gender_prediction_result = genger_prediction(img_name=time_stamp)
    gender_prediction_result = genderV2(time_stamp)

    print(gender_prediction_result)
    nationality_prediction_result = nationality_prediction(img_name=time_stamp)

    age_prediction_result = total_age_prediction(img_name=time_stamp)
    # if nationality_prediction_result == "EASTERN":
    #     age_prediction_result = eastern_age_prediction(img_name=time_stamp)
    # else:
    #     age_prediction_result = western_age_prediction(img_name=time_stamp)

    response_data = {
        "gender": gender_prediction_result,
        "nationality": nationality_prediction_result,
        "age": age_prediction_result,
    }
    print(response_data)
    return jsonify(response_data)

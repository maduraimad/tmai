from flask import jsonify, Blueprint,abort
from flask_cors import CORS
from flask import Flask, request, send_file
import os
import random
import boto3
from botocore.client import Config
from PIL import Image
from predictor.predictions import predictions
from predictor.AnnoyCustomImageSearch import  AnnoyCustomImageSearch
import logging
import json
import io
from werkzeug.exceptions import HTTPException

base_folder = os.getenv("base_folder")
custom_image_upload=base_folder+'custom_image_upload/'

bp = Blueprint("app", __name__)
app = Flask(__name__)

CORS(app)
prediction = predictions()
annoyCustomImageSearch= AnnoyCustomImageSearch()
log = logging.getLogger()

@bp.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response


@bp.route('/')
def hello_world():
    return 'Hello, World!'


@bp.route('/similaritySearch/<serial_number>', methods=["GET"])
def get_prediction_by_number(serial_number):
    predicted_results = {}
    if prediction.check_file_exist(serial_number):
        results = prediction.get_predcition_by_number(serial_number)
        predicted_results = prediction.generate_signed_url(serial_number, results, False, "",False)
    else:
        predicted_results['serialNumber'] = serial_number
        predicted_results['predictions'] = []
    return jsonify(predicted_results)


@bp.route('/similaritySearch/image', methods=["POST"])
def get_prediction_by_image():
    fill_color = (255, 255, 255)
    predictions_results = {}
    try:
        if "file" in request.files:
            file = request.files['file']
            serial_number = 'C' + str(random.randint(1000000, 9999999))
            im = Image.open(file)
            im = im.convert("RGBA")
            if im.mode in ('RGBA', 'LA'):
                background = Image.new(im.mode[:-1], im.size, fill_color)
                background.paste(im, im.split()[-1])  # omit transparency
                im = background
            im.convert("RGB").save(custom_image_upload + serial_number + '.jpg')
            log.info("predciting result for uploaded image with serial number "+serial_number)
            image_feature = annoyCustomImageSearch.extract_features(custom_image_upload + serial_number + '.jpg')
            res = annoyCustomImageSearch.get_nns_by_vector(image_feature)[:1000]
            predictions_results = prediction.generate_signed_url(serial_number, res, True, "",False)
        return jsonify(predictions_results)
    except Exception as e:
        log.error("Error occurred while processing the image with serial_number "+str(e))
        raise e


@bp.route("/feedback", methods=["POST"])
def save_user_feedback():
    feedback = []
    serialnumber = None
    emailId = None
    data = request.get_json(force=True)
    try:
        if data['serialNumber']:
            serialnumber = data['serialNumber']
            emailId = data['emailId']
        if len(data['feedback']) >= 1:
            log.info("saving feedback of serialnumber "+serialnumber)
            feedback = data['feedback']
        return jsonify(prediction.persist_feedback(serialnumber, emailId, feedback))
    except Exception as e:
        log.error("Error occurred while saving feedback "+e)
        raise e


@bp.route('/similaritySearch/', methods=["POST"])
def get_prediction_by_post_number():
    log.info("processing serial number")
    serial_number = None
    email_id = None
    predicted_results = {}
    data = request.get_json(force=True)
    try:
        if data['serialNumber']:
            serial_number = data["serialNumber"]
            email_id = data['emailId']
            if prediction.check_file_exist(serial_number):
                log.info("processing results for serial number using json " + serial_number)
                results = prediction.get_predcition_by_number(serial_number)[1:]
                predicted_results = prediction.generate_signed_url(serial_number, results, False, email_id,False)
            elif prediction.check_image_s3(serial_number):
                log.info("processing results for serial number using  image" + serial_number)
                file_stream_string=prediction.get_file_content_s3(serial_number)
                image_feature = annoyCustomImageSearch.extract_features_s3(file_stream_string)
                res = annoyCustomImageSearch.get_nns_by_vector(image_feature)
                predicted_results = prediction.generate_signed_url(serial_number, res, True, "",True)
            else:
                predicted_results['serialNumber'] = serial_number
                predicted_results['predictions'] = []
        return jsonify(predicted_results)
    except Exception as e:
        log.error("Error occurred while retrieving  results for serial number "+str(e))
        raise e



@bp.route('/serve/<filename>', methods=["GET"])
def download_file(filename):
    session = boto3.Session()
    s3_client = session.client('s3', config=Config(signature_version='s3v4'))
    file_byte_string = s3_client.get_object(Bucket='uspto-tm-img-search', Key='tm_images/' + filename + '.jpg')['Body'].read()
    return send_file(io.BytesIO(file_byte_string), mimetype='image/jpeg', as_attachment=True,attachment_filename=filename+'.jpg')


app_prefix = os.getenv("app_prefix", "")
app.register_blueprint(bp, url_prefix=app_prefix)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5001,threaded=False)

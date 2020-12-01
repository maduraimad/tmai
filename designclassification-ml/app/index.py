from flask import jsonify, Blueprint
from predictors.ClassificationPredictor import *
from flask_cors import CORS
from flask import Flask, request
import random
from PIL import Image
import os


bp = Blueprint("app", __name__)
app = Flask(__name__)
CORS(app)

base_folder = os.getenv("base_ensemble_folder")
custom_image_upload = base_folder+"/custom_image_upload/"
custom_mark_desc=base_folder+"/custom_mark_desc/"


ensemble_predictor = EnsemblePredictor()



@bp.route('/')
def hello_world():
    return 'Hello, World!'


@bp.route("/designCodes/serialnumber", methods=["POST"])
def get_design_codes():
    result=[]
    data = request.get_json(force=True)
    if data['serialNumber']:
     serial_number = data['serialNumber']
     email_id = data['emailId']
     result = ensemble_predictor.get_predictions_by_serial(serial_number,email_id)
    return jsonify(result)

@bp.route("/designCodes", methods=["POST"])
def get_design_codes_for_custom_upload():
    image_buffer = None
    fill_color = (255, 255, 255)
    serial_number = 'C' + str(random.randint(1000000, 9999999))
    if "file" in request.files:
        file = request.files['file']
        image_buffer = bytearray(file.read())
        im = Image.open(file)
        im = im.convert("RGBA")
        if im.mode in ('RGBA', 'LA'):
            background = Image.new(im.mode[:-1], im.size, fill_color)
            background.paste(im, im.split()[-1])  # omit transparency
            im = background
            im.convert("RGB").save(custom_image_upload + serial_number + '.jpg')
    mark_description = None
    if "markDescription" in request.form:
        if len(request.form["markDescription"])>2:
         mark_description = request.form["markDescription"]
         with open(custom_mark_desc+serial_number+".txt", 'a') as f: f.write(mark_description)
    result = ensemble_predictor.get_predictions(image_buffer=image_buffer, mark_desc=mark_description,serial_number=serial_number)
    return jsonify(result)
'''
@bp.route("/designCodes/text/<serial_number>", methods=["GET"])
def get_design_codes_using_text_model(serial_number):
    result = ensemble_predictor.get_text_predictions(serial_number=serial_number)
    return jsonify(result)

@bp.route("/designCodes/text", methods=["POST"])
def get_design_codes_for_custom_upload_using_text_model():
    mark_description = None
    if "markDescription" in request.form:
        mark_description = request.form["markDescription"]
    result = ensemble_predictor.get_text_predictions(mark_desc=mark_description)
    return jsonify(result)
'''

@bp.route("/designCodes/feedback", methods=["POST"])
def save_user_feedback():
    feedback = []
    serialnumber = ''
    emaild = ''
    data = request.get_json(force=True)
    if data['serialNumber']:
        serialnumber = data['serialNumber']
        emailId = data['emailId']
    if len(data['feedback']) >= 1:
        feedback = data['feedback']
    return jsonify(ensemble_predictor.persist_feedback(serialnumber, emailId, feedback))



app_prefix = os.getenv("app_prefix", "")
app.register_blueprint(bp, url_prefix=app_prefix)


if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)

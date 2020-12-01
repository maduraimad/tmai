from flask import jsonify, Blueprint,json
from werkzeug.exceptions import HTTPException
from predictor.GoodServicesPredictor import *
from flask_cors import CORS
from flask import Flask, request
import os
import random




bp = Blueprint("app", __name__)
app = Flask(__name__)
CORS(app)
base_folder = os.getenv("base_folder")
own_desc_dir=base_folder+"own-desc-dir/"


gs_predict = GoodServicesPredictor()

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

@bp.route("/gsdescription/<serial_number>", methods=["GET"])
def get_good_services_description(serial_number):
    return jsonify(gs_predict.get_trademark_info(serial_number))



@bp.route("/gsclassification", methods=["POST"])
def get_good_services_codes():
    gs_description = None
    serial_number = None
    email_id=None
    result = None
    text_comparassion=None
    negativeresult=[]
    data = request.get_json(force=True)
    if "serialNumber" in data :
        serial_number =data["serialNumber"]
        email_id = data['emailId']
        gs_description=gs_predict.get_trademark_info(serial_number)
        result = gs_predict.get_predictions(serial_number,email_id,gs_description)
        textComparison = gs_predict.processSimilarDescription(gs_description)
    elif "gsDescription" in data:
        gs_description = data["gsDescription"]
        serial_number ='C'+ str(random.randint(1000000, 9999999))
        email_id = data['emailId']
        file = open(own_desc_dir+serial_number+".txt", "a")
        file.write(gs_description)
        file.close()
        result = gs_predict.get_predictions(serial_number,email_id,gs_description)
        textComparison = gs_predict.processSimilarDescription(gs_description)
    else:
        result = ''
    return jsonify({'predictions':result,'serialNumber':serial_number,'text_comparison':textComparison})


@bp.route("/feedback", methods=["POST"])
def save_user_feedback():
      feedback=[]
      serialnumber=''
      emaild=''
      data = request.get_json(force=True)
      if  data['serialNumber']:
        serialnumber=data['serialNumber']
        emaild = data['emailId']
      if len(data['feedback'])>=1:
         feedback=data['feedback']
      return jsonify(gs_predict.persist_feedback(serialnumber,emaild,feedback))

@bp.route("/similarDescription", methods=["POST"])
def get_similar_description():
    description=''
    if "gsDescription" in request.form:
        description=request.form["gsDescription"]
    elif "serialnumber"in request.form:
       serial_number = request.form["serialnumber"]
       description=gs_predict.get_trademark_info(serial_number)
    else:
        description=''
    return jsonify({'desc':gs_predict.processSimilarDescription(description)})

app_prefix = os.getenv("app_prefix", "")
app.register_blueprint(bp, url_prefix=app_prefix)


if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)

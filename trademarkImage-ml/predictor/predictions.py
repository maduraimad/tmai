import pathlib
import boto3
from botocore.client import Config
from PIL import Image
from io import BytesIO
import os
import json
import urllib.request
import logging
import datetime
import pymysql


base_folder = os.getenv("base_folder")
json_loc='/trademarksdata/nn_results_1373dsc_network/'
custom_image_upload = base_folder+"custom_image_upload/"
mysql_database_user=os.getenv("MYSQL_DATABASE_USER")
mysql_database_password=os.getenv("MYSQL_DATABASE_PASSWORD")
mysql_database_db=os.getenv("MYSQL_DATABASE_DB")
mysql_database_data_db=os.getenv("MYSQL_DATABASE_DATA_DB")
mysql_database_host=os.getenv("MYSQL_DATABASE_HOST")
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('nose').setLevel(logging.CRITICAL)

log = logging.getLogger()
class predictions:  
    # return the predcited results by serail number
    def get_predcition_by_number(self, serialNumber):
        precicted_result = []
        file_path = json_loc + serialNumber + ".json"
        try:
            if os.path.exists(file_path):
                file = open(file_path)
                log.info("loading results from file for  serial number"+serialNumber)
                precicted_result = json.load(file)
            return precicted_result
        except Exception as e:
            log.error("Error occurred while accessing parsing json for serail number"+serialNumber)
            raise e

    # check predicted result for a serial number exist or not
    def check_file_exist(self, serialNumber):
        isExit = False
        if pathlib.Path(json_loc + serialNumber + '.json').exists():
            isExit = True
        else:
            isExit = False
        return isExit
    '''
    # check image for a serial number exist
    def check_image_exist(self, serialNumber):
        isExit = False
        try:
            session = boto3.Session()
            if pathlib.Path(image_path + serialNumber + '.jpg').exists():
                isExit = True
                s3_client = session.client('s3',config=Config(signature_version='s3v4'))
                with open(image_path + serialNumber + ".jpg",'rb') as f:
                    s3_client.upload_fileobj(f, 'uspto-tm-img-search','tm_images/'+serialNumber +".jpg")
            else:
                isExit = False
            return isExit
        except Exception as e:
          log.info("error ouccred while checking file or uploading an image To s3")'''

    def check_image_s3(self,serial_number):
        session = boto3.Session()
        s3_client = session.client('s3',config=Config(signature_version='s3v4'))
        response = s3_client.list_objects(Bucket='uspto-tm-img-search', Prefix='tm_images/' + serial_number + '.jpg')
        if ('Contents' in response):
            return True
        else:
            return False

    def get_file_content_s3(self,serial_number):
        session = boto3.Session()
        s3_client = session.client('s3', config=Config(signature_version='s3v4'))
        file_byte_string =s3_client.get_object(Bucket='uspto-tm-img-search', Key='tm_images/'+serial_number+'.jpg')['Body'].read()
        return file_byte_string

    # download image from TSDR if it doent exist in dataset
    def read_image_url(self, serial_number):
        val = False
        try:
            urllib.request.urlretrieve("https://tsdr.uspto.gov/img/" + serial_number + "/large?1589916761862",
                                       custom_image_upload + serial_number + ".jpg")
            val = True
        except:
            val = False
        return val


    def generate_signed_url(self, serial_number, prediction_data, isUpload, emailId, isNewImage):
     updated_signed_urls = []
     results = {}
     results['serialNumber'] = serial_number
     hasFeedback = False
     feedBackData = []
     session = boto3.Session()
     prediction_data=self.update_trademark_status(prediction_data)
     s3_client =session.client('s3',config=Config(signature_version='s3v4'))
     log.info("genrating signed url for the serial number" + serial_number)
     if isNewImage:
        with open(json_loc + serial_number + '.json', 'w') as outfile:
            json.dump(prediction_data, outfile)
     try:
        if emailId:
            db_feedback = self.get_feedback(serial_number, emailId)
            if not db_feedback:
                hasFeedback = False
            else:
                hasFeedback = True
                feedBackData = ",".join(map(str, db_feedback)).split(",")
        for i in prediction_data:
       
            signed_url = s3_client.generate_presigned_url('get_object', Params={'Bucket': 'uspto-tm-img-search',
                                                                                'Key': 'tm_images/' + i[
                                                                                    'filename'] + '.jpg'},
                                                          ExpiresIn=1800)
            i['url'] = signed_url
            if isUpload == True:
                i['similarity'] = ''
                i['selected'] = False
            else:
                i['selected'] = self.isValidCheck(hasFeedback, feedBackData, i['filename'])
            updated_signed_urls.append(i)
        results['predictions'] = updated_signed_urls;
        return results
     except Exception as e:
        log.error("Error occurred while generating the signed urls for  serial number" + str(e))
        raise e

    def update_trademark_status(self,predicted_result):
       try:
        conn = self.get_connection(mysql_database_data_db)
        cursor = conn.cursor()
        serialnumbers = [i['filename'] for i in predicted_result]
        sql = 'SELECT `serial_number`,`tm_status`,`tm_status_desc` FROM `trademark_app_info` WHERE `serial_number`  IN (%s)'
        in_p = ', '.join(list(map(lambda x: '%s', serialnumbers)))
        sql = sql % in_p
        cursor.execute(sql, serialnumbers)
        tmstatus_results = cursor.fetchall()
        for item in tmstatus_results:
            for element in predicted_result:
                if str(item[0]) == element['filename']:
                    element['status'] = int(item[1])
                    element['statusDesc'] = item[2]
        return predicted_result
       except Exception as e:
         log.error("Error ocurred while updating the predicted results with status"+str(e))
       finally:
            cursor.close()
            conn.close()

    def get_feedback(self, serial_number, emailId):
        result = []
        try:
            conn = self.get_connection(mysql_database_db)
            cursor = conn.cursor()
            log.info("retrieving results from datbase for serial numebr "+serial_number+" emailId "+emailId)
            sql = "SELECT `negative_serial_numbers` FROM `user_feedback` WHERE `email_id`=%s and `serial_number`=%s"
            cursor.execute(sql, (emailId, serial_number,))
            data = cursor.fetchone()
            return data
        except Exception as e:
            log.error("Error occurred while getting data from database "+str(e))
            raise e
        finally:
            cursor.close()
            conn.close()

    def get_connection(self,mysql_database_schema):
        try:
            connection = pymysql.connect(mysql_database_host, mysql_database_user, mysql_database_password,
                                         mysql_database_schema)
            return connection
        except Exception as e:
            log.error("Error occurred while making connection"+str(e))
            raise e

    def isValidCheck(self, hasfeedback, feedbackData, filename):
        state = ''
        if hasfeedback == True and (str(filename) in feedbackData):
            state = True
        else:
            state = False
        return state

    def persist_feedback(self, serial_number, emailId, userfeedback):
        neg_list = []
        rep = {'success': True, 'message': 'successfully saved your feedback'}
        log.info("saving feedback to the database for serial number "+serial_number+ "emailId"+ emailId)
        try:
            conn = self.get_connection(mysql_database_db)
            cursor = conn.cursor()
            for i in userfeedback:
                neg_list.append(i)
            db_feedback = self.get_feedback(serial_number, emailId)
            if not db_feedback:
                if len(neg_list) >= 1:
                    currentDT = datetime.datetime.now()
                    sql = "INSERT INTO user_feedback (email_id,serial_number,negative_serial_numbers,timestamp ) VALUES (%s,%s,%s,%s)"
                    cursor.execute(sql,
                                   (emailId, serial_number, ','.join([elem for elem in neg_list]), str(currentDT),))
                    conn.commit()
                    return rep
            else:
                currentDT = datetime.datetime.now()
                sql = "UPDATE user_feedback SET negative_serial_numbers=%s,timestamp=%s WHERE email_id=%s and serial_number=%s"
                if len(neg_list) >= 1:
                    cursor.execute(sql, (
                        ','.join([elem for elem in neg_list]), str(currentDT), emailId, serial_number,))
                    conn.commit()
                    return rep
                else:
                    sql = "delete from user_feedback where email_id=%s and serial_number=%s"
                    cursor.execute(sql, (emailId, serial_number,))
                    conn.commit()
                    return rep
            return rep
        except Exception as e:
            log.error("Error occurred while saving or update or deleting feedback ",str(e))
            raise e
        finally:
            cursor.close()
            conn.close()

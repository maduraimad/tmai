import pymysql
import json
import configparser
config = configparser.ConfigParser()
filespath='/data1/json/'
path='/data1/gs_description_data/trademarkSimilarity/serialNumberExtractor/allSNs/NewImageSNs_2020-05-01_2020-10-06.txt'
connection = pymysql.connect(config['mysqlDB']['host'],config['mysqlDB']['user'],config['mysqlDB']['pass'],config['mysqlDB']['db'])
cursor = connection.cursor()
try:
  with open(path) as f:
    lines = [line.rstrip() for line in f]
  for file in lines:
    f = open(filespath+'json/'+file+'.json', "r")
    trademark = json.load(f)
    status_code=''
    status_desc=''
    status=''
    serial_number=''
    gs=''
    mark_desc=''
    gs_list=[]
    gs_desc=''
    original_labels=[]
    status = trademark['trademarks'][0]['status']
    serial_number=trademark['trademarks'][0]['status']['serialNumber']
    gs = trademark["trademarks"][0]["gsList"]
    if "descOfMark" in status:
        mark_desc=status["descOfMark"]
    if "tm5Status" in status:
       status_code=status["tm5Status"]
    if "tm5StatusDesc" in status:
        status_desc=status["tm5StatusDesc"]
    for i in gs:
        gs_list.append(str(i['description']))
    gs_list.reverse()
    gs_desc = ";".join(gs_list)
    original_labels = [item["code"] for item in status["designSearchList"]]
    mark=(serial_number,status_code,status_desc,gs_desc,mark_desc,original_labels,)
    sql="INSERT IGNORE INTO trademark_app_info(serial_number,tm_status,tm_status_desc,goods_services_desc,mark_desc,design_codes) VALUES(%s,%s,%s,%s,%s,%s)"
    cursor.execute(sql,(serial_number,status_code,status_desc,gs_desc,mark_desc,str(original_labels)))
    connection.commit()
except Exception as e:
   print("error occurred while loading",file)


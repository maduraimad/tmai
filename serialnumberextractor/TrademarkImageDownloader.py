import requests
import boto3
from botocore.client import Config
download_base_folder = "/data1/gs_description_data/trademarkSimilarity/tm_images_2020-05-01_2020-10-06_new"
s3_bucket = ""
save_in_hierarchy = True
serials_file_path = "/data1/gs_description_data/trademarkSimilarity/serialNumberExtractor/allSNs/NewImageSNs_2020-05-01_2020-10-06.txt"
session = boto3.Session()
s3_client = session.client('s3', config=Config(signature_version='s3v4'))
def download_image_by_serial(serial):
    image_prefix = ""
    if save_in_hierarchy:
        image_prefix = "{}/{}/".format(serial[0:4], serial[4:8])
    image_suffix = serial+".jpg"
    url = "http://tsdr.uspto.gov/img/" + serial + "/large";
    response = requests.get(url)
    image_data = response.content
    if s3_bucket is not None:
      dest_key = image_suffix
      s3_client.put_object(Body=image_data, Bucket=s3_bucket, Key="tm_images/"+dest_key)
   
   
def download_images_by_serials():
    total_failed = 0
    with open(serials_file_path, "r") as file:
        serials =  file.read().splitlines()
        print("Total serials to download - "+str(len(serials)))
    for index, serial in enumerate(serials):
        try:
            download_image_by_serial(serial)
            print("Downloaded {} - {}".format(str(index), serial))
        except Exception as e:
            print("Error downloading - "+serial+" "+str(e))
            total_failed += 1
    print("Done")
    if total_failed > 0:
        print("Total failed - "+str(total_failed))


download_images_by_serials()

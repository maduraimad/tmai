# designClassification-ml

## Installation

### Install Python 3

### Install Dependencies

cd tmai/designclassification-ml/ <br>
pip install --upgrade pip <br>
pip install -r requirements.txt

### Create directory
1373_new_model and sub directory  image_resources,text_resources,custom_image_upload and custom_mark_desc.

### Copy files into  1373_new_model
1383-design-codes.txt, 1373-designCodes.txt,esign_code_descriptions.txt,text_ensemble_weights.pickle,image_ensemble_weights.pickle.

### Copy files  image_resources
image_network_weights.pickle,model1.h5.

### Copy files  text_resources
200seq_300d_1381_BI_GRU_tm_embedding.hdf5,customstopwords.csv,1381-design-codes.txt,latest_tm_markdesc_300.vec.

### Setup RDS Database
Connect to RDS MySQL DB instance with  user credentials and set up the tables for storing the user feedback.


### Setup IAM role and access S3 bucket through EC2 instance


### Update variables
Update environment variables in serverEnv.sh.


## Development server

### Serve
Run `python index.py`. It runs the flask dev server at localhost:5001 (if port is in use, it will exit).

# gsclassification-ml

## Installation

### Install Python 3

### Install Dependencies

cd tmai/gsclassification-ml/ <br>
pip install --upgrade pip <br>
pip install -r requirements.txt

### Create directory
good_services_resources

### Copy files
45-codes.txt, customstopwords.csv, tokenizer_input.txt, tm_goods_services.hdf5, tm_goods_services.vec, gsdescription.txt under good_services_resources folder.<br>

### Setup Solr 
Create a core in solr and index Trademark ID manual data.

### Update variables
Update environment variables in serverEnv.sh

### Setup RDS Database
Connect to RDS MySQL DB instance with  user credentials and set up the tables for storing the user feedback.

## Development server

### Serve
Run `python index.py`. It runs the flask dev server at localhost:5001 (if port is in use, it will exit).

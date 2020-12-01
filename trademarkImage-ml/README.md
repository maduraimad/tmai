# trademarkImage-ml

## Installation

### Install Python 3

### Install Dependencies

cd tmai/trademarkImage-ml/ <br>
pip install --upgrade pip <br>
pip install -r requirements.txt

### Create directories
custom_image_search, custom_image_upload and nn_results_1373dsc_network

### Copy files
Copy model files under custom_image_search folder.<br>
Copy json files under nn_results_1373dsc_network.

### Create S3 bucket
Copy images to S3 bucket.

### Setup IAM role and access S3 bucket through EC2 instance

### Update variables
Update environment variables in serverEnv.sh

### Setup RDS Database
Connect to RDS MySQL DB instance with  user credentials and set up the tables for storing the user feedback and trademark text data.
  

## Development server

### Serve
Run `python index.py`. It runs the flask dev server at localhost:5001 (if port is in use, it will exit).
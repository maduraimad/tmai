#serialNumberExtractor

## Prerequisites

The client project has dependencies that require **Node and NPM**.

### Install Python 3

## Installation

### Getting Started

Install [Node.js](https://nodejs.org)  and [npm](https://www.npmjs.com/)


### Install Dependencies

cd tmai/serialNumberExtractor/ <br>
node install <br>
pip3 install requirements.txt

### Setup IAM role and access S3 bucket through EC2 instance

### Database Setup
Setup database and create a table for storing trademarks application information.<br>
update config.ini with the database credentials 

### Load trademarks text data into database 
node serialNumberSearcher.js <br>
node statusFileDownloader.js <br>
python3 DatabaseImport.py 

### Load trademarks image data into s3
uncomment TD > 0 flag and modify dates. <br>
node serialNumberSearcher.js <br>
python3 TrademarkImageDownloader.py



#export model_file=/home/ubuntu/ml/15-design-codes-2/model1.h5
#export predictions_file=/home/ubuntu/ml/15-design-codes-2/predictions1.txt
#export weights_file=/home/ubuntu/ml/15-design-codes-2/weights1.txt
#export logs_dir=/home/ubuntu/ml/15-design-codes-2/logs/version1/
#export hdf5_path=/home/ubuntu/ml/15-design-codes-2/data-224.h5

#  variables for transfer learning with vgg16 - base model
version_number=1
model_type=topmodel
base_folder=/data/tm_model_data/trained_networks/1373_new_model
label_count=818
top_model_epochs=30
epochs=40
model_save_period=5
design_codes_file=/home/ubuntu/trained_networks/1373_new_model/1373-designCodes.txt
num_of_gpus=4
base_ensemble_folder=/data/tm_model_data/trained_networks/1373_new_model
log_file_name=/data/1373_ml/flask.log
#base_index_folder=/data1/ml/1381-design-codes/rest_service_testing/indexing_resources
#rpt_base_dir=/home/ubuntu/related_patent_files/model_files/
app_prefix=/ml
MYSQL_DATABASE_USER=
MYSQL_DATABASE_PASSWORD=
MYSQL_DATABASE_DB=
MYSQL_DATABASE_DATA_DB=
MYSQL_DATABASE_HOST=
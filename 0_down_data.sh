dataset=$1 #MovieSum, MENSA
type=$2 #train, validation, test


python down_data.py \
    --dataset ${dataset} \
    --data_type ${type}
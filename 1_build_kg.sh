export OPENAI_API_KEY="your_api_key_here"

# parameters
dataset=$1 #MovieSum, MENSA
type=$2 #train, validation, test

python build_kg.py \
    --data_path dataset/${dataset}/0_raw/${type}.jsonl \
    --kg_path dataset/${dataset}/1_kg/${type} \
    --refine gpt \
    --response_exist
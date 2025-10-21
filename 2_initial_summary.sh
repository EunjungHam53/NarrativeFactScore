export OPENAI_API_KEY="your_api_key_here"

# parameters
dataset=$1 #MovieSum, MENSA 
type=$2 #train, validation, test
model=${GEMINI_MODEL:-models/gemini-2.0-flash-lite}

python initial_summary.py \
    --chunk_size 2048 \
    --data_path dataset/${dataset}/0_raw/${type}.jsonl \
    --model ${model} \
    --output_path dataset/${dataset}/2_summary/${model}/${type}
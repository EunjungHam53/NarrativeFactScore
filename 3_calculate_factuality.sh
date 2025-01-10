#export OPENAI_API_KEY="your_api_key_here"

# parameters
dataset=$1 #MovieSum, MENSA 
type=$2 #train, validation, test
iter=$3
model=gpt-4o-mini-2024-07-18

# base paths
data_path=dataset/${dataset}/0_raw/${type}.jsonl
kg_path=dataset/${dataset}/1_kg/${type}
output_path=dataset/${dataset}/3_factscore/iter_${iter}/${model}/${type}

if [ $iter -eq 0 ]; then
    # Initial iteration (iter 0)
    summary_path=dataset/${dataset}/2_summary/${model}/${type}/summary.json
else
    # Subsequent iterations (iter 1+)
    summary_path=dataset/${dataset}/4_corrected_summary/iter_${iter}/${model}/${type}/summary.json
fi

python calculate_factuality.py \
    --data_path ${data_path} \
    --kg_path ${kg_path} \
    --summary_path ${summary_path} \
    --output_path ${output_path} \
    --model ${model}


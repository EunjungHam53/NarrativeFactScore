#export OPENAI_API_KEY="your_api_key_here"

# parameters
dataset=$1 #MovieSum, MENSA 
type=$2 #train, validation, test
iter=$3

if [ $iter -eq 0 ]; then
    summary_path=dataset/${dataset}/2_summary/${model}/${type}/summary.json
else
    summary_path=dataset/${dataset}/4_corrected_summary/iter_${iter}/${model}/${type}/summary.json
fi

python calculate_metrics.py \
    --data_path dataset/${dataset}/0_raw/${type}.jsonl \
    --summary_path ${summary_path} \
    --output_path dataset/${dataset}/5_eval_metrics/iter_${iter}/${model}/${type}
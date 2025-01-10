#export OPENAI_API_KEY="your_api_key_here"

# parameters
dataset=$1 #MovieSum, MENSA 
type=$2 #train, validation, test
iter=$3
model=gpt-4o-mini-2024-07-18

# base paths
data_path=dataset/${dataset}/0_raw/${type}.jsonl
output_path=dataset/${dataset}/4_corrected_summary/iter_${iter}/${model}/${type}

if [ $iter -eq 1 ]; then
    init_summary_path=dataset/${dataset}/2_summary/${model}/${type}
    fact_path=dataset/${dataset}/3_factscore/iter_0/${model}/${type}
else
    prev_iter=$(($iter - 1))
    init_summary_path=dataset/${dataset}/4_corrected_summary/iter_${prev_iter}/${model}/${type}
    fact_path=dataset/${dataset}/3_factscore/iter_${prev_iter}/${model}/${type}
fi

python self_correction.py \
    --data_path $data_path \
    --model $model \
    --init_summary_path $init_summary_path \
    --fact_path $fact_path \
    --output_path $output_path

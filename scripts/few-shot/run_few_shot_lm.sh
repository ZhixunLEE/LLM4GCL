declare -A methods

LM=('BERT' 'RoBERTa' 'LLaMA' 'SimpleCIL')
datasets=('cora' 'citeseer' 'wikics' 'photo' 'products' 'arxiv_23' 'arxiv')


for method in "${LM[@]}"; do
    for dataset in "${datasets[@]}"; do
        # First search hyper-parameters
        python main.py \
            --dataset "$dataset" \
            --model_type "LM" \
            --model "$method" \
            --cl_type 'class' \
            --task_type 'FSNCIL' \
            --ntrail 1 \
            --hyperparam_search \
            --search_type 'grid' \
            --num_samples 10

        output_dir="/YOUR_PATH/results/${type}/${method}/FSNCIL"
        mkdir -p "$output_dir"

        # Then Repeat with best hyper-parameters
        output_file="${output_dir}/${dataset}.txt"
        python main.py \
            --dataset "$dataset" \
            --model_type "LM" \
            --model "$method" \
            --cl_type 'class' \
            --task_type 'FSNCIL' \
            --ntrail 3 > "$output_file"
    done
done

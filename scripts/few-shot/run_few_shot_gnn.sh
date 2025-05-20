declare -A methods

GNN=('BareGNN' 'EWC' 'LwF' 'cosine' 'TPP' 'TEEN')
datasets=('cora' 'citeseer' 'wikics' 'photo' 'products' 'arxiv_23' 'arxiv')


for method in "${GNN[@]}"; do
    for dataset in "${datasets[@]}"; do
        # First search hyper-parameters
        python main.py \
            --dataset "$dataset" \
            --model_type "GNN" \
            --model "$method" \
            --cl_type 'class' \
            --task_type 'FSNCIL' \
            --ntrail 1 \
            --hyperparam_search \
            --search_type 'grid' \
            --num_samples 10

        output_dir="/root/autodl-tmp/results/${type}/${method}/FSNCIL"
        mkdir -p "$output_dir"

        # Then Repeat with best hyper-parameters
        output_file="${output_dir}/${dataset}.txt"
        python main.py \
            --dataset "$dataset" \
            --model_type "GNN" \
            --model "$method" \
            --cl_type 'class' \
            --task_type 'FSNCIL' \
            --ntrail 5 > "$output_file"
    done
done

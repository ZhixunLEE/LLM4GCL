declare -A methods

LM=('OLoRA')

GNN=('BareGNN' 'JointGNN' 'EWC' 'MAS' 'GEM' 'LwF' 'cosine' 'ERGNN' 'SSM' 'CaT' 'DeLoMe' 'TPP')
# LM=('BareLM' 'SimpleCIL' 'OLoRA')
Graph_LM=('LM_emb' 'GraphPrompter' 'ENGINE')
datasets=('cora')
# datasets=('cora' 'citeseer' 'wikics' 'photo' 'products' 'arxiv_23' 'arxiv')


# Settings: 
# cl_type: ClassCIL
# task_type: Normal
# session_size: 2

# for method in "${GNN[@]}"; do
#     for dataset in "${datasets[@]}"; do
#         # First search hyper-parameters
#         python main.py \
#             --dataset "$dataset" \
#             --model_type "GNN" \
#             --model "$method" \
#             --cl_type 'class' \
#             --task_type 'normal' \
#             --session_size 2 \
#             --ntrail 1 \
#             --hyperparam_search \
#             --search_type 'grid' \
#             --num_samples 10

#         output_dir="/root/autodl-tmp/results/${type}/${method}"
#         mkdir -p "$output_dir"

#         # Then Repeat with best hyper-parameters
#         output_file="${output_dir}/${dataset}.txt"
#         python main.py \
#             --dataset "$dataset" \
#             --model_type "GNN" \
#             --model "$method" \
#             --cl_type 'class' \
#             --task_type 'normal' \
#             --session_size 2 \
#             --ntrail 5 > "$output_file"
#     done
# done


for method in "${LM[@]}"; do
    for dataset in "${datasets[@]}"; do
        # First search hyper-parameters
        python main.py \
            --dataset "$dataset" \
            --model_type "LM" \
            --model "$method" \
            --cl_type 'class' \
            --task_type 'normal' \
            --session_size 2 \
            --ntrail 1 \
            --hyperparam_search \
            --search_type 'grid' \
            --num_samples 10

        output_dir="/root/autodl-tmp/results/${type}/${method}"
        mkdir -p "$output_dir"

        # Then Repeat with best hyper-parameters
        output_file="${output_dir}/${dataset}.txt"
        python main.py \
            --dataset "$dataset" \
            --model_type "LM" \
            --model "$method" \
            --cl_type 'class' \
            --task_type 'normal' \
            --session_size 2 \
            --ntrail 3 > "$output_file"
    done
done


# for method in "${Graph_LM[@]}"; do
#     for dataset in "${datasets[@]}"; do
#         # First search hyper-parameters
#         python main.py \
#             --dataset "$dataset" \
#             --model_type "Graph_LM" \
#             --model "$method" \
#             --cl_type 'class' \
#             --task_type 'normal' \
#             --session_size 2 \
#             --ntrail 1 \
#             --hyperparam_search \
#             --search_type 'grid' \
#             --num_samples 10

#         output_dir="/root/autodl-tmp/results/${type}/${method}"
#         mkdir -p "$output_dir"

#         # Then Repeat with best hyper-parameters
#         output_file="${output_dir}/${dataset}.txt"
#         python main.py \
#             --dataset "$dataset" \
#             --model_type "Graph_LM" \
#             --model "$method" \
#             --cl_type 'class' \
#             --task_type 'normal' \
#             --session_size 2 \
#             --ntrail 3 > "$output_file"
#     done
# done
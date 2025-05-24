<h1 align="center"> Can LLMs Alleviate Catastrophic Forgetting in Graph Continual Learning? A Systematic Study </h1>

<h5 align="center">

![](https://img.shields.io/badge/license-MIT-blue)

</h5>

This is the official implementation of the following paper:

> **Can LLMs Alleviate Catastrophic Forgetting in Graph Continual Learning? A Systematic Study** 
>
> Ziyang Cheng*, Zhixun Li*, Yuhan Li, Yixin Song, Kangyi Zhao, Dawei Cheng, Jia Li, and Jeffrey Xu Yu**


## 🚀 Environment Setup
 
 Before you begin, ensure that you have Anaconda or Miniconda installed on your system. This guide assumes that you have a CUDA-enabled GPU. After creating your conda environment (our environment is Python-3.10 with CUDA-11.8), please run:
 
 ```
 pip install -r requirements.txt
 ```
 
 to install Python packages.
 
 
 ## 🎈 Datasets

 | **Dataset**       | **Cora** | **Citeseer** | **WikiCS** | **Photo** | **Products** | **Arxiv-23** | **Arxiv** |
|-------------------|----------|--------------|------------|-----------|--------------|--------------|-----------|
| **# Nodes**       | 2,708    | 3,186        | 11,701     | 48,362    | 53,994*      | 46,196*      | 169,343   |
| **# Edges**       | 5,429    | 4,277        | 215,863    | 500,928   | 72,242*      | 78,542*      | 1,166,243 |
| **# Features**    | 1,433    | 3,703        | 300        | 768       | 100          | 300          | 128       |
| **# Classes**     | 7        | 6            | 10         | 12        | 31*          | 37*          | 40        |
| **Avg. # Token**  | 183.4    | 210.0        | 629.9      | 201.5     | 152.9*       | 237.7*       | 239.8     |
| **Domain**        | Citation | Citation     | Web link   | E-Commerce| E-Commerce   | Citation     | Citation  |

| **NCIL**               | **Cora** | **Citeseer** | **WikiCS** | **Photo** | **Products** | **Arxiv-23** | **Arxiv** |
|------------------------|----------|--------------|------------|-----------|--------------|--------------|-----------|
| **# Classes per session**  | 2        | 2            | 3          | 3         | 4            | 4            | 4         |
| **# Sessions**         | 3        | 3            | 3          | 4         | 8            | 9            | 10        |
| **# Shots per class**      | 100      | 100          | 200        | 400       | 400          | 400          | 800       |

| **FSNCIL**                | **Cora** | **Citeseer** | **WikiCS** | **Photo** | **Products** | **Arxiv-23** | **Arxiv** |
|---------------------------|----------|--------------|------------|-----------|--------------|--------------|-----------|
| **# Base Classes**        | 3        | 2            | 4          | 4         | 11           | 13           | 12        |
| **# Novel Classes**       | 4        | 4            | 6          | 8         | 20           | 24           | 28        |
| **# Ways per session**        | 2        | 2            | 3          | 4         | 4            | 4            | 4         |
| **# Sessions**            | 3        | 3            | 3          | 3         | 6            | 7            | 8         |
| **# Shots per base class**    | 100      | 100          | 200        | 400       | 400          | 400          | 800       |
| **# Shots per novel class**   | 5        | 5            | 5          | 5         | 5            | 5            | 5         |

*Statistics marked with '\*' are adjusted from the original dataset.*
 
**Download Options:**  
1. **Automatic Download**: Files will be fetched from Hugging Face and saved to your specified path.  
2. **Manual Download**: Alternatively, you can download the files manually and place them in the `--data_path` directory.  

## 📖 Code Structure

```
├── LLM4GCL
│   ├── backbones           # GNN and LLM backbones (e.g., GCN, BERT)
│   ├── common              # Shared utilities (e.g., prompts, metrics)
│   ├── data                # Data loader and spliter
│   ├── experiment.py       # Call GCL methods to run experiments 
│   └── models              # Supported GCL methods (GNN/LM/GLM)
├── configs                 # Configurations and hyperparameters of GCL methods
├── main.py                 # The entry of LLM4GCL
└── scripts                 # Bash scripts for automated experiment execution
```


## 🎯 Run
 
#### 👉 Reproduction

* To run experiments for **NCIL** and **FSCIL**, you can execute the provided `.sh` files in the `scripts` dir:

```bash
GLM=('LM_emb' 'ENGINE' 'GraphPrompter' 'GraphGPT' 'LLaGA' 'SimGCL')
datasets=('cora' 'citeseer' 'wikics' 'photo' 'products' 'arxiv_23' 'arxiv')

for method in "${GLM[@]}"; do
    for dataset in "${datasets[@]}"; do
        # First search hyper-parameters
        python main.py --dataset "$dataset" --model_type "GLM" --model "$method" --cl_type 'class' --task_type 'NCIL' --ntrail 1 --hyperparam_search --search_type 'grid' --num_samples 10

        output_dir="/YOUR_PATH/${type}/${method}/NCIL"
        mkdir -p "$output_dir"

        # Then Repeat with best hyper-parameters
        output_file="${output_dir}/${dataset}.txt"
        python main.py --dataset "$dataset" --model_type "GLM" --model "$method" --cl_type 'class' --task_type 'NCIL' --ntrail 3 > "$output_file"
    done
done
```

* For experiments involving **backbone model size** or **session number**, custom configuration is required. See the following guide:

#### 👉 Customization

**1. Backbone Model**

Currently, LLM4GCL supports:

* **GNNs**: Including the GCN, GAT, SAGE, and SGC, which are located in `LLM4GCL/backbones/GNN/*.py`;
* **LMs**: Including the family of BERT, RoBERTa, and LLaMA, which are located in `LLM4GCL/backbones/LM/*.py`; These backbones support auto downloading from Hugging Face. *Notice: LLaMA requires an access token.*

To switch backbones, modify the corresponding files above:

```python
class BERTNet(torch.nn.Module):

    def __init__(self, num_classes, model_path, lora_config, dropout, att_dropout):
        super(BERTNet, self).__init__()
        # self.model_name = 'bert-large-uncased' # 340M
        self.model_name = 'bert-base-uncased' # 110M
        # self.model_name = "prajjwal1/bert-medium" # 41.7M
        # self.model_name = "prajjwal1/bert-small" # 29.1M
        # self.model_name = "prajjwal1/bert-mini" # 11.3M
        # self.model_name = "prajjwal1/bert-tiny" # 4.9M
        ...
        if self.model_name == 'bert-base-uncased':
            self.hidden_dim = 768
        elif self.model_name == 'bert-large-uncased':
            self.hidden_dim = 1024
        elif self.model_name == "prajjwal1/bert-medium" or self.model_name == "prajjwal1/bert-small":
            self.hidden_dim = 512
        elif self.model_name == "prajjwal1/bert-mini":
            self.hidden_dim = 256
        elif self.model_name == "prajjwal1/bert-tiny":
            self.hidden_dim = 128
        ...

```

**2. Session Size**

To modify **NCIL** and **NFSCIL** settings, edit the configuration in `main.py`:

```python
exp_settings = {
    'NCIL': {
        'ways': {'cora': 2, 'citeseer': 2, 'wikics': 3, 'photo': 3, 'products': 4, 'arxiv_23': 4, 'arxiv': 4},
        'sessions': {'cora': 3, 'citeseer': 3, 'wikics': 3, 'photo': 4, 'products': 8, 'arxiv_23': 9, 'arxiv': 10},
        'train_shots': {'cora': 100, 'citeseer': 100, 'wikics': 200, 'photo': 400, 'products': 400, 'arxiv_23': 400, 'arxiv': 800},
        'valid_shots': {'cora': 50, 'citeseer': 50, 'wikics': 50, 'photo': 50, 'products': 50, 'arxiv_23': 50, 'arxiv': 50},
        'test_shots': {'cora': 100, 'citeseer': 100, 'wikics': 200, 'photo': 400, 'products': 400, 'arxiv_23': 400, 'arxiv': 800},
    },
    'FSNCIL': {
        'base_session': {'cora': 3, 'citeseer': 2, 'wikics': 4, 'photo': 4, 'products': 11, 'arxiv_23': 13, 'arxiv': 12},
        'novel_session': {'cora': 4, 'citeseer': 4, 'wikics': 6, 'photo': 8, 'products': 20, 'arxiv_23': 24, 'arxiv': 28},
        'ways': {'cora': 2, 'citeseer': 2, 'wikics': 3, 'photo': 4, 'products': 4, 'arxiv_23': 4, 'arxiv': 4},
        'sessions': {'cora': 3, 'citeseer': 3, 'wikics': 3, 'photo': 3, 'products': 6, 'arxiv_23': 7, 'arxiv': 8},
        'base_train_shots': {'cora': 100, 'citeseer': 100, 'wikics': 200, 'photo': 400, 'products': 400, 'arxiv_23': 400, 'arxiv': 800},
        'train_shots': {'cora': 5, 'citeseer': 5, 'wikics': 5, 'photo': 5, 'products': 5, 'arxiv_23': 5, 'arxiv': 5},
        'valid_shots': {'cora': 50, 'citeseer': 50, 'wikics': 50, 'photo': 50, 'products': 50, 'arxiv_23': 50, 'arxiv': 50},
        'test_shots': {'cora': 100, 'citeseer': 100, 'wikics': 200, 'photo': 400, 'products': 400, 'arxiv_23': 400, 'arxiv': 800},
    }
}

```

**3. Hyper-parameters**

All model hyperparameters are defined in the `config/` dir:

```yaml
default:
  gnn: 'GCN'
  seed: [0, 1, 2, 3, 4]
  epochs: 300
  valid_epoch: 10
  lr: 1e-4
  weight_decay: 5e-4
  layer_num: 2
  hidden_dim: 128
  dropout: 0.5
  patience: 20
  batch_size: 256
  num_heads: 4 # only available for GATConv
  aggr: 'mean' # ['mean', 'max', 'lstm'], only available for SAGEConv
```

You can also change the search space:

```yaml
search_space:
  gnn: ['GCN', 'SAGE']
  lr: [1e-5, 1e-4, 1e-3]
  hidden_dim: [64, 128, 256]
```

*Notice: The optimal hyperparameters will be automatically merged with default settings.*

## 🙏 Acknowledgement

We gratefully acknowledge all cited papers for their valuable contributions! The original implementations of baseline methods and other relevant repositories we adopted are listed below:


* **[NeurIPS 2022]** **CGLB: Benchmark Tasks for Continual Graph Learning**. 
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/548a41b9cac6f50dccf7e63e9e1b1b9b-Paper-Datasets_and_Benchmarks.pdf)][[Code](https://github.com/QueuQ/CGLB)] ![](https://img.shields.io/badge/Benchmark-A52A2A)

* **[NeurIPS 2023]** **Few-Shot Class-Incremental Learning via Training-Free Prototype Calibration**. 
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/30dfe47a3ccbee68cffa0c19ccb1bc00-Paper-Conference.pdf)][[Code](https://github.com/wangkiw/TEEN)] ![](https://img.shields.io/badge/Research-A52A2A)

* **[NeurIPS 2023]** **A Comprehensive Study on Text-attributed Graphs: Benchmarking and Rethinking**. 
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/37d00f567a18b478065f1a91b95622a0-Paper-Datasets_and_Benchmarks.pdf)][[Code](https://github.com/sktsherlock/TAG-Benchmark)] ![](https://img.shields.io/badge/Dataset%20\&%20Benchmark-A52A2A)

* **[NeurIPS 2024]** **Replay-and-Forget-Free Graph Class-Incremental Learning: A Task Profiling and Prompting Approach**.
[[Paper](https://arxiv.org/pdf/2410.10341)][[Code](https://github.com/mala-lab/TPP)] ![](https://img.shields.io/badge/Research-A52A2A)

* **[SIGKDD Explorations Newsletter 2024]** **Exploring the Potential of Large Language Models (LLMs)in Learning on Graphs**.
[[Paper](https://arxiv.org/pdf/2307.03393)][[Code](https://github.com/CurryTang/Graph-LLM)] ![](https://img.shields.io/badge/Explorations%20Newsletter-A52A2A)

* **[ICLR 2024]** **One for All: Towards Training One Graph Model for All Classification Tasks**. 
[[Paper](https://arxiv.org/pdf/2310.00149)][[Code](https://github.com/LechengKong/OneForAll)] ![](https://img.shields.io/badge/Research-A52A2A)

* **[ICLR 2024]** **Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning**.
[[Paper](https://arxiv.org/pdf/2305.19523)][[Code](https://github.com/XiaoxinHe/TAPE)] ![](https://img.shields.io/badge/Research-A52A2A)

* **[IJCV 2024]** **Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need**.
[[Paper](https://arxiv.org/pdf/2305.19523)][[Code](https://github.com/LAMDA-CL/RevisitingCIL)] ![](https://img.shields.io/badge/Research-A52A2A)

* **[IJCAI 2024]** **Efficient Tuning and Inference for Large Language Models on Textual Graphs**.
[[Paper](https://arxiv.org/pdf/2401.15569)][[Code](https://github.com/ZhuYun97/ENGINE)] ![](https://img.shields.io/badge/Research-A52A2A)

* **[WWW 2024]** **Can we Soft Prompt LLMs for Graph Learning Tasks?**.
[[Paper](https://arxiv.org/pdf/2402.10359)][[Code](https://github.com/franciscoliu/graphprompter)] ![](https://img.shields.io/badge/Research%20(Short%20Paper)-A52A2A)

* **[SIGIR 2024]** **GraphGPT: Graph Instruction Tuning for Large Language Models**.
[[Paper](https://arxiv.org/pdf/2310.13023)][[Code](https://github.com/HKUDS/GraphGPT)] ![](https://img.shields.io/badge/Research-A52A2A)

* **[ICML 2024]** **LLaGA: Large Language and Graph Assistant**.
[[Paper](https://arxiv.org/pdf/2402.08170)][[Code](https://github.com/VITA-Group/LLaGA)] ![](https://img.shields.io/badge/Research-A52A2A)

* **[NeurIPS 2024]** **GLBench: A Comprehensive Benchmark for Graphs with Large Language Models**.
[[Paper](https://arxiv.org/pdf/2407.07457)][[Code](https://github.com/NineAbyss/GLBench)] ![](https://img.shields.io/badge/Benchmark-A52A2A)

* **[NeurIPS 2024]** **TEG-DB: A Comprehensive Dataset and Benchmark of Textual-Edge Graphs**.
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/7054d2c49863c1c41be1d53f4377b82a-Paper-Datasets_and_Benchmarks_Track.pdf)][[Code](https://github.com/Zhuofeng-Li/TEG-Benchmark)] ![](https://img.shields.io/badge/Dataset%20&%20Benchmark-A52A2A)

* **[ICML 2025]** **A Comprehensive Analysis on LLM-based Node Classification Algorithms**.
[[Paper](https://arxiv.org/pdf/2502.00829)][[Code](https://github.com/WxxShirley/LLMNodeBed)] ![](https://img.shields.io/badge/Benchmark-A52A2A)

* **[Arxiv 2025]** **Exploring Graph Tasks with Pure LLMs: A Comprehensive Benchmark and Investigation**.
[[Paper](https://arxiv.org/pdf/2502.18771)][[Code](https://github.com/myflashbarry/LLM-benchmarking)] ![](https://img.shields.io/badge/Benchmark-A52A2A)


## 📌 Contact

For discussions and inquiries, you can reach **Ziyang Cheng** at: **[zycheng2003@gmail.com](mailto:zycheng2003@gmail.com)**.  
  
Moreover, feel free to open issues or suggest improvements—we appreciate your contributions 🤩!  

## MomentumSMoE: Integrating Momentum into Sparse Mixture of Experts
MomentumSMoE: Integrating Momentum into Sparse Mixture of Experts 

https://arxiv.org/abs/2410.14574

### Prerequisites

- pytorch
- fastmoe: https://github.com/laekov/fastmoe
- The toolkit supports [Weights & Biases](https://docs.wandb.ai/) for monitoring jobs. If you use it, also install `wandb`.

### Usage


#### Prepare WikiText-103 Datasets: 

- Download the WikiText-103 dataset from [here](https://github.com/laekov/fastmoe/blob/master/examples/transformer-xl/scripts/getdata.sh), then change bash scripts based on your local data paths.
```bash
data_directory/
    └── wikitext-103
        ├── test.txt
        ├── train.txt
        └── valid.txt
```

#### Pretraining <u>SMoE</u> (SwitchTransformers) on WikiText-103: 

``` # WikiText-103 dataset: 
bash scripts/smoe-s.sh
bash scripts/smoe-m.sh
bash scripts/smoe-l.sh
```

#### Pretraining <u>*Momentum*SMoE</u> on WikiText-103: 

``` # WikiText-103 dataset: 
bash scripts/smoe-mom-s.sh
bash scripts/smoe-mom-m.sh
bash scripts/smoe-mom-l.sh
```

#### Pretraining <u>*Adam*SMoE</u> on WikiText-103: 

``` # WikiText-103 dataset: 
bash scripts/smoe-adam-m.sh
```

#### Pretraining <u>GLaM</u> on WikiText-103: 

``` # WikiText-103 dataset: 
bash scripts/glam-m.sh
```

#### Pretraining <u>*Momentum*GLaM</u> on WikiText-103: 

``` # WikiText-103 dataset: 
bash scripts/glam-mom-m.sh
```

#### Pretraining <u>*Adam*GLaM</u> on WikiText-103: 

``` # WikiText-103 dataset: 
bash scripts/glam-adam-m.sh
```

#### Wandb support:
- Add these flags to bash script with your project and job name
``` # Wandb: 
--wandb 
--project-name test 
--job-name test 
```

### Checkpoints
Please find our SMoE checkpoints at <https://drive.google.com/drive/folders/1C0-39hmGyXO1ZbKvM_HH0hx2hewgU7ig?usp=sharing>. 



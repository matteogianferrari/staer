# STAER: Temporal Aligned Rehearsal for Continual Spiking Neural Network

## Citing the paper

```bibtex
@article{gianferrari2026staer,
  title={STAER: Temporal Aligned Rehearsal for Continual Spiking Neural Network},
  author={Gianferrari, Matteo and Moussadek, Omayma and Salami, Riccardo and Fiorini, Cosimo and Tartarini, Lorenzo and Gandolfi, Daniela and Calderara, Simone},
  journal={arXiv preprint arXiv:2601.20870},
  year={2026}
}
```

## Setup

+ Use `./main.py` to run experiments.
+ The general mandatory arguments are `--model`, `--dataset` and `--backbone`. To specify these refer to the name use in the decorator function of the respective `.py` file (e.g., `@register_backbone("sresnet19-cifar10")`).
+ New datasets can be added to the `datasets/` folder.
+ New models can be added to the `models/` folder.
+ New backbones can be added to the `backbone/` folder.
+ Runs can be logged with wandb by setting --wandb=True, specifying a --wandb_entity and a --wandb_project.

## Datasets

### Visual

+ Sequential MNIST
+ Sequential CIFAR-10

## Models

+ snn-ER
+ snn-DER
+ snn-DER++
+ STAER

## In order to run STAER and replicate the results, use the following commands as guidelines:

+ MNIST
```bash
python main.py --dataset=seq-smnist --model=staer --backbone=sresnet19-mnist --T=2 --buffer_size=200 --alpha1=0.5 --alpha2=0.5 --beta=1e-4 --temp_sep=0 --lr=3e-3 --optimizer=adamw --lr_scheduler=cosine --seed=41 -O=2
```
```bash
python main.py --dataset=seq-smnist --model=staer --backbone=sresnet19-mnist --T=2 --buffer_size=500 --alpha1=0.5 --alpha2=0.5 --beta=1e-4 --temp_sep=0 --lr=3e-3 --optimizer=adamw --lr_scheduler=cosine --seed=41 -O=2
```
```bash
python main.py --dataset=seq-smnist --model=staer --backbone=sresnet19-mnist --T=2 --buffer_size=5120 --alpha1=0.5 --alpha2=0.5 --beta=1e-4 --temp_sep=0 --lr=3e-3 --optimizer=adamw --lr_scheduler=cosine --seed=41 -O=2
```
```bash
python main.py --dataset=seq-smnist --model=staer --backbone=sresnet19-mnist --T=4 --buffer_size=200 --alpha1=0.5 --alpha2=0.5 --beta=1e-4 --temp_sep=0 --lr=3e-3 --optimizer=adamw --lr_scheduler=cosine --seed=41 -O=2
```
```bash
python main.py --dataset=seq-smnist --model=staer --backbone=sresnet19-mnist --T=4 --buffer_size=500 --alpha1=0.5 --alpha2=0.5 --beta=1e-4 --temp_sep=0 --lr=3e-3 --optimizer=adamw --lr_scheduler=cosine --seed=41 -O=2
```
```bash
python main.py --dataset=seq-smnist --model=staer --backbone=sresnet19-mnist --T=4 --buffer_size=5120 --alpha1=0.5 --alpha2=0.5 --beta=1e-4 --temp_sep=0 --lr=3e-3 --optimizer=adamw --lr_scheduler=cosine --seed=41 -O=2
```

+ CIFAR-10
```bash
python main.py --dataset=seq-scifar10 --model=staer --backbone=sresnet19-cifar10 --T=2 --buffer_size=200 --alpha1=0.5 --alpha2=0.5 --temp_sep=0 --beta=1e-4 --lr=3e-3 --optimizer=adamw --lr_scheduler=cosine --seed=41 -O=2
```
```bash
python main.py --dataset=seq-scifar10 --model=staer --backbone=sresnet19-cifar10 --T=2 --buffer_size=500 --alpha1=0.5 --alpha2=0.5 --temp_sep=0 --beta=1e-4 --lr=3e-3 --optimizer=adamw --lr_scheduler=cosine --seed=41 -O=2
```
```bash
python main.py --dataset=seq-scifar10 --model=staer --backbone=sresnet19-cifar10 --T=2 --buffer_size=5120 --alpha1=0.5 --alpha2=0.5 --temp_sep=0 --beta=1e-4 --lr=3e-3 --optimizer=adamw --lr_scheduler=cosine --seed=41 -O=2
```
```bash
python main.py --dataset=seq-scifar10 --model=staer --backbone=sresnet19-cifar10 --T=4 --buffer_size=200 --alpha1=0.5 --alpha2=0.5 --temp_sep=0 --beta=1e-4 --lr=3e-3 --optimizer=adamw --lr_scheduler=cosine --seed=41 -O=2
```
```bash
python main.py --dataset=seq-scifar10 --model=staer --backbone=sresnet19-cifar10 --T=4 --buffer_size=500 --alpha1=0.5 --alpha2=0.5 --temp_sep=0 --beta=1e-4 --lr=3e-3 --optimizer=adamw --lr_scheduler=cosine --seed=41 -O=2
```
```bash
python main.py --dataset=seq-scifar10 --model=staer --backbone=sresnet19-cifar10 --T=4 --buffer_size=5120 --alpha1=0.5 --alpha2=0.5 --temp_sep=0 --beta=1e-4 --lr=3e-3 --optimizer=adamw --lr_scheduler=cosine --seed=41 -O=2
```
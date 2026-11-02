# WLIA_RNR
Efficient Label Distribution Inference and Defense via Weight Norms in Federated Learning


## Implementation
### Ubuntu
```sh
pyenv install 3.10
pyenv local 3.10

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
# if cuda version 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```


### Dataset setting
```sh
cd WLIA/dataset
python generate_cifar10.py
```

## Label Distribution Attack and Defence
### RUN WLIA
```sh
cd WLIA/src
python -u main.py -did 0 -data Cifar10_dir_N_10_alpha_0.5_aux_10_seed_0 -m cnn -algo FedAvg -dia -lda WLIA -go WLIA_TEST 
```

### RUN WLIA + RNR
```sh
cd WLIA/src
python -u main.py -did 0 -data Cifar10_dir_N_10_alpha_0.5_aux_10_seed_0 -m cnn -algo FedAvg -dia -lda WLIA -rw RNR -go WLIA_RNR_TEST > WLIA_RNR_TEST.out 2>&1 &
```


## 📌 Framework Foundation
This repository contains the implementation code for our paper, built upon **[PFLlib: Personalized Federated Learning Library and Benchmark](https://github.com/TsingZ0/PFLlib)**.

PFLlib is a comprehensive Python library featuring 39+ federated learning algorithms, 24 datasets, and specialized support for addressing data heterogeneity in personalized federated learning scenarios.

## 🙏 Acknowledgments
We gratefully acknowledge the PFLlib team for providing an excellent foundation for personalized federated learning research.
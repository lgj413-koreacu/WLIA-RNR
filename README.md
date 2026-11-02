# Efficient Label Distribution Inference Attack and Defense on Classifier Weights in Federated Learning
Official repository for the paper "Efficient Label Distribution Inference Attack and Defense on Classifier Weights in Federated Learning", accepted to NeurIPS 2026.

Label distribution leakage in federated learning (FL) poses a critical privacy threat, exposing population-level patterns beyond individual data breaches. Building on known correlations between class sample frequencies and classifier weight magnitudes, we demonstrate how adversaries can exploit these relationships to infer label distributions from shared model parameters in FL. To characterize this threat, we propose WLIA (Weight Norm-based Label Distribution Inference Attack), which infers client label distributions by analyzing variations in classifier weight norms across communication rounds. WLIA operates on the classifier layer without requiring auxiliary data, synthetic samples, or meta-classifiers, enabling broader applicability across heterogeneous model architectures. To counter this threat, we introduce RNR (Randomized Weight Norm Regularization), a defense mechanism that disrupts frequency-weight correlations by randomly regularizing weight norms during local training. RNR adds only a penalty term to the cross-entropy loss, making it efficient and easy to integrate. Comprehensive experiments across six datasets demonstrate that WLIA achieves superior inference accuracy compared to existing attacks, while RNR effectively mitigates leakage with minimal impact on utility.

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

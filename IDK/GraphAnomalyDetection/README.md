
# Subgraph Centralization: A Necessary Step for Graph Anomaly Detection

## GCAD

This project implements the Subgraph Centralization: A Necessary Step for Graph Anomaly Detection for graph anomaly detection


## Model Usage

### Dependencies 

```markdown
numpy
scipy
scikit-learn
```

### Get start

Run the code with the following command.

```markdown
python run.py --psi 2 --dataset cora --h 1 --lamda 0.0625
python run.py --psi 2 --dataset citeseer --h 1 --lamda 0.125
python run.py --psi 2 --dataset pubmed --h 1 --lamda 0.125
python run.py --psi 2 --dataset ACM --h 1 --lamda 0.0625

python run.py --psi 2 --dataset lattice_l --h 1 --lamda 0.0
python run.py --psi 8 --dataset lattice_s --h 1 --lamda 0.0
python run.py --psi 4 --dataset RGG_s --h 2 --lamda 0.0
python run.py --psi 8 --dataset RGG_l --h 1 --lamda 0.0
python run.py --psi 2 --dataset SBM_str --h 1 --lamda 0.25
python run.py --psi 2 --dataset watts_strogatz --h 1 --lamda 0.0
```

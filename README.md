## Conformal Prediction Interval Counterfactuals

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License)

This repository provides a demonstration of Conformal Prediction Interval Counterfactuals (CPICFs) -- a way of generating counterfactual examples for tabular data chosen to be informative to the recpient by choosing those with a large conformal prediction interval. It accompanies the submission of the [2025 COPA](https://copa-conference.com/) conference paper "Individualised Counterfactual Examples Using Conformal Prediction Intervals". 


#### Conformal Counterfactual Generation Demo

[![Open In Github](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](notebooks/CPICF_demo.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/alan-turing-institute/CPICF/blob/COPA_demo/notebooks/CPICF_demo.ipynb)


#### Installation Instructions

For running the jupyter notebook locally:

1. Create a virtual environment for your operating system 
`python3.10 -m venv env`
2. Install requirements for notebook
```
pip install -r requirements.txt
```
and set up ipykernel
```
python3 -m pip install ipykernel
python3 -m ipykernel install --user --name CPICF
```

3. Open the jupyter notebook and select the CPICF kernel





<!-- 
#### Citation

#### Acknowledgements

#### License
 -->

## Wakening Past Concepts without Past Data: <br>Class-Incremental Learning from Online Placebos

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/online-placebos/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg?style=flat-square&logo=python&color=3776AB&logoColor=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.8-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

This repository contains the PyTorch implementation for the [WACV 2024](https://wacv2024.thecvf.com/) Paper ["Wakening Past Concepts without Past Data: Class-Incremental Learning from Online Placebos"](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Wakening_Past_Concepts_Without_Past_Data_Class-Incremental_Learning_From_Online_WACV_2024_paper.pdf).

### Initialize the environment

```bash
conda create --name onlineplacebos python=3.8
conda activate onlineplacebos
conda install cudatoolkit=11.1 -c nvidia
conda install pytorch torchvision torchaudio -c pytorch-lts
```

```bash
pip install tqdm 
pip install scipy 
pip install quadprog
pip install POT
```
### Run the code
```bash
bash run_exp_all.py
```

### Citation

Please cite our paper if it is helpful to your work:

```bibtex
@inproceedings{Liu2024OnlinePlacebos,
  author       = {Yaoyao Liu and
                  Yingying Li and
                  Bernt Schiele and
                  Qianru Sun},
  title        = {Wakening Past Concepts without Past Data: Class-Incremental Learning
                  from Online Placebos},
  booktitle    = {{IEEE/CVF} Winter Conference on Applications of Computer Vision, {WACV}
                  2024, Waikoloa, HI, USA, January 3-8, 2024},
  pages        = {2215--2224},
  publisher    = {{IEEE}},
  year         = {2024},
  url          = {https://doi.org/10.1109/WACV57701.2024.00222}
}
```

### Acknowledgements

Our implementation uses the source code from the following repositories:
- <https://github.com/G-U-N/PyCIL>

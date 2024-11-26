## Wakening Past Concepts without Past Data: <br>Class-Incremental Learning from Online Placebos

### Initialize the environment

```bash
conda create --name pycil-1 python=3.8
conda activate pycil-1
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

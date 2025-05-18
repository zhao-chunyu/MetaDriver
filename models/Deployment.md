# One/Few-shot Driver Saliency Prediction Configuration

## 📚Table of Contents

* [⚙️Environment for MetaDriver](#⚙️environment-for-metadriver)
  + [🛠️Envir: MetaDriver](#our)
* [⚙️Environment for One/Few-shot](#fewshot)
  + [🛠️Envir: MetaDriver_ori](#fewshot1)
  + [🛠️Envir: MetaDriver_mmcv](#fewshot2)
* [🔧Utilities](#🔧utilities)
  + [🗂️One-click to modify dataset path](#🗂️one-click-to-modify-dataset-path)
  + [📊One-click to collect metrics data](#📊one-click-to-collect-metrics-data)
* [⭐️Cite](#⭐️cite)

<a name="MetaDriver"></a>

## 		⚙️Environment for MetaDriver

<a name="our"></a>

### 	🛠️Envir: MetaDriver

* Git clone this repository

```python
git clone https://github.com/zhao-chunyu/MetaDriver.git
cd MetaDriver
```

* Create conda environment

```python
conda create -n MetaDriver python=3.11
conda activate MetaDriver
```

* Install PyTorch 2.5.1+cu121

```python
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

* Install requirement

```python
pip install -r utils/requirements.txt
```

<a name="fewshot"></a>

## ⚙️Environment for One/Few-shot

*Different models may take different environments, we collate and find that 2 main ones are used, one is `metadriver_ori` and the other is `metadriver_mmcv`* .

<a name="fewshot1"></a>

### 	🛠️Envir: MetaDriver_ori

This environment is used for the training of `PFENet`, `BAM` networks.

* Git clone this repository

```python
git clone https://github.com/zhao-chunyu/MetaDriver.git
cd MetaDriver
```

* Create conda environment

```python
conda create -n MetaDriver_ori python=3.11
conda activate MetaDriver_ori
```

* Install PyTorch 2.5.1+cu121

```python
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

* Install requirement

```python
pip install -r utils/metadriver_ori.txt
```

<a name="fewshot2"></a>

### 	🛠️Envir: MetaDriver_mmcv

This environment is used for the training of `HDMNet`, `AMNet`, `AENet` networks.

* Git clone this repository

```python
git clone https://github.com/zhao-chunyu/MetaDriver.git
cd MetaDriver
```

* Create conda environment

```python
conda create -n metadriver_mmcv python=3.11
conda activate metadriver_mmcv
```

* Install PyTorch 2.5.1+cu121

```python
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

* Install requirement

```python
pip install -r utils/metadriver_mmcv.txt
```

## 🔧Utilities

### 	🗂️One-click to modify dataset path

You need to change the dataset path of the yaml file, but this can be laborious. We provide a script for you to change it with one click.

```python
# new_dada_data_root = '/data/dataset/MetaDADA'      # Replace it with your MetaDADA path.
# new_psad_data_root = '/data/dataset/MetaPSAD'      # Replace it with your MetaPSAD path.
python utils/modify_yaml.py
```

> You need to change two params in your `utils/modify_yaml.py` to the path to your dataset. (params: `new_dada_data_root`, `new_psad_data_root`)

### 📊One-click to collect metrics data

Even though we've stored the metrics in `.xlsx` file when calculating them, it's still inconvenient to have too many files. We provide a script for one-click collection of these metrics.

```python
cd utils
python collect_metrics.py
```

> You need to change a params in your `utils/collect_metrics.py` to change the dataset name. (param: `dataset`)

## ⭐️Cite


If you find this repository useful, please use the following BibTeX entry for citation  and give us a star⭐.

```python
waiting accepted
```


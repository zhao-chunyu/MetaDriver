<div align="center">
<a name="start-anchor"></a>
</div>
<div align="center">
  <img src="assert\titlelogo.jpg" alt="logo" width="600" height="auto" />
</div>

<div align="center">
Anonymous Author(s)
</div>
<div align="center">
 Affiliation
 </div>
<div align="center">
 Address
 </div>
<div align="center">
 email
 </div>

<div align="center">
  <img src="assert\dataset_cases.gif" alt="logo" width="800" height="auto" />
</div>


## 📚Table of Contents

* [🔥Update](#update)
* [⚡Proposed Model](#model)
* [📖Datasets](#Datasets)
* [🛠️Deployment](#Deployment)
  + [`[1] Environment`](#envir-anchor)
  + [`[2] Run train`](#run-train)
  + [`[3] Run test`](#run-test)
  + [`[4] Run visualization`](#run-visualization)
  + [`[5] Sota One/Few-shot`](#run-fewshot)
* [🚀Demo](#Demo)
* [⭐️Cite](#Cite)

<a name="update"></a>

## 🔥Update


- **2025/05/16**: We provide two handy scripts.
    - `modify_yaml.py`: One-click to modify dataset path. [script](models/Deployment.md#Utilities1)
    - `collect_metrics.py`: One-click to collect metrics data. [script](models/Deployment.md#Utilities2)

- **2025/05/02**: All the code and models are completed.
    - How to configure:  [command](#envir-anchor) & [script](models/Deployment.md#MetaDriver1)
    - How to train:  [command](#run-train)
    - How to evaluate:  [command](#run-test )
    - How to visualize:  [command](#run-visualization )

- **2025/03/05**: We collect existing meta-learning methods for driver saliency prediction.
     - Environment configuration
     	- *metadriver_ori*: `PFENet (TPAMI'22)`, `BAM (CVPR'22)`. [Details](models/Deployment.md#fewshot1)
     	- *metadriver_mmcv*: `HDMNet (CVPR'23)`, `AMNet (NIPS'23)`, `AENet (ECCV'24)`. [Details](models/Deployment.md#fewshot2)
     - How to use: [command](#run-fewshot) & [script](models/Deployment.md#fewshot)
- **2025/02/01**: We collect driving accident data with data categories, divided into 4 folds by category.
    - DADA: 52 categories. It repartitioned into `DADA-52i`.
    - PSAD: 4 categories. It repartitioned into `PSAD-4i`.
- **2025/01/11**: We propose a model in order to ``learn to learn`` driver attention in driving accident scenarios.

<a name="model"></a>

## ⚡Proposed Model [🔁](#start-anchor)

we propose a framework that combines a one-shot learning strategy with scene-level semantic alignment, enabling the model to achieve human-like attentional perception in previously unseen scenarios. (named **MetaDriver**)

<img src="assert\model.png" style="zoom: 100%;">

1. **A one-shot learning framework for drivers' attention prediction**. 
Inspired by how humans learn from limited examples, we filter support masks to extract salient regions, which are then used to guide a driver attention learner. Semantic alignment between support and query samples further refines model optimization.
2. **Two novel modules: Saliency Map Filter and Salient Semantic Alignment**. 
Since driver attention masks are often dispersed and semantic overlap between query-support pairs exists only in salient regions, we filter out irrelevant mask information and align salient semantic features to boost prediction accuracy.
3. **Extensive evaluation on DADA-52i and PSAD-4i with SOTA results**. 
Our method consistently outperforms 10 competitive baselines across both datasets and backbones (Resnet-50, Vgg-16), surpassing existing driver saliency prediction and one/few-shot models.

<a name="Datasets"></a>

## 📖Datasets [🔁](#start-anchor)

<div align="center">
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Accident</th>
      <th>Fold-0</th>
      <th>Fold-1</th>
      <th>Fold-2</th>
      <th>Fold-3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DADA-52i</td>
      <td>52</td>
      <td>[1] a pedestrian crosses the road. ...</td>
      <td>[14] there is an object crash. ...</td>
      <td>[27] a motorbike is out of control. ...</td>
      <td>[40] vehicle changes lane with the same direction to ego-car. ...</td>
    </tr>
    <tr>
      <td>PSAD-4i</td>
      <td>4</td>
      <td>moving ahead or waiting.</td>
      <td>lateral.</td>
      <td>oncoming.</td>
      <td>turning.</td>
    </tr>
  </tbody>
</table> 
</div>


【note】 For all datasets we will provide our download link with the official link. Please choose according to your needs.

> **DADA-52i**: This dataset we will upload in BaiduYun (please wait). Official web in [link](https://github.com/JWFangit/LOTVS-DADA "Official DADA").
>
> **PSAD-4i**: This dataset we will upload in BaiduYun (please wait).  Official web in [link](https://github.com/Shun-Gan/PSAD-dataset "Official PSAD").

<table align="center" border="0" cellspacing="30" cellpadding="30">
  <thead>
    <tr>
      <th>DADA-52i</th>
      <th>PSAD-4i</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <pre>
MetaDADA/
├── 1/
│   ├── 001/ 
│   │   ├── images/   
│   │   └── maps/  
│   ├── 002/
│   └── ...
├── 2/
│   ├── 001/ 
│   │   ├── images/   
│   │   └── maps/  
│   ├── 002/
│   └── ...
└── ...</pre>
      </td>
      <td>
        <pre>
MetaPSAD/
├── images/
│   ├── 2/ 
│   │   ├── 0004/   
│   │   └── ...  
│   ├── 3/
│   └── ...
└── maps/
    ├── 2/ 
    │   ├── 0004/   
    │   └── .../  
    ├── 3/
    └── ...</pre>
      </td>
    </tr>
  </tbody>
</table>

<a name="Deployment"></a>

## 🛠️Deployment [🔁](#start-anchor)

<a name="envir-anchor"></a>

### 		[1] Environment 

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

<a name="run-train"></a>

### 		[2] Run train

> Before training, scripts you might use: `modify_yaml.py`: One-click to modify dataset path. [Details](models/Deployment.md#Utilities1)

*If you wish to train with our model, please use the command below.* 

```python
sh scripts/train_MetaDriver.sh [datasaet] [split#] [backbone] [gpu]
```

> `dataset`: metadada, metapsad.
>
> `#`: 0, 1, 2, 3.
>
> `backbone`: resnet50, vgg.
>
> `gpu`: 0 or other.

​	*Example:*
```python
sh scripts/train_MetaDriver.sh metadada split0 resnet50 0
```
<a name="run-test"></a>

### 		[3] Run test

*We calculate the predicted values and then use python for the prediction.* 

```python
sh scripts/test_MetaDriver.sh [datasaet] [split#] [backbone] [gpu]
```

​	*Example:*
```python
sh scripts/test_MetaDriver.sh metadada split0 resnet50 0
```

> After testing, scripts you might use: `collect_metrics.py`: One-click to collect metrics data. [Details](models/Deployment.md#Utilities2)

<a name="run-visualization"></a>

### 		[4] Run visualization

*If you want to visualize all the data of a certain dataset directly, you can use the following command.*

```python
sh scripts/visual_MetaDriver.sh [datasaet] [split#] [backbone] [gpu]
```

​	*Example:*
```python
sh scripts/visual_MetaDriver.sh metadada split0 resnet50 0
```

<a name="run-fewshot"></a>

### 		[5] Sota One/Few-shot

*If you want to run other comparison models, we have supported 5 SOTA models.*  [Details](models/Deployment.md#fewshot)

<details close>
<summary>Supported model</summary>
<table class="tg"><thead>
  <tr>
    <td class="tg-0pky">Model</td>
    <td class="tg-0pky"><a href="https://github.com/dvlab-research/PFENet">PFENet (TPMAI'22)</a></td>
    <td class="tg-0pky"><a href="https://github.com/chunbolang/BAM">BAM (CVPR'22)</td>
    <td class="tg-0pky"><a href="https://github.com/Pbihao/HDMNet">HDMNet (CVPR'23)</td>
    <td class="tg-0pky"><a href="https://github.com/Wyxdm/AMNet">AMNet (NIPS'23)</td>
    <td class="tg-0pky"><a href="https://github.com/Sam1224/AENet">AENet (ECCV'24)</td>
  </tr></thead>
</table>
</details>

*Then, you can run the following command.*

```python
sh scripts/[train_*.sh] [dataset] [split#] [backbone] [gpu]
```
> `*`: PFENet, BAM, HDMNet, AMNet, AENet.
>
> `dataset`: metadada, metapsad.
>
> `#`: 0, 1, 2, 3.
>
> `backbone`: resnet50, vgg.
>
> `gpu`: 0 or other.

​	*Example:*
```python
sh scripts/train_PFENet.sh metadada split0 resnet50 0
```
> Model result `testing` and model result `visualization` are similar to our `MetaDriver` model.

<a name="Demo"></a>

## 🚀Demo [🔁](#start-anchor)

<img src="assert\compare.png" style="zoom: 100%;">

<a name="Cite"></a>

## ⭐️Cite [🔁](#start-anchor)


If you find this repository useful, please use the following BibTeX entry for citation  and give us a star⭐.

```python
waiting accepted
```


<div align="center">
<a name="start-anchor"></a>
</div>
<div align="center">
  <img src="assert\titlelogo.jpg" alt="logo" width="600" height="auto" />
</div>

<hr />

## 🔥Update

- **2025/05/15**: All the code and models are completed.
    - How to configure:  [command](#envir-anchor) & [script](deployment.md)
    - How to train:  [command](#run-train)
    - How to evaluate:  [command](#run-test )
    - How to visualize:  [command](#run-visualization )

- **2025/03/25**: We collect existing meta-learning methods for driver saliency prediction.
     - Environment configuration
     	- *metadriver*: `PFENet (TPAMI'22)`, `BAM (CVPR'22)`. [Details](deployment.md)
     	- *metadriver_mmcv*: `HDMNet (CVPR'23)`, `AMNet (NIPS'23)`, `AENet (ECCV'24)`. [Details](deployment.md)
     	- *metadriver_mamba*: `HMNet (NIPS'24)`. [Details](deployment.md)
     - How to use: [command](#run-fewshot)
- **2025/02/05**: We collect driving accident data with data categories, divided into 4 folds by category.
    - DADA: 52 categories. It repartitioned into `DADA-52i`.
    - PSAD: 4 categories. It repartitioned into `PSAD-4i`.
- **2025/02/01**: We propose a model in order to ``learn to learn`` driver attention in driving accident scenarios.

## 💬Highlight [🔁](#start-anchor)

<div align="center">
<img src="assert\highlight.jpg" width="auto" height="auto" />
</div>



## ⚡Proposed Model [🔁](#start-anchor)

we propose a saliency mamba model, named **SalM²** that uses "Top-down" driving scene semantic information to guide "Bottom-up" driving scene image information to simulate human drivers' attention allocation. 

<img src="assert\model.jpg" style="zoom: 100%;">

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
      <td>a pedestrian crosses the road. ...</td>
      <td>there is an object crash. ...</td>
      <td>a motorbike is out of control. ...</td>
      <td>vehicle changes lane with the same direction to ego-car. ...</td>
    </tr>
    <tr>
      <td>PSAD-4i</td>
      <td>4</td>
      <td>a pedestrian crosses the road.</td>
      <td>there is an object crash.</td>
      <td>a motorbike is out of control.</td>
      <td>vehicle changes lane with the same direction to ego-car.</td>
    </tr>
  </tbody>
</table>
</div>

【note】 For all datasets we will provide our download link with the official link. Please choose according to your needs.

> **DADA-52i**: This dataset we will upload in BaiduYun (please wait). Official web in [link](https://github.com/JWFangit/LOTVS-DADA "Official DADA").
>
> **PSAD-4i**: This dataset we will upload in BaiduYun (please wait).  Official web in [link](https://github.com/Shun-Gan/PSAD-dataset "Official PSAD").


## 🛠️ Deployment [🔁](#start-anchor)

<a name="envir-anchor"></a>

### 		[1] Environment 

*If you have downloaded our `repository code` and installed `PyTorch` and `CUDA`.*  [More details](deployment.md#(1)-Environment)

```python
pip install -r requirements.txt
```
<a name="run-train"></a>

### 		[2] Run train

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

### 		[5] Sota Few-shot

*If you want to run other comparison models, we have supported 6 SOTA models.*

<details close>
<summary>Supported model</summary>
<table class="tg"><thead>
  <tr>
    <td class="tg-0pky"><a href="https://github.com/dvlab-research/PFENet">PFENet (TPMAI'22)</a></td>
    <td class="tg-0pky"><a href="https://github.com/chunbolang/BAM">BAM (CVPR'22)</td>
    <td class="tg-0pky"><a href="https://github.com/Pbihao/HDMNet">HDMNet (CVPR'22)</td>
    <td class="tg-0pky"><a href="https://github.com/Wyxdm/AMNet">AMNet (NIPS'23)</td>
    <td class="tg-0pky"><a href="https://github.com/Sam1224/HMNet">HMNet (NIPS'24)</td>
    <td class="tg-0pky"><a href="https://github.com/Sam1224/AENet">AENet (ECCV'24)</td>
  </tr></thead>
</table>
</details>
*Then, you can run the following command.*

```python
sh scripts/[train_*.sh] [dataset] [split#] [backbone] [gpu]
```
> `*`: PFENet, BAM, HDMNet, AMNet, HMNet, AENet.
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


## 🚀 Live Demo [🔁](#start-anchor)

<div align="center">
  <img src="assert/demo-example1.gif" alt="BDDA-1" width="230" height="auto" />
  <img src="assert/demo-example2.gif" alt="BDDA-2" width="230" height="auto" />
  <img src="assert/demo-example3.gif" alt="BDDA-3" width="230" height="auto" />
</div>


## ⭐️ Cite [🔁](#start-anchor)


If you find this repository useful, please use the following BibTeX entry for citation  and give us a star⭐.

```python
waiting accepted
```


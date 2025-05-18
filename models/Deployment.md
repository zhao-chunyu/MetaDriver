# One/Few-shot Driver Saliency Prediction Configuration



## 		Environment 

### [1] MetaDriver

```python
sh scripts/train_MetaDriver.sh metadada split0 resnet50 0
```



<a name="run-train"></a>

### 		[2] Sota One/Few-shot

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

*If you want to run other comparison models, we have supported 5 SOTA models.*

<details close>
<summary>Supported model</summary>
<table class="tg"><thead>
  <tr>
    <td class="tg-0pky">Model</td>
    <td class="tg-0pky"><a href="https://github.com/dvlab-research/PFENet">PFENet (TPMAI'22)</a></td>
    <td class="tg-0pky"><a href="https://github.com/chunbolang/BAM">BAM (CVPR'22)</td>
    <td class="tg-0pky"><a href="https://github.com/Pbihao/HDMNet">HDMNet (CVPR'22)</td>
    <td class="tg-0pky"><a href="https://github.com/Wyxdm/AMNet">AMNet (NIPS'23)</td>
    <td class="tg-0pky"><a href="https://github.com/Sam1224/HMNet">HMNet (NIPS'24)</td>
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


## 🚀 Demo [🔁](#start-anchor)

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


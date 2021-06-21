# eBay_product_retrieval

---
## What can I find here?

This repository is for eBay product retrieval.




---
## Some Notes:

If you use this code in your research, please cite
```

```

---

**[All implemented methods and metrics are listed at the bottom!](#-implemented-methods)**

---

---

## How to use this Repo

### Requirements:

* PyTorch 1.2.0+ 
* Python 3.6+
* torchvision 0.3.0+
* numpy, PIL, oprncv-python
* timm



### Training:


**[I.]** **Advanced Runs**:


```
python main.py --loss triplet --bs 512  --net efficientnet-b4 --cls-num 20000 --lamda 0.9 --lr 1e-4  --loss triplet --batch_mining distance

```





### DML criteria

* **smoothap** [[Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval](https://arxiv.org/abs/2007.12163)] `--loss smoothap`
* **Circle loss** [] `--loss circle`
* **Contrastive** [] `--loss contrastive`
* **Triplet** [] `--loss triplet`
...

### Architectures

* **ResNet50&101** [[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)] e.g. `--net resnet50&101`.
* **Efficientnet**  e.g. `--net efficientnet-b4`.
* **Vision Transformer**  e.g. `--net vit_base_patch16_224`.

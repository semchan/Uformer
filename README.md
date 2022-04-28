# Video summarization with u-shaped transformer


A PyTorch implementation of our paper [Video summarization with u-shaped transformer](https://link.springer.com/article/10.1007/s10489-022-03451-1). Published in [Applied Intelligence](https://www.springer.com/journal/10489).

## Getting Started

This project is developed on Ubuntu 16.04 with CUDA 9.0.176.


```sh
git clone https://github.com/semchan/Uformer.git
```


Install python dependencies.

```sh
pip install -r requirements.txt
```

## Datasets and pretraining models Preparation

Download the pre-processed datasets into `datasets/` folder, including [TVSum](https://github.com/yalesong/tvsum), [SumMe](https://gyglim.github.io/me/vsum/index.html), [OVP](https://sites.google.com/site/vsummsite/download), and [YouTube](https://sites.google.com/site/vsummsite/download) datasets.



+ (Baidu Cloud) Link: https://pan.baidu.com/s/1LUK2aZzLvgNwbK07BUAQRQ Extraction Code: x09b


Now the datasets structure should look like

```
UFormer
└── datasets/
    ├── eccv16_dataset_ovp_google_pool5.h5
    ├── eccv16_dataset_summe_google_pool5.h5
    ├── eccv16_dataset_tvsum_google_pool5.h5
    ├── eccv16_dataset_youtube_google_pool5.h5
    └── readme.txt
```

## Evaluation

To evaluate your anchor-based models, run

```sh
sh evaluate.sh
```



## Training


To train anchor-based attention model on TVSum and SumMe datasets with canonical settings, run

```sh
python train.py --model anchor-based --model-dir ./models/ab_basic --splits ./splits/tvsum.yml ./splits/summe.yml

```






## Acknowledgments

We gratefully thank the below open-source repo, which greatly boost our research.
+ Thank Part of the code is referenced from:  [DSNet](https://github.com/li-plus/DSNet). Thanks for their great work before.
+ Thank [KTS](https://github.com/pathak22/videoseg/tree/master/lib/kts) for the effective shot generation algorithm.
+ Thank [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) for the pre-processed public datasets.
+ Thank [VASNet](https://github.com/ok1zjf/VASNet) for the training and evaluation pipeline.

## Citation

If you find our codes or paper helpful, please consider citing.

```
@article{chen2022video,
  title={Video summarization with u-shaped transformer},
  author={Chen, Yaosen and Guo, Bing and Shen, Yan and Zhou, Renshuang and Lu, Weichen and Wang, Wei and Wen, Xuming and Suo, Xinhua},
  journal={Applied Intelligence},
  pages={1--17},
  year={2022},
  publisher={Springer}
}

```

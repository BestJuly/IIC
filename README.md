# Self-supervised Video Representation Learning Using Inter-intra Contrastive Framework
Official code for paper, Self-supervised Video Representation Learning Using Inter-intra Contrastive Framework [ACMMM'20]. 

[Arxiv paper](https://arxiv.org/abs/2008.02531) [Project page](https://bestjuly.github.io/IIC/)

## Requirements
> This is my experimental enviroment. 

- PyTorch 1.3.0
> It seems that PyTorch 1.7.0 is not compatible with current codes, causing poor performance. [#9](https://github.com/BestJuly/IIC/issues/9)
- python  3.7.4
- accimage 

## Inter-intra contrastive (IIC) framework
For samples, we have
- [ ] Inter-positives: samples with **same labels**, not used for self-supervised learning;
- [x] Inter-negatives: **different samples**, or samples with different indexes;
- [x] Intra-positives: data from the **same sample**, in different views / from different augmentations; 
- [x] Intra-negatives: data from the **same sample** while some kind of information has been broken down. In video case, temporal information has been destoried.

Our work makes use of all usable parts (in this classification category) to form an inter-intra contrastive framework. The experiments here are mainly based on Contrastive Multiview Coding. 

It is flexible to extend this framework to other contrastive learning methods which use negative samples, such as MoCo and SimCLR.

![image](https://github.com/BestJuly/Inter-intra-video-contrastive-learning/blob/master/fig/general.png)

## Highlights
### Make the most of data for contrastive learning.
Except for inter-negative samples, all possible data are used to help train the network. This **inter-intra learning framework** can make the most use of data in contrastive learning.

### Flexibility of the framework
The **inter-intra learning framework** can be extended to
- Different contrastive learning methods: CMC, MoCo, SimCLR ...
- Different intra-negative generation methods: frame repeating, frame shuffling ...
- Different backbones: C3D, R3D, R(2+1)D, I3D ...

## Updates
Oct. 1, 2020 - Results using C3D and R(2+1)D are added; fix random seed more tightly.
Aug. 26, 2020 - Add pretrained weights for R3D.

## Usage of this repo
> Notification: we have added codes to fix random seed more tightly for better reproducibility. However, results in our paper used previous random seed settings. Therefore, there should be tiny differences for the performance from that reported in our paper. To reproduce retrieval results same as our paper, please use the provided model weights.

### Data preparation
You can download UCF101/HMDB51 dataset from official website: [UCF101](http://crcv.ucf.edu/data/UCF101.php) and [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/). Then decoded videos to frames.    
I highly recommend the pre-computed optical flow images and resized RGB frames in this [repo](https://github.com/feichtenhofer/twostreamfusion).

If you use pre-computed frames, the folder architecture is like `path/to/dataset/video_id/frames.jpg`. If you decode frames on your own, the folder architecture may be like `path/to/dataset/class_name/video_id/frames.jpg`, in which way, you need pay more attention to the corresponding paths in dataset preparation.

For pre-computed frames, find `rgb_folder`, `u_folder` and `v_folder` in `datasets/ucf101.py` for UCF101 datasets and change them to meet your environment. Please note that all those modalities are prepared even though in some settings, optical flow data are not used train the model. 

If you do not prepare optical flow data, simply set `u_folder=rgb_folder` and `v_folder=rgb_folder` should help to avoid errors.

### Train self-supervised learning part
```
python train_ssl.py --dataset=ucf101
```

This equals to

```
python train_ssl.py --dataset=ucf101 --model=r3d --modality=res --neg=repeat
```

This default setting uses frame repeating as intra-negative samples for videos. R3D is used.

We use two views in our experiments. View #1 is a RGB video clip, View #2 can be RGB/Res/Optical flow video clip. Residual video clips are default modality for View #2. You can use `--modality` to try other modalities. Intra-negative samples are generated from View #1. 

It may be wired to use only one optical flow channel *u* or *v*. We use only one channel to make it possible for **only one model** to handle inputs from different modalities. It is also an optional setting that using different models to handle each modality.

### Retrieve video clips
```
python retrieve_clips.py --ckpt=/path/to/your/model --dataset=ucf101 --merge=True
```
One model is used to handle different views/modalities. You can set `--modality` to decide which modality to use. When setting `--merge=True`, RGB for View #1 and the specific modality for View #2 will be jointly used for joint retrieval.

By default training setting, it is very easy to get over 30%@top1 for video retrieval in ucf101 and around 13%@top1 in hmdb51 without joint retrieval.

### Fine-tune model for video recognition
```
python ft_classify.py --ckpt=/path/to/your/model --dataset=ucf101
```
Testing will be automatically conducted at the end of training.

Or you can use
```
python ft_classify.py --ckpt=/path/to/your/model --dataset=ucf101 --mode=test
```
In this way, only testing is conducted using the given model.

**Note**: The accuracies using residual clips are not stable for validation set (this may also caused by limited validation samples), the final testing part will use the best model on validation set.

If everything is fine, you can achieve around 70% accuracy on UCF101. The results will vary from each other with different random seeds. 

## Results
### Retrieval results
The table lists retrieval results on UCF101 *split* 1. We reimplemented CMC and report its results. Other results are from corresponding paper. VCOP, VCP, CMC, PRP, and ours are based on R3D network backbone.

Method | top1 | top5 | top10 | top20 | top50
---|---|---|---|---|---
Jigsaw  | 19.7 | 28.5 | 33.5 | 40.0 | 49.4
OPN  | 19.9 | 28.7 | 34.0 | 40.6 | 51.6
R3D (random)  | 9.9 | 18.9 | 26.0 | 35.5 | 51.9
VCOP  | 14.1  |  30.3 | 40.4 | 51.1 | 66.5
VCP | 18.6 | 33.6 | 42.5 | 53.5 | 68.1
CMC  |  26.4  |  37.7  |  45.1  |  53.2  |  66.3 
Ours (repeat + res)  |  36.5  |  54.1  |  62.9  |  72.4  |  83.4 
Ours (repeat + u)  |  41.8  |  60.4  |  **69.5**  |  **78.4**  |  **87.7** 
Ours (shuffle + res)  |  34.6  |  53.0  |  62.3  |  71.7  |  82.4 
Ours (shuffle + v)  |  **42.4**  |  **60.9**  |  69.2  |  77.1  |  86.5 
PRP | 22.8 | 38.5 | 46.7 | 55.2 | 69.1
RTT | 26.1 | 48.5	| 59.1 | 69.6 | 82.8
MemDPC-RGB | 20.2 |	40.4 | 52.4 | 64.7 | -
MemDPC-Flow | 40.2 |	63.2 | 71.9 | 78.6 | -


### Recognition results
We only use R3D as our network backbone. In this table, all reported results are pre-trained on UCF101. 

Usually, 1. using Resnet-18-3D, R(2+1)D or deeper networks; 2.pre-training on larger datasets; 3. using larger input resolutions; 4. combining with audios or other features will also help. 

Method | UCF101 | HMDB51
---|---|---
Jigsaw |  51.5  |  22.5 
O3N (res)  |  60.3  |  32.5 
OPN  |  56.3  |  22.1
OPN (res)  |  71.8  |  36.7
CrossLearn  |  58.7  |  27.2 
CMC (3 views)  |  59.1  |  26.7
R3D (random)  | 54.5 | 23.4
ImageNet-inflated  |  60.3  |  30.7
3D ST-puzzle  |  65.8  |  33.7
VCOP (R3D)  | 64.9 |  29.5
VCOP (R(2+1)D) | 72.4 | 30.9 
VCP (R3D)  |  66.0 |  31.5 
Ours (repeat + res, R3D) |  72.8  |  35.3 
Ours (repeat + u, R3D)  |  72.7  |  36.8 
Ours (shuffle + res, R3D) |  **74.4**  |  **38.3**
Ours (shuffle + v, R3D)  |  67.0  |  34.0 
PRP (R3D) | 66.5 | 29.7
PRP (R(2+1)D) | 72.1 | 35.0

**Residual clips + 3D CNN** The residual clips with 3D CNNs are effective, especially for scratch training. More information about this part can be found in [Rethinking Motion Representation: Residual Frames with 3D ConvNets for Better Action Recognition](https://arxiv.org/abs/2001.05661) (previous but more detailed version) and [Motion Representation Using Residual Frames with 3D CNN](https://arxiv.org/abs/2006.13017) (short version with better results).

The key code for this part is 
```
shift_x = torch.roll(x,1,2)
x = ((shift_x -x) + 1)/2
```
which is slightly different from that in papers.

We also reimplement VCP in this [repo](https://github.com/BestJuly/VCP). By simply using residual clips, significant improvements can be obtained for both video retrieval and video recognition.


## Pretrained weights
Pertrained weights from self-supervised training step: R3D[(google drive)](https://drive.google.com/file/d/17c5KJuPFEHt0vCjrMPO3UfS7BN8nNESX/view?usp=sharing). 

> With this model, for video retrieval, you should achieve
> - 33.4% @top1 with `--modality=res --merge=False`
> - 34.8% @top1 with `--modality=rgb --merge=False`
> - 36.5% @top1 with`--modality=res --merge=True`

Finetuned weights for action recognition: R3D[(google drive)](https://drive.google.com/file/d/12uzHArg5hMGLuEUz36H4fJgGaeN4QyhZ/view?usp=sharing).

> With this model, for video recognition, you should achieve
> 72.7% @top1 with `python ft_classify.py --model=r3d --modality=res --mode=test -ckpt=./path/to/model --dataset=ucf101 --split=1`.
> This result is better than that reported in paper. Results may be further improved with strong data augmentations.

For any questions, please contact Li TAO (taoli@hal.t.u-tokyo.ac.jp).

### Results for other network architectures

Results are averaged on 3 splits without using optical flow. R3D and R21D are the same as VCOP / VCP / PRP.   

UCF101 | top1 | top5 | top10 | top20 | top50 | Recong 
---    |---   |---   |---    |---    |---    |---
C3D (VCOP) | 12.5 |	29.0 | 39.0 | 50.6 |	66.9 | 65.6 
C3D (VCP) | 17.3 | 31.5 | 42.0 |	52.6 | 	67.7 | 68.5 
C3D (PRP) | 23.2 | 38.1 | 46.0 | 55.7 | 68.4 | 69.1
C3D (ours, repeat) | **31.9** | **48.2** |	**57.3**|	**67.1** |	**79.1** | **70.0** 
C3D (ours, shuffle) | 28.9	| 45.4	| 55.5	| 66.2	| 78.8 | 69.7 
R21D (VCOP) | 10.7 |	25.9 | 35.4	| 47.3 | 63.9 | 72.4 
R21D (VCP) | 19.9 | 33.7 | 42.0	| 50.5 | 64.4 | 66.3 
R21D (PRP) | 20.3 | 34.0 | 41.9 | 51.7 | 64.2 | 72.1
R21D (ours, repeat) | **34.7** | **51.7** | **60.9** | **69.4** | **81.9** | 72.4
R21D (ours, shuffle) | 30.2	| 45.6	| 55.0	| 64.4 |	77.6 | **73.3**
Res18-3D (ours, repeat) | 36.8	| 54.1 |	63.1 |	72.0 |	83.3 | 72.4
Res18-3D (ours, shuffle) | 33.0 |	49.2 |	59.1 |	69.1 |	80.6 | 73.1


HMDB51 |  top1 | top5 | top10 | top20 | top50 | Recong
---    |---   |---   |---    |---    |---    |---
C3D (VCOP) | 7.4	| 22.6	| 34.4	| 48.5	| 70.1 | 28.4
C3D (VCP) | 7.8	| 23.8	| 35.3	| 49.3	| 71.6 | 32.5
C3D (PRP) | 10.5 | 27.2 | 40.4 | 56.2 | 75.9 | **34.5**
C3D (ours, repeat) | 9.9 |	29.6 |	42.0 | 	57.3 |	78.4 | 30.8
C3D (ours, shuffle) | **11.5**	| **31.3**	| **43.9**	| **60.1**	| **80.3** | 29.7
R21D (VCOP) | 5.7	| 19.5	| 30.7	| 45.6	| 67.0 | 30.9
R21D (VCP) | 6.7	| 21.3	| 32.7	| 49.2	| 73.3 | 32.2
R21D (PRP) | 8.2 | 25.3 | 36.2 | 51.0 | 73.0 | **35.0**
R21D (ours, repeat)| **12.7**	| **33.3**	| **45.8**	| **61.6**	| **81.3** | 34.0
R21D (ours, shuffle)| 12.6	| 31.9 |	44.2 | 	59.9 |	80.7 | 31.2
Res18-3D (ours, repeat) | 15.5 |	34.4 |	48.9 |	63.8 |	83.8 | 34.3
Res18-3D (ours, shuffle) | 12.4 |	33.6 |	46.9 |	63.2 |	83.5 | 34.3

## Citation
If you find our work helpful for your research, please consider citing the paper
```
@article{tao2020selfsupervised,
    title={Self-supervised Video Representation Learning Using Inter-intra Contrastive Framework},
    author={Li Tao and Xueting Wang and Toshihiko Yamasaki},
    journal={arXiv preprint arXiv:2008.02531},
    year={2020},
    eprint={2008.02531},
}
```

If you find the residual input helpful for video-related tasks, please consider citing the paper
```
@article{tao2020rethinking,
  title={Rethinking Motion Representation: Residual Frames with 3D ConvNets for Better Action Recognition},
  author={Tao, Li and Wang, Xueting and Yamasaki, Toshihiko},
  journal={arXiv preprint arXiv:2001.05661},
  year={2020}
}

@article{tao2020motion,
  title={Motion Representation Using Residual Frames with 3D CNN},
  author={Tao, Li and Wang, Xueting and Yamasaki, Toshihiko},
  journal={arXiv preprint arXiv:2006.13017},
  year={2020}
}
```


## Acknowledgements
Part of this code is inspired by [CMC](https://github.com/HobbitLong/CMC) and [VCOP](https://github.com/xudejing/video-clip-order-prediction).


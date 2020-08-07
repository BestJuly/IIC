Official code for [Self-supervised Video Representation Learning Using Inter-intra Contrastive Framework](arxiv.org/abs/2008.02531) [ACMMM'20]

## Requirements
> This is my experimental enviroment.   
PyTorch 1.3.0
python  3.7.4

## Inter-intra contrastive framework
For samples, we have
- [ ] Inter-positives: samples with same labels, not used for self-supervised learning;
- [x] Inter-negatives: different samples, or samples with different indexes;
- [x] Intra-positives: data from the same sample, in different views / from different augmentations; 
- [x] Intra-negatives: data from the same sample while some kind of information has been broken down. In video case, temporal information has been destoried.
Our work make use of all usable parts to form an inter-intra contrastive framework. The experiments here are mainly based on Contrastive Multiview Coding. It is flexible to extend this framework to other contrastive learning methods such as MoCo and SimCLR.

## Highlights
### Make the most of data for contrastive learning.
Except for inter-negative samples, all possible data are used to help train the network. This **inter-intra learning framework** can make the most use of data in contrastive learning.

### Flexibility of the framework
The **inter-intra learning framework** can be extended to
- Different contrastive learning methods: CMC, MoCo, SimCLR ...
- Different intra-negative generation methods: frame repeating, frame shuffling ...
- Different backbone: C3D, R3D, R(2+1)D, I3D ...


## Usage of this repo
### Data preparation
You can download UCF101/HMDB51 dataset from official website: [UCF101](http://crcv.ucf.edu/data/UCF101.php) and [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/). Then decoded videos to frames.    
I highly recommend the pre-comupeted optical flow images and resized RGB frames in this [repo](https://github.com/feichtenhofer/twostreamfusion).

### Train self-supervised learning part
```
python train_ssl.py --dataset=ucf101
```
The default setting uses frame repeating as intra-negative samples for videos. R3D is used by default. You can use `--model` to try different models. 

We use two views in our experiments. View #1 is a RGB video clip, View #2 can be RGB/Res/Optical flow video vlip. Residual video clips are default modality for View # 2. You can use `--modality` to try other modalities. Intra-negative samples are generated from View #1. 

### Retrieve video clips
```
python retrieve_clips.py --ckpt=/path/to/your/model --dataset=ucf101 --merge=True
```
Only one model is used for different views. You can set `--modality` to decide which modality to use. When setting `--merge=True`, RGB for View #1 and the specific modality for View #2 will be jointly tested.

### Fine-tune model for video recognition
```
python ft_classify.py --ckpt=/path/to/your/model --dataset=ucf101
```

## Citation
If you find our work helpful for your research, please consider citing the paper
```
@article{tao2020rethinking,
  title={Rethinking Motion Representation: Residual Frames with 3D ConvNets for Better Action Recognition},
  author={Tao, Li and Wang, Xueting and Yamasaki, Toshihiko},
  journal={arXiv preprint arXiv:2001.05661},
  year={2020}
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
Part of this code is inspired by [CMC](https://github.com/HobbitLong/CMC) and [VCOP](https://github.com/xudejing/video-clip-order-prediction)

# Video-TransUNet
This is the official repo for the paper **[<Video-TransUNet: Temporally Blended Vision Transformer for CT VFSS Instance Segmentation>](https://arxiv.org/abs/2208.08315)**.<br/>
Chengxi Zeng, Xinyu Yang, Majid Mirmehdi, Alberto M Gambaruto and Tilo Burghardt
SPIE Internation Conference on Machine Vision

Please also see our latest update using Swintransformer .<br/>
**[<Video-SwinUNet: Spatio-temporal Deep Learning Framework for VFSS Instance Segmentation>](https://arxiv.org/abs/2302.11325)**.<br/>
IEEE International Conference on Image Processing<br/>
[Github](https://github.com/SimonZeng7108/Video-SwinUNet)<br/>




## Abstract
We propose Video-TransUNet, a deep architecture for instance segmentation in medical CT videos constructed by integrating temporal feature blending into the TransUNet deep learning framework. In particular, our approach amalgamates strong frame representation via a ResNet CNN backbone, multi-frame feature blending via a Temporal Context Module (TCM), non-local attention via a Vision Transformer, and reconstructive capabilities for multiple targets via a UNet-based convolutional-deconvolutional architecture with multiple heads. We show that this new network design can significantly outperform other state-of-the-art systems when tested on the segmentation of bolus and pharynx/larynx in Videofluoroscopic Swallowing Study (VFSS) CT sequences. On our  VFSS2022 dataset it achieves a dice coefficient of $0.8796\%$ and an average surface distance of $1.0379$ pixels. Note that tracking the pharyngeal bolus accurately is a particularly important application in clinical practice since it constitutes the primary method for diagnostics of swallowing impairment. Our findings suggest that the proposed model can indeed enhance the TransUNet architecture via exploiting temporal information and improving segmentation performance by a significant margin. We publish key source code, network weights, and ground truth annotations for simplified performance reproduction.

## Architecture Overview
<img src="https://github.com/SimonZeng7108/Video-TransUNet/blob/main/FIGS/model.png" width="800" height="318"><br/>
(a) Multi-frame ResNet-50-based feature extractor; (b) Temporal Context Module for temporal feature blending across frames; (c) Vision Transformer Block for non-local attention-based learning of multi-frame encoded input; (d) Cascaded expansive decoder with skip connections as used in original UNet architectures, however, here with multiple prediction heads co-learning the two  instances of clinical interest.<br/>

## Grad-Cam results
<img src="https://github.com/SimonZeng7108/Video-TransUNet/blob/main/FIGS/grad_cam.png" width="600" height="426"><br/>
Based on four sample frames (top) we show for TransUNet and our model boundary segmentations (lower rows) and GradCam output (upper rows) highlighting where models are paying attention to. Results for the bolus and pharynx are next to each other left and right, respectively, for every sample image. Note the much more target-focused results of our model.<br/>

## Repo usage
### Requirements 
- `torch == 1.10.1`
- `torchvision`
- `torchsummary`
- `numpy == 1.21.5`
- `scipy`
- `skimage`
- `matplotlib`
- `PIL`
- `mmcv == 1.5.0`
- `Medpy`

### 1. Download ViT pre-trained models
* R50-ViT-B_16
* ViT-B_16
* ViT-L_16
...
[Get models in this link](https://console.cloud.google.com/storage/browser/vit_models/imagenet21k?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)


### 2. Data 
Our data ethics approval only grants usage and showing on paper, not yet support full release. 
To fully utlise the Temporal Blending feature of the model, sequential image data should be converted to numpy arrays and concated in the format of `[T, H, W]` for BW data and `[T, C, H, W]` for colored data.

### 3. Train/Test
A small batch size is recommanded as the size of the data and nature of TCM components.<br/>
Train:<br/>
`python train.py --dataset Synapse --vit_name R50-ViT-B_16`<br/>
Test:<br/>
`python test.py --dataset Synapse --vit_name R50-ViT-B_16`<br/>

## Ref Repo
[Vision Transformer](https://github.com/google-research/vision_transformer)<br/>
[TransUNet](https://github.com/Beckschen/TransUNet/blob/main/README.md)<br/>
[TCM](https://github.com/youshyee/Greatape-Detection)<br/>

## Citation
```
@misc{https://doi.org/10.48550/arxiv.2208.08315,
  doi = {10.48550/ARXIV.2208.08315},
  
  url = {https://arxiv.org/abs/2208.08315},
  
  author = {Zeng, Chengxi and Yang, Xinyu and Mirmehdi, Majid and Gambaruto, Alberto M and Burghardt, Tilo},
  
  keywords = {Image and Video Processing (eess.IV), Computer Vision and Pattern Recognition (cs.CV), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Video-TransUNet: Temporally Blended Vision Transformer for CT VFSS Instance Segmentation},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```


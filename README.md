# Video-TransUNet
This is the official repo for the paper **<Video-TransUNet: Temporally Blended Vision Transformer for CT VFSS Instance Segmentation>**

## Abstract
We propose Video-TransUNet, a deep architecture for instance segmentation in medical CT videos constructed by integrating temporal feature blending into the TransUNet deep learning framework. In particular, our approach amalgamates strong frame representation via a ResNet CNN backbone, multi-frame feature blending via a Temporal Context Module (TCM), non-local attention via a Vision Transformer, and reconstructive capabilities for multiple targets via a UNet-based convolutional-deconvolutional architecture with multiple heads. We show that this new network design can significantly outperform other state-of-the-art systems when tested on the segmentation of bolus and pharynx/larynx in Videofluoroscopic Swallowing Study (VFSS) CT sequences. On our  VFSS2022 dataset it achieves a dice coefficient of $0.8796\%$ and an average surface distance of $1.0379$ pixels. Note that tracking the pharyngeal bolus accurately is a particularly important application in clinical practice since it constitutes the primary method for diagnostics of swallowing impairment. Our findings suggest that the proposed model can indeed enhance the TransUNet architecture via exploiting temporal information and improving segmentation performance by a significant margin. We publish key source code, network weights, and ground truth annotations for simplified performance reproduction.

## Architecture Overview
<img src="https://github.com/SimonZeng7108/Video-TransUNet/blob/main/FIGS/model.png" width="1000" height="398"><br/>
(a) Multi-frame ResNet-50-based feature extractor; (b) Temporal Context Module for temporal feature blending across frames; (c) Vision Transformer Block for non-local attention-based learning of multi-frame encoded input; (d) Cascaded expansive decoder with skip connections as used in original UNet architectures, however, here with multiple prediction heads co-learning the two  instances of clinical interest.<br/>

## Grad-Cam results
<img src="https://github.com/SimonZeng7108/Video-TransUNet/blob/main/FIGS/grad_cam.png" width="800" height="568"><br/>
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

### 1.Download ViT pre-trained models
[Get models in this link](https://console.cloud.google.com/storage/browser/vit_models/imagenet21k?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)
* R50-ViT-B_16
* ViT-B_16
* ViT-L_16
...







import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from skimage.segmentation import find_boundaries
import cv2
from torchvision.transforms.functional import to_tensor

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - dice

        
        #compute metric
        out = torch.zeros_like(score)
        gt = torch.zeros_like(target)
        out[score > 0.5] = 1.0
        gt[target > 0.5] = 1.0

        inter = (out * gt).sum(dim=(1,2))
        uni= out.sum(dim=(1,2)) + gt.sum(dim=(1,2)) 
        dice_metric  = 2.0 * (inter) / (uni + smooth)
        
        num_zero_metric = len(dice_metric) - torch.count_nonzero(dice_metric)


        return loss, dice_metric.sum(), num_zero_metric

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        # target = self._one_hot_encoder(target)
        # if weight is None:
        #     weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        # class_wise_dice = []
        # loss = 0.0
        # for i in range(0, self.n_classes):
        #     print(target[:, i].shape)
        #     dice = self._dice_loss(inputs[:, i], target[:, i])
        #     class_wise_dice.append(1.0 - dice.item())
        #     loss += dice * weight[i]
        inputs = torch.sigmoid(inputs)
        loss, dice_metric, num_zero_metric = self._dice_loss(inputs, target)

        return loss, dice_metric, num_zero_metric 


def calculate_metric_percase(pred, gt):
    pred[pred > 0.5] = 1
    gt[gt > 0.5] = 1

    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt) #hd95
        sensitivity = metric.binary.sensitivity(pred, gt)
        specificity = metric.binary.specificity(pred, gt)
        precision = metric.binary.precision(pred, gt)

        return [dice, hd95, asd, sensitivity, specificity, precision]
    else:
        return [0, 0, 0, 0, 0, 0]


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

        
    
    input = torch.from_numpy(image).unsqueeze(
        0).float().cuda()
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    net.eval()
    with torch.no_grad():
        # out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        # prediction = out.cpu().detach().numpy()

        
        out_3layers = net(input)

        out_layer1 = torch.sigmoid(out_3layers[:, 0, :, :]).cpu().detach().numpy().squeeze(0)
        out_layer2 = torch.sigmoid(out_3layers[:, 1, :, :]).cpu().detach().numpy().squeeze(0)

        
        bolus_prediction = np.uint8(out_layer1>=0.5)
        pharynx_prediction = np.uint8(out_layer2>=0.5)


        bolus_label = label[0, :, :]
        pharynx_label = label[1, :, :]
            


    bolus_metric = calculate_metric_percase(bolus_prediction, bolus_label)
    pharynx_metrc = calculate_metric_percase(pharynx_prediction, pharynx_label)

    if test_save_path is not None:

        image = image[int((image.shape[0]-1)/2), :, :]
        bolus_prediction = bolus_prediction.astype(np.float32)
        pharynx_prediction = pharynx_prediction.astype(np.float32)
        label = label.astype(np.float32)
        bolus_label = label[0, :, :]
        pharynx_label = label[1, :, :]


        #Define a show mask on image function

        bolus_label_to_show = find_boundaries(bolus_label, mode = 'thick')
        bolus_label_to_show = np.uint8(bolus_label_to_show)
        bolus_label_to_show = cv2.normalize(bolus_label_to_show, None, 0, 255, cv2.NORM_MINMAX)

        pharynx_label_to_show = find_boundaries(pharynx_label, mode = 'thick')
        pharynx_label_to_show = np.uint8(pharynx_label_to_show)
        pharynx_label_to_show = cv2.normalize(pharynx_label_to_show, None, 0, 255, cv2.NORM_MINMAX)


        bolus_mask_pred = find_boundaries(bolus_prediction, mode ='thick')
        bolus_mask_pred = np.uint8(bolus_mask_pred)
        bolus_mask_pred = cv2.normalize(bolus_mask_pred, None, 0, 255, cv2.NORM_MINMAX)

        pharynx_mask_pred = find_boundaries(pharynx_prediction, mode ='thick')
        pharynx_mask_pred = np.uint8(pharynx_mask_pred)
        pharynx_mask_pred = cv2.normalize(pharynx_mask_pred, None, 0, 255, cv2.NORM_MINMAX)

        bolus_composite = np.zeros((bolus_mask_pred.shape[0], bolus_mask_pred.shape[1], 3))
        bolus_composite[:, :, 0] = bolus_label_to_show
        bolus_composite[:, :, 2] = bolus_mask_pred
        pharynx_composite = np.zeros((pharynx_mask_pred.shape[0], pharynx_mask_pred.shape[1], 3))
        pharynx_composite[:, :, 0] = pharynx_label_to_show
        pharynx_composite[:, :, 2] = pharynx_mask_pred


        # img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        image_three_layers = np.zeros((bolus_mask_pred.shape[0], bolus_mask_pred.shape[1], 3))
        image_three_layers[:, :, 0] = image
        image_three_layers[:, :, 1] = image
        image_three_layers[:, :, 2] = image

        bolus_composite_img = image_three_layers + bolus_composite
        pharynx_composite_img = image_three_layers + pharynx_composite

        

        img_concats = np.concatenate((bolus_composite_img, pharynx_composite_img), axis=1)

        
        #set figures
        def add_text(postion, color, title_text):
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = postion
            fontScale              = 0.4
            fontColor              = color
            lineType               = 1

            cv2.putText(img_concats,
                title_text, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
        #set figures
        txt1 = add_text((380-256,30), (255,0,0), '-Ground Truth')
        txt2 = add_text((380-256,45), (0,0,255), '-Test')
        txt3 = add_text((380-256,60), (0,0,0), '-DSC: {:.4f}'.format(bolus_metric[0]))
        txt3_5 = add_text((380-256,75), (0,0,0), '-ASD: {:.4f}'.format(bolus_metric[1]))
        txt4 = add_text((350+256-256,30), (255,0,0), '-Ground Truth')
        txt5 = add_text((350+256-256,45), (0,0,255), '-Test')
        txt6 = add_text((350+256-256,60), (0,0,0), '-DSC: {:.4f}'.format(pharynx_metrc[0]))
        txt6_5 = add_text((350+256-256,75), (0,0,0), '-ASD: {:.4f}'.format(pharynx_metrc[1]))


        img_concats = cv2.normalize(img_concats, None, 0, 430, cv2.NORM_MINMAX)
        cv2.imwrite('./test_log/results/{}.png'.format(case), img_concats) 

    return bolus_metric, pharynx_metrc
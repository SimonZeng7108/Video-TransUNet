import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    bolus_label = label[0, :, :]
    pharynx_label = label[1, :, :]
    k = np.random.randint(0, 4)
    for i in range(image.shape[0]):
        image[i, :, :] = np.rot90(image[i, :, :], k).copy()
    bolus_label = np.rot90(bolus_label, k)
    pharynx_label = np.rot90(pharynx_label, k)

    axis = np.random.randint(0, 2)
    for i in range(image.shape[0]):
        image[i, :, :] = np.flip(image[i, :, :], axis=axis).copy()
    bolus_label = np.flip(bolus_label, axis=axis).copy()
    pharynx_label = np.flip(pharynx_label, axis=axis).copy()

    label[0, :, :] = bolus_label
    label[1, :, :] = pharynx_label
    return image, label


def random_rotate(image, label):
    bolus_label = label[0, :, :]
    pharynx_label = label[1, :, :]
    angle = np.random.randint(-20, 20)
    for i in range(image.shape[0]):
        image[i, :, :] = ndimage.rotate(image[i, :, :], angle, order=0, reshape=False).copy()
    bolus_label = ndimage.rotate(bolus_label, angle, order=0, reshape=False)
    pharynx_label = ndimage.rotate(pharynx_label, angle, order=0, reshape=False)
    label[0, :, :] = bolus_label
    label[1, :, :] = pharynx_label
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.numpy()
        label = label.numpy()


        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape[1], image.shape[2]
        
        if x != self.output_size[0] or y != self.output_size[1]:
            bolus_label = label[0, :, :]
            pharynx_label = label[1, :, :]
            newimage = np.zeros((image.shape[0], self.output_size[0], self.output_size[1]))
            for i in range(image.shape[0]):
                newimage[i, :, :] = zoom(image[i, :, :], (self.output_size[0] / x, self.output_size[1] / y), order=0).copy()  # why not 3?
            bolus_label = zoom(bolus_label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            pharynx_label = zoom(pharynx_label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            newlabel = np.zeros((2, bolus_label.shape[0], bolus_label.shape[1]))
            newlabel[0, :, :] = bolus_label
            newlabel[1, :, :] = pharynx_label
        newimage = torch.from_numpy(newimage.astype(np.float32))
        newlabel = torch.from_numpy(newlabel.astype(np.float32))
        sample = {'image': newimage, 'label': newlabel.long()}
        
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt').replace('\\', '/')).readlines()
        
        self.data_dir = base_dir

    def __len__(self):

        return len(self.sample_list)
        
    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path, allow_pickle=True)
            image, label = data['image'], data['label']

        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npz".format(vol_name)
            data = np.load(filepath, allow_pickle=True)
            image, label = data['image'], data['label']
            image = image.numpy()
            newimage = np.zeros((image.shape[0], 224, 224))
            for i in range(image.shape[0]):
                newimage[i, :, :] = torch.from_numpy((zoom(image[i, :, :], (224 / 512, 224 / 512), order=0).copy()).astype(np.float32)).unsqueeze(0)
            image = newimage

            bolus_label = label[0, :, :]
            pharynx_label = label[1, :, :]
            bolus_label = zoom(bolus_label, (224 / 512, 224 / 512), order=0)
            pharynx_label = zoom(pharynx_label, (224 / 512, 224 / 512), order=0)
            
            bolus_label = torch.from_numpy(bolus_label.astype(np.float32))
            pharynx_label = torch.from_numpy(pharynx_label.astype(np.float32))

            newlabel = np.zeros((2, bolus_label.shape[0], bolus_label.shape[1]))
            newlabel[0, :, :] = bolus_label
            newlabel[1, :, :] = pharynx_label
            label = newlabel

        label[label > 0.5] = 1
        label[label <= 0.5] = 0
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

if __name__ == '__main__':
    root_path = '../data/Synapse/train_npz'
    list_dir = './lists/lists_Synapse'
    img_size = 224
    #"""train data"""
    # from torchvision import transforms
    # db_train = Synapse_dataset(
    #     base_dir=root_path, list_dir=list_dir, split="train",
    #     transform=transforms.Compose([RandomGenerator(output_size=[img_size, img_size])]))

    # from torch.utils.data import DataLoader
    # batch_size = 1
    # seed = 1234
    # def worker_init_fn(worker_id):
    #     random.seed(seed + worker_id)
    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    #                         worker_init_fn=worker_init_fn)
    
    # for sample in trainloader:
    #     image, mask = sample['image'][0], sample['label'][0]
    #     bolus_mask = mask[0, :, :]
    #     pharynx_bolus = mask[1, :, :]
    #     # break

    #     from torchvision.transforms.functional import to_pil_image
    #     import matplotlib.pylab as plt
    #     from skimage.segmentation import mark_boundaries
    #     def show_img_mask(img, mask): 
    #         if torch.is_tensor(img):
    #             img=to_pil_image(img)
    #             mask=to_pil_image(mask)

    #         img_mask=mark_boundaries(
    #             np.array(img), 
    #             np.array(mask),
    #             outline_color=(0,1,0),
    #             color=(0,1,0))
    #         plt.imshow(img_mask)
    #     image=to_pil_image(image)
    #     mask=mask.cpu().detach().numpy()        

    #     plt.figure('demo image')
    #     plt.subplot(1, 3, 1) 
    #     plt.imshow(image, cmap="gray")

    #     plt.subplot(1, 3, 2) 
    #     show_img_mask(image, bolus_mask)

    #     plt.subplot(1, 3, 3) 
    #     show_img_mask(image, pharynx_bolus)
    #     plt.show()

    #"""val data"
    from sklearn.model_selection import ShuffleSplit
    from torch.utils.data import Subset
    from torchvision import transforms
    from torch.utils.data import DataLoader
    db_train = Synapse_dataset(base_dir=root_path, list_dir=list_dir, split="train",
    transform=transforms.Compose([RandomGenerator(output_size=[img_size, img_size])]))
    db_val = Synapse_dataset(base_dir=root_path, list_dir=list_dir, split="val", transform=None)

    #Split data into train/validation set
    sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    indices=range(len(db_train))

    for train_index, val_index in sss.split(indices):
        pass

    train_ds=Subset(db_train, train_index)
    val_ds=Subset(db_val, val_index)

    def worker_init_fn(worker_id):
        random.seed(1234 + worker_id)
    batch_size = 8
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                            worker_init_fn=worker_init_fn)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    for sample in trainloader:
        image, mask = sample['image'][0], sample['label'][0]

        bolus_mask = mask[0, :, :].cpu().detach().numpy()       
        pharynx_bolus = mask[1, :, :].cpu().detach().numpy()


        # break

        from torchvision.transforms.functional import to_pil_image
        import matplotlib.pylab as plt
        from skimage.segmentation import mark_boundaries
        def show_img_mask(img, mask): 
            if torch.is_tensor(img):
                img=to_pil_image(img)
                mask=to_pil_image(mask)

            img_mask=mark_boundaries(
                np.array(img), 
                np.array(mask),
                outline_color=(0,1,0),
                color=(0,1,0))
            return img_mask

        contactedimages = np.zeros((image.shape))
        for i in range(image.shape[0]):
            contactedimages[i, :, :] = to_pil_image(image[i, :, :])
        print(contactedimages.shape)



        fig = plt.figure('demo image')
        for i in range(image.shape[0]):
            if i == (image.shape[0]-1) /2:
                ax1 = plt.subplot(3, image.shape[0], i+1)
            else:
                plt.subplot(3, image.shape[0], i+1) 
            plt.axis('off')
            plt.imshow(contactedimages[i, :,:], cmap="gray")

        ax2 = plt.subplot(3, int(image.shape[0]), int(image.shape[0] + (image.shape[0]+1)/2)) 
        plt.axis('off')
        plt.imshow(bolus_mask, cmap="gray")
        ax3 = plt.subplot(3, int(image.shape[0]), int(image.shape[0]*2 + (image.shape[0]+1)/2)) 
        plt.axis('off')
        plt.imshow(pharynx_bolus, cmap="gray")

        ax1.title.set_text('image')
        ax2.title.set_text('bolus')
        ax3.title.set_text('pharynx')

        plt.show()

        # plt.figure('demo image')
        # plt.subplot(1, 3, 1) 
        # plt.imshow(image, cmap="gray")

        # plt.subplot(1, 3, 2) 
        # show_img_mask(image, bolus_mask)

        # plt.subplot(1, 3, 3) 
        # show_img_mask(image, pharynx_bolus)
        # plt.show()


    # '''test data'''
    # root_path = '../data/Synapse/test_vol_h5'
    # list_dir = './lists/lists_Synapse'
    # img_size = 224
    # from torchvision import transforms
    # db_test = Synapse_dataset(base_dir=root_path, split="test_vol", list_dir=list_dir)

    # from torch.utils.data import DataLoader
    # testloader = DataLoader(db_test, batch_size=2, shuffle=False, num_workers=0)

    # for sample in testloader:
    #     image, mask = sample['image'][0], sample['label'][0]
    #     print(image.shape)
        

    #     from torchvision.transforms.functional import to_pil_image
    #     import matplotlib.pylab as plt
    #     from skimage.segmentation import mark_boundaries
    #     def show_img_mask(img, mask): 
    #         if torch.is_tensor(img):
    #             img=to_pil_image(img)
    #             mask=to_pil_image(mask)

    #         img_mask=mark_boundaries(
    #             np.array(img, dtype = np.uint8), 
    #             np.array(mask, dtype = np.int64), 
    #             outline_color=(0,1,0),
    #             color=(0,1,0))
    #         plt.imshow(img_mask)

    #     image=to_pil_image(image)
    #     mask=mask.cpu().detach().numpy()        

    #     plt.figure('demo image')
    #     plt.subplot(1, 3, 1) 
    #     plt.imshow(image, cmap="gray")

    #     plt.subplot(1, 3, 2) 
    #     plt.imshow(mask, cmap="gray")

    #     plt.subplot(1, 3, 3) 
    #     show_img_mask(image, mask)
    #     plt.show()


    
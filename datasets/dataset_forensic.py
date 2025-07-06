import os
import random
from builtins import print

import numpy as np
import torch
from torchvision import transforms
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from PIL import Image
from torch.utils.data import Dataset
# debug
from torch.utils.data import DataLoader


def augmentations(image):  # not use
    transform_1 = transforms.RandomApply(torch.nn.Sequential([
        transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]), p=0.2)
    transform_2 = transforms.RandomApply(torch.nn.ModuleList([
        transforms.ColorJitter(brightness=(0, 10),
                               contrast=(0, 10),
                               saturation=(0, 10),
                               hue=0, )]), p=0.2)
    transform_3 = transforms.RandomPerspective(distortion_scale=0.5, p=0.2)

    transform = transforms.Compose([transform_1,
                                    transform_2,
                                    transform_3])
    image = transform(image)

    return image


def check_aug(image, label):
    transform = transforms.ToPILImage()
    img = transform(image)
    lab = transform(label)
    img.save('1.jpg')
    lab.save('1_lab.png')


'''
class RandomGenerator(object):
    def __init__(self, split, size):
        self.split = split
        self.size = size
        assert isinstance(split, str)
        #self.guassblur = transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    def random_rot_flip(self, image, label, degree=20):
        if random.random() > 0.5: # 0.5 probability horizontal flip
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:  # 0.5 probability vertical flip
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() > 0.5:
            angle = [Image.ROTATE_90, Image.ROTATE_270]
            k = random.randint(0, 1)
            image = image.transpose(angle[k])
            label = label.transpose(angle[k])
        if random.random() > 0.8: # 0.2 probability random rotation from -10 to 10 degrees
            angle = np.random.randint(-1 * degree, degree)
            image = image.rotate(angle,fillcolor="black")
            label = label.rotate(angle,fillcolor="black")
        #image.show()
        #label.show()
        return image, label

    def random_scale_crop(self, image, label):
        if random.random() > 0.8:
            w, h = image.size
            crop_w_size = random.randint(int(w * 0.85), int(w * 0.99))
            crop_h_size = random.randint(int(h * 0.85), int(h * 0.99))
            left = random.randint(0, (w - crop_w_size))
            top = random.randint(0, (h - crop_h_size))
            image = image.crop((left, top, left + crop_w_size, top + crop_h_size))
            label = label.crop((left, top, left + crop_w_size, top + crop_h_size))
        #image.show()
        #label.show()
        return image, label

    def __call__(self, image, label):
        if self.split == 'train' or self.split == 'fine_tuned':
            image, label = self.random_rot_flip(image, label)
            #image, label = self.random_scale_crop(image,label)
            #if random.random() > 0.8:
             #   image = self.guassblur(image)
        w, h = image.size
        image = np.array(image)
        label = np.array(label) / 255

        if h != self.size[0] or w != self.size[1]:
            image = zoom(image, (self.size[0] / h, self.size[1] / w, 1), order=3)
            label = zoom(label, (self.size[0] / h, self.size[1] / w), order=0)

        image = self.transform(image)
        label = torch.from_numpy(label).float()
        #print('transed_label:', label.shape, label.min(), label.max())
        #print('transed_image:',image.shape,image.min(),image.max())
        #check_aug(image,label)
        return image, label
'''


class RandomGenerator(object):
    def __init__(self, split, size):
        self.split = split
        self.size = size
        assert isinstance(split, str)
        # self.guassblur = transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def random_rot_flip(self, image, label):
        if random.random() > 0.5:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
        return image, label

    def random_rotate(self, image, label):
        if random.random() > 0.5:
            angle = np.random.randint(-20, 20)
            image = ndimage.rotate(image, angle, order=0, reshape=False)
            label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label

    def random_scale_crop(self, image, label):
        if random.random() > 0.8:
            h, w, _ = image.shape
            crop_w_size = random.randint(int(w * 0.9), int(w * 0.99))
            crop_h_size = random.randint(int(h * 0.9), int(h * 0.99))
            left = random.randint(0, (w - crop_w_size))
            top = random.randint(0, (h - crop_h_size))
            image = image[top:top + crop_h_size, left:left + crop_w_size]
            label = label[top:top + crop_h_size, left:left + crop_w_size]
        return image, label

    def __call__(self, image, label):

        if self.split == 'train' or self.split == 'fine_tuned':
            image, label = self.random_scale_crop(image, label)
            image, label = self.random_rot_flip(image, label)
            image, label = self.random_rotate(image, label)
            # if random.random() > 0.8:
            #   image = self.guassblur(image)

        h, w, _ = image.shape
        if h != self.size[0] or w != self.size[1]:
            image = zoom(image, (self.size[0] / h, self.size[1] / w, 1), order=3)
            label = zoom(label, (self.size[0] / h, self.size[1] / w), order=0)

        image = self.transform(image)
        label = torch.from_numpy(label).float()
        # print('transed_label:', label.shape, label.min(), label.max())
        # print('transed_image:',image.shape,image.min(),image.max())
        # check_aug(image,label)
        return image, label


class Forensic_dataset(Dataset):
    def __init__(self, args, base_dir, list_dir, split, img_size):

        if split == 'train':
            data_names = args.train_data
            # data_names = ['DEFACTO']#debgu
        elif split == 'val':
            data_names = [args.val_data]
        elif split == 'fine_tuned':
            data_names = args.ft_data
        elif split == 'test':
            data_names = [args.test_data]
        else:
            data_names = None
        assert data_names != None, 'no datasets'

        self.split = split
        self.data_names = data_names
        self.img_size = img_size
        self.data_dir = base_dir
        self.list_dir = list_dir
        self.transform = RandomGenerator(split=split, size=img_size)  # using transform in torch!
        self.sample_list = self.get_sample_list(data_names)

    def get_sample_list(self, data_names):
        sample_list = []
        for name in data_names:
            sample_list.extend(
                open(os.path.join(self.list_dir, 'lists_' + name, name + '_' + self.split + '.txt')).readlines())
        print(len(sample_list))
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = {}
        data_path = self.sample_list[idx].strip('\n')
        '''
        data = np.load(data_path)
        image, label = data['image'], data['label']
        sample_name = data_path.split('/')[-1].strip('.npz')
        cls = np.max(label)

        '''
        img_path, gt_path, cls = data_path.split()
        sample_name = img_path.split('/')[-1]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        label = np.array(Image.open(gt_path).convert('L')) / 255
        '''
        if data_path.endswith('.npz'):
            data = np.load(data_path)
            image, label = data['image'], data['label']
            sample_name = data_path.split('/')[-1].strip('.npz')
            cls = np.max(label)
        else:
            img_path, gt_path, cls = data_path.split()
            sample_name = img_path.split('/')[-1]
            image = Image.open(img_path).convert('RGB')
            w, h = image.size
            image = np.array( image)
            if gt_path != 'None':
                label = np.array(Image.open(gt_path).convert('L')) / 255
            else:
                # label = Image.fromarray(np.zeros((h,w))).convert('L')
                label = np.zeros((h, w))
        '''
        if self.transform != None:
            image, label = self.transform(image, label)

        sample['image'] = image
        sample['label'] = label
        sample['name'] = sample_name
        sample['cls'] = cls
        return sample


if __name__ == '__main__':
    torch.utils.data.get_worker_info()
    root_path = '.../datasets'
    list_path = '../lists'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default=['NIST2016'],)
    args = parser.parse_args()
    test_data = Forensic_dataset(args=args, base_dir=root_path, list_dir=list_path, split='train', img_size=(256, 256))
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=0, pin_memory=True,
                             worker_init_fn=random.seed(42))

    for i, sample_batch in enumerate(test_loader):
        image, mask = sample_batch['image'], sample_batch['label']




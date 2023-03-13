from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import numpy as np
import time
from tqdm import tqdm
from networks.models import Generator
import torch.nn.functional as F
import torch.utils.data as data
import os
import csv
from PIL import Image


class Dataset_dynamic(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, mode="train", device=torch.device("cuda")):
        self.device = device
        self.target_type = opt.target_type
        self.dataset = self.addTrigger(opt, full_dataset, opt.target_label, inject_portion, mode)

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, opt, dataset, target_label, inject_portion, mode):
        state_dict = torch.load('./data/all2one_cifar10_ckpt.pth.tar')
        netG = Generator(opt)
        netG.load_state_dict(state_dict["netG"])
        # netG.to(opt.device)
        netG.eval()
        netG.requires_grad_(False)
        netM = Generator(opt, out_channels=1)
        netM.load_state_dict(state_dict["netM"])
        # netM.to(opt.device)
        netM.eval()
        netM.requires_grad_(False)

        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if self.target_type == 'all2one':

                if mode == 'train':
                    # img = np.array(data[0])
                    img = data[0].unsqueeze(0)
                    if i in perm:
                        # select trigger
                        patterns = netG(img)
                        patterns = netG.normalize_pattern(patterns)
                        masks_output = netM.threshold(netM(img))
                        img = img + (patterns - img) * masks_output

                        # change target
                        dataset_.append((np.array(img.squeeze(0)), target_label))
                        cnt += 1
                    else:
                        dataset_.append((np.array(img.squeeze(0)), data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    # img = np.array(data[0])
                    img = data[0].unsqueeze(0)
                    if i in perm:
                        patterns = netG(img)
                        patterns = netG.normalize_pattern(patterns)
                        masks_output = netM.threshold(netM(img))
                        img = img + (patterns - img) * masks_output

                        dataset_.append((np.array(img.squeeze(0)), target_label))
                        cnt += 1
                    else:
                        dataset_.append((np.array(img.squeeze(0)), data[1]))
        return dataset_


class Dataset_wanet(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, mode="train", device=torch.device("cuda")):
        self.device = device
        self.target_type = opt.target_type
        self.dataset = self.addTrigger(opt, full_dataset, opt.target_label, inject_portion, mode)

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, opt, dataset, target_label, inject_portion, mode):
        state_dict = torch.load('./data/gtsrb_all2one_morph.pth.tar')
        identity_grid = state_dict["identity_grid"].cpu()
        noise_grid = state_dict["noise_grid"].cpu()

        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if self.target_type == 'all2one':

                if mode == 'train':
                    # img = np.array(data[0])
                    img = data[0].unsqueeze(0)
                    if i in perm:
                        # select trigger
                        grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
                        grid_temps = torch.clamp(grid_temps, -1, 1)
                        img = F.grid_sample(img, grid_temps.repeat(1, 1, 1, 1), align_corners=True)

                        # change target
                        dataset_.append((np.array(img.squeeze(0)), target_label))
                        cnt += 1
                    else:
                        dataset_.append((np.array(img.squeeze(0)), data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    # img = np.array(data[0])
                    img = data[0].unsqueeze(0)
                    if i in perm:
                        grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
                        grid_temps = torch.clamp(grid_temps, -1, 1)
                        img = F.grid_sample(img, grid_temps.repeat(1, 1, 1, 1), align_corners=True)

                        dataset_.append((np.array(img.squeeze(0)), target_label))
                        cnt += 1
                    else:
                        dataset_.append((np.array(img.squeeze(0)), data[1]))
        return dataset_


class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms=None):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()
        # self.dataset = list(zip(self.images,self.labels))

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.labels[index]
        return image, label


def get_train_loader(opt):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomRotation(3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(1, 3)
    ])

    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True)
    else:
        raise Exception('Invalid dataset')

    train_data = DatasetCL(opt, full_dataset=trainset, transform=tf_train)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    return train_loader


def get_test_loader(opt):
    print('==> Preparing test data..')
    tf_test = transforms.Compose([transforms.ToTensor()])
    if (opt.dataset == 'CIFAR10'):
        testset = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True)
    elif (opt.dataset == 'gtsrb'):
        tf = transforms.Compose([transforms.Resize((opt.input_height, opt.input_width))])
        gtsrb = GTSRB(opt, train=False, transforms=tf)
        testset = []
        for i in tqdm(range(len(gtsrb))):
            img, label = gtsrb[i]
            testset.append((img, label))
    else:
        raise Exception('Invalid dataset')

    test_data_clean = DatasetBD(opt, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
    test_data_bad = DatasetBD(opt, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test')
    if opt.trigger_type == 'dynamic' and (opt.dataset == 'CIFAR10'):
        transform = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True,
                                   transform=transform)
        test_data_bad = Dataset_wanet(opt, full_dataset=testset, inject_portion=1, mode='test')

    if opt.trigger_type == 'dynamic' and (opt.dataset == 'gtsrb'):
        tf = transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), transforms.ToTensor()])
        gtsrb = GTSRB(opt, train=False, transforms=tf)
        testset = []
        for i in tqdm(range(len(gtsrb))):
            img, label = gtsrb[i]
            testset.append((img, label))
        test_data_bad = Dataset_dynamic(opt, full_dataset=testset, inject_portion=1, mode='test')

    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                   batch_size=opt.batch_size,
                                   shuffle=False,
                                   num_workers= 4,
                                   )
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 )

    return test_clean_loader, test_bad_loader


def get_backdoor_loader(opt):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(1, 3)
    ])
    if (opt.dataset == 'CIFAR10'):
        transform = transforms.Compose([transforms.ToTensor()])
        if opt.trigger_type == 'dynamic':
            trainset = datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform)
        else:
            trainset = datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True)
    elif (opt.dataset == 'gtsrb'):
        if opt.trigger_type == 'dynamic':
            tf = transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), transforms.ToTensor()])
        else:
            tf = transforms.Compose([transforms.Resize((opt.input_height, opt.input_width))])
        gtsrb = GTSRB(opt, train=True, transforms=tf)
        trainset = []
        for i in tqdm(range(len(gtsrb))):
            img, label = gtsrb[i]
            trainset.append((img, label))
        # trainset = datasets.GTSRB(root='./data/gtsrb', train=True, download=True)
    else:
        raise Exception('Invalid dataset')

    train_data_bad = DatasetBD(opt, full_dataset=trainset, inject_portion=opt.inject_portion, transform=tf_train,
                               mode='train')
    # train_data_bad = Dataset_wanet(opt, full_dataset=trainset, inject_portion=opt.inject_portion, mode='train')
    train_bad_loader = DataLoader(dataset=train_data_bad,
                                  batch_size=opt.batch_size,
                                  shuffle=True, num_workers=4)

    return train_data_bad, train_bad_loader


class Dataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        ind = 0

        if self.transform:
            image = self.transform(image)
        # print(type(image), image.shape)
        return image, label, ind

    def __len__(self):
        return self.dataLen


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class DatasetCL(Dataset):
    def __init__(self, opt, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=opt.ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset


class DatasetBD(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda"),
                 distance=1):
        self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance, opt.trig_w,
                                       opt.trig_h, opt.trigger_type, opt.target_type)
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        ind = self.dataset[item][2]
        img = self.transform(img)

        return img, label, ind

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type,
                   target_type):
        print("Generating " + mode + "bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if target_type == 'all2one':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        # select trigger
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        # change target
                        dataset_.append((img, target_label, 1))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1], 0))

                else:
                    if data[1] == target_label and inject_portion != 0.:
                        continue

                    img = np.array(data[0], dtype=np.uint8)
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label, 0))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1], 1))

            # all2all attack
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:

                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif target_type == 'cleanLabel':

                if mode == 'train':
                    img = np.array(data[0], dtype=np.uint8)
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                            dataset_.append((img, data[1]))
                            cnt += 1

                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0], dtype=np.uint8)
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")

        return dataset_

    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger', 'dynamic']

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)

        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # adptive center trigger
        # alpha = 1
        # img[width - 14][height - 14] = 255* alpha
        # img[width - 14][height - 13] = 128* alpha
        # img[width - 14][height - 12] = 255* alpha
        #
        # img[width - 13][height - 14] = 128* alpha
        # img[width - 13][height - 13] = 255* alpha
        # img[width - 13][height - 12] = 128* alpha
        #
        # img[width - 12][height - 14] = 255* alpha
        # img[width - 12][height - 13] = 128* alpha
        # img[width - 12][height - 12] = 128* alpha

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.1
        # load signal mask
        '''
        signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10'''
        img = np.float32(img)
        pattern = np.zeros_like(img)
        m = pattern.shape[1]
        f = 6
        delta = 20
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)
        img = alpha * np.uint32(img) + (1 - alpha) * pattern
        blend_img = np.clip(img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load trojanmask
        trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        return img_

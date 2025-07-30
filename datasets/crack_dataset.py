import os.path
import cv2
from PIL import Image
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from .image_folder import make_dataset
from .utils import MaskToTensor

class CrackDataset(BaseDataset):

    def __init__(self, args):
        BaseDataset.__init__(self, args)
        self.inference_mask = args.inference_mask
        self.modals = args.modals
        self.modal_num = len(args.modals)

        self.modal_img_paths = []

        for i in range(self.modal_num):
            self.data_temp = make_dataset(os.path.join(args.dataset_path, '{}_img_{}'.format(args.phase, (self.modals)[i])))
            self.modal_img_paths.append(self.data_temp)

        self.lab_dir = os.path.join(args.dataset_path, '{}_lab'.format(args.phase))

        if not self.inference_mask:
            self.mask_dir = os.path.join(args.dataset_path, '{}_mask'.format(args.phase))
        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])
        self.lab_transform = MaskToTensor()
        self.phase = args.phase

    def split_into_sublists(self, modal_img_path, n):
        if len(modal_img_path) % n != 0:
            raise ValueError("The list length must be an integer multiple of n.")
        sublist_length = len(modal_img_path) // n

        sublists = [
            modal_img_path[i * sublist_length: (i + 1) * sublist_length]
            for i in range(n)
        ]
        return sublists

    def __getitem__(self, index):
        modal_img_path = []

        for i in range(self.modal_num):
            modal_img_path.append((self.modal_img_paths[i])[index])

        sub_modal_img_path = self.split_into_sublists(modal_img_path, self.modal_num)
        lab_path = os.path.join(self.lab_dir, os.path.basename(modal_img_path[0]).split('.')[0] + '.png')

        if not self.inference_mask:
            mask_path = os.path.join(self.mask_dir, os.path.basename(modal_img_path[0]).split('.')[0] + '.png')

        w, h = self.args.load_width, self.args.load_height
        imgs = []
        for i in range(self.modal_num):
            imgs.append(cv2.resize(cv2.cvtColor(cv2.imread(sub_modal_img_path[i][0], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_CUBIC))

        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)
        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
        lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)

        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)

        imgs_transfer = []
        for i in range(self.modal_num):
            imgs_transfer.append(self.img_transforms(Image.fromarray(imgs[i].copy())))

        lab = self.lab_transform(lab.copy()).unsqueeze(0)

        returned_data = dict()
        for i in range(self.modal_num):
            returned_data[self.modals[i]] = imgs_transfer[i]

        returned_data['label'] = lab
        returned_data['image_path'] = modal_img_path[0]
        returned_data['label_path'] = lab_path

        if not self.inference_mask:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            mask = self.lab_transform(mask.copy()).unsqueeze(0)
            returned_data['mask'] = mask

        return returned_data

    def __len__(self):
        return len((self.modal_img_paths)[0])



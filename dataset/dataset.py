import os
import torch
import numpy as np
from torch.utils.data import Dataset
import PIL
from PIL import Image
from os.path import join as pjoin, splitext as spt
import dataset.transforms as T 
from torchvision.transforms import functional as F

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class SCDDataset(Dataset):
    def __init__(self, root, transforms=None):
        super(SCDDataset, self).__init__()
        self.root = root
        self.gt, self.t0, self.t1 = [], [], []
        self.transforms = transforms
        self._revert_transforms = None
        self.name = ''
        self.num_classes = 2 
    
    def _check_validness(self, f):
        return any([i in spt(f)[1] for i in ['jpg', 'png']])
    
    def _pil_loader(self, path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def _init_data_list(self):
        pass
    
    def get_raw(self, index):
        fn_t0 = self.t0[index]
        fn_t1 = self.t1[index]
        fn_mask = self.gt[index]

        
        img_t0 = self._pil_loader(fn_t0)
        img_t1 = self._pil_loader(fn_t1)
        imgs = [img_t0, img_t1]
        
        # 흑백의 Image 자료형을 반환한다. raw이미지에서 물체의 마스크는 0 또는 255외의 값을 갖는다. ex) 43.0
        mask = self._pil_loader(fn_mask).convert('L')
                


        return imgs, mask

    def __getitem__(self, index):
        imgs, mask = self.get_raw(index)
        if self.transforms is not None:
            imgs, mask = self._transforms(imgs, mask)
        return imgs, mask
    
    def __len__(self):
        return len(self.gt)
    
    def get_mask_ratio(self):
        all_count = 0
        mask_count = 0
        for i in range(len(self.gt)):
            # 흑백으로 변환된 이미지를 가져옴
            _ , mask = self.get_raw(i)
            # Image자료형을 텐서로 변환
            target = (F.to_tensor(mask) != 0).long() # = 검은색이 아닌 부분이 True값
            mask_count += target.sum() 
            all_count  += target.numel() # torch.numel(input) 전체 element개수 반환
        mask_ratio = mask_count / float(all_count)
        background_ratio = (all_count - mask_count) / float(all_count)
        return [mask_ratio, background_ratio]
    
    def get_mask_ratio_pscd(self):
        all_count = 0
        mask_count = 0
        all_count_t1 = 0
        mask_count_t1 = 0
        for i in range(len(self.gt)):
            # 흑백으로 변환된 이미지를 가져옴
            _ , masks = self.get_raw(i)
            # Image자료형을 텐서로 변환
            mask = masks[0]
            target = (F.to_tensor(mask) != 0).long() # = 검은색이 아닌 부분이 True값
            mask_count += target.sum() 
            all_count  += target.numel() # torch.numel(input) 전체 element개수 반환
            
        for i in range(len(self.gt_t1)):
            # 흑백으로 변환된 이미지를 가져옴
            _ , masks = self.get_raw(i)
            # Image자료형을 텐서로 변환
            mask = masks[1]
            target = (F.to_tensor(mask) != 0).long() # = 검은색이 아닌 부분이 True값
            mask_count_t1 += target.sum() 
            all_count_t1  += target.numel() # torch.numel(input) 전체 element개수 반환
            
        mask_ratio = mask_count / float(all_count)
        background_ratio = (all_count - mask_count) / float(all_count)
        
        mask_ratio_t1 = mask_count_t1 / float(all_count_t1)
        background_ratio_t1 = (all_count_t1 - mask_count_t1) / float(all_count_t1)
        
        return [mask_ratio, background_ratio], [mask_ratio_t1, background_ratio_t1]

    def get_pil(self, imgs, mask_gt, mask_pred_t0, mask_pred_t1):
        assert self._revert_transforms is not None
        t0, t1 = self._revert_transforms(imgs.cpu())

        if 'PSCD' in self.root or 'TSUNMAI' in self.root:
            t0, t1 = T.Resize((256,1024))(t0, t1)
        else:
            # VL-CMU-CD
            t0, t1 = T.Resize((512,512))(t0, t1)
        
        w,h = t0.size
        output = Image.new('RGB', (w*3, h*2))
        output.paste(t0)
        output.paste(t1, (w,0))

        # groundtruth binary mask
        mask = F.to_pil_image(mask.cpu().float())
        output.paste(mask_gt, (2*w,0))

        pred_t0 = F.to_pil_image(mask_pred_t0.cpu().float())
        output.paste(pred_t0, (0, h))
        pred_t1 = F.to_pil_image(mask_pred_t1.cpu().float())
        output.paste(pred_t1, (w, h))

        return output
    
def get_transforms(args, train, size_dict=None):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if size_dict is not None:
        assert args.input_size in size_dict, "input_size: {}".format(size_dict.keys())
        input_size = size_dict[args.input_size]
    else:
        input_size = args.input_size
    
    mode = "Train" if train else "Test"
    print("{} Aug:".format(mode))
    augs = []
    if train:
        if args.randomcrop:
            if args.input_size == 256:
                augs.append(T.Resize(286))
                augs.append(T.RandomCrop(256))
            elif args.input_size == 224:
                augs.append(T.RandomCrop(224))
            elif args.input_size == 512:
                augs.append(T.RandomCrop(512))
            elif args.input_size == 1024:
                augs.append(T.RandomCrop(1024))
                # augs.append(T.Resize((1024, 1024)))
            else:
                raise ValueError(args.input_size)
        else:
            # augs.append(T.Resize((1024, 1024)))
            augs.append(T.Resize(input_size))
                
        augs.append(T.RandomHorizontalFlip(args.randomflip))
    else:
        if 'TSUNAMI' in args.test_dataset2 or 'TSUNAMI' in args.test_dataset3:
            test_size = (256,1024)
            augs.append(T.Resize(test_size))
        elif 'PSCD' in args.test_dataset2 or 'PSCD' in args.test_dataset3:
            test_size = (256,1024)
            augs.append(T.Resize(test_size))
        else:
            # VL-CMU-CD
            test_size = (512, 512)
            augs.append(T.Resize(test_size))
        print(f'Test image has the shape of {test_size}')
        
    augs.append(T.ToTensor())
    augs.append(T.Normalize(mean=mean, std=std))
    augs.append(T.ConcatImages())
    transforms = T.Compose(augs)
    revert_transforms = T.Compose([
        T.SplitImages(),
        T.RevertNormalize(mean=mean, std=std),
        T.ToPILImage()
    ])
    return transforms, revert_transforms
        




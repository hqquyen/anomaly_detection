import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


CLASS_NAMES = ['object']

class Data_Loader(Dataset):
    def __init__(self, dataset_path='data', class_name='object', is_train=True,
                 resize=400, cropsize=400):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        # load dataset
        self.x, self.y = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)
        return x, y

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y = [], []
        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        img_types = sorted(os.listdir(img_dir))

        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])      
            x.extend(img_fpath_list)
            # load labels
            if img_type == 'normal':
                y.extend([0] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))

        return list(x), list(y)

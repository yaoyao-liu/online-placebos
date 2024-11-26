import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iImageNetAuxCIFAR100All, iImageNetAuxCIFAR100Matching, iImageNetAuxCIFAR100NoMatching, iImageNetAuxCIFAR100ExactMatching


class AuxDataManager(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name)

    def get_raw_dataset(
        self, ret_data=False
    ):
        data, targets = self._train_data, self._train_targets

        trsf = transforms.Compose([*self._trsf])

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_selected_dataset(self, data, targets):

        trsf = transforms.Compose([*self._trsf])
        return DummyDataset(data, targets, trsf, self.use_path)

    def _setup_data(self, dataset_name):
        idata = _get_aux_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._trsf = idata.trsf

class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _get_aux_idata(dataset_name):
    name = dataset_name.lower()
    if name == "imagenet_all":
        return iImageNetAuxCIFAR100All()
    elif name == "imagenet_matching":
        return iImageNetAuxCIFAR100Matching()
    elif name == "imagenet_no_matching":
        return iImageNetAuxCIFAR100NoMatching()
    elif name == "imagenet_exact_matching":
        return iImageNetAuxCIFAR100ExactMatching()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
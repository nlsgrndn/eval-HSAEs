import os
from torchvision import datasets
from torch.utils.data import Dataset

path_to_imagenet1k = "~/thesis/datasets/imagenet"
class ImageNet1kLocalDataset(Dataset):
    """
    Wrapper for the ImageNet 1k dataset stored locally.

    Handles ImageNet 1k data stored in local directory structure with numbered subdirectories
    containing only image files.
    """
    def __init__(self, data_dir=path_to_imagenet1k, data_split='train', preprocess=None, return_None_label=False, load_image=False, return_file_path=False):
        self.data_dir = os.path.expanduser(data_dir)
        self.data_split = data_split
        self.return_None_label = return_None_label
        self.load_image = load_image
        self.return_file_path = return_file_path
        self.dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, data_split),
            transform=preprocess
        )
        self.folder_name_classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.load_image:
            image, label = self.dataset[idx]
        else:
            # Get the image path without loading the actual image
            path, label = self.dataset.samples[idx]
            if self.return_file_path:
                return path, label
            image = None
            
        if self.return_None_label:
            return image, None
        
        ret_dict = {
            'class_label': label,
        }
        if self.load_image:
            ret_dict['image'] = image
        return ret_dict
    
if __name__ == "__main__":
    dataset = ImageNet1kLocalDataset(load_image=True, data_split='val')
    import ipdb; ipdb.set_trace()
import os 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DICOMDataset(Dataset):
    """
    A PyTorch Dataset for loading DICOM axial and coronal images and associated patient data.

    Args:
        data_dir_prefix (str): Prefix for data directories (e.g., 'train' or 'test').
                                Assumes 'data_dir_prefix-axial' and 'data_dir_prefix-coronal'.
        VAlogmars (np.ndarray): NumPy array of VAlogmar values for each sample.
        labels (np.ndarray): NumPy array of labels (e.g., IOSONSbinary) for each sample.
        transform (callable, optional): Optional transform to be applied on a sample.
        target_transform (callable, optional): Optional transform to be applied on a label.
        shared_base_directory (str): Root folder on Drive (or local) containing
            ``{prefix}-axial`` and ``{prefix}-coronal`` subdirectories.
    """
    def __init__(self, data_dir_prefix: str, VAlogmars: np.ndarray,
                 labels: np.ndarray, transform: transforms.Compose = None,
                 target_transform: callable = None,
                 shared_base_directory: str = "/content/drive/MyDrive/Farabi"):
        self.labels = labels
        self.VAlogmars = VAlogmars
        self.axial_images_dir = os.path.join(shared_base_directory, f"{data_dir_prefix}-axial")
        self.coronal_images_dir = os.path.join(shared_base_directory, f"{data_dir_prefix}-coronal")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'axial' (axial image tensor),
                  'coronal' (coronal image tensor), 'VAlogmar' (scalar),
                  and 'label' (scalar).
        """
        axial_image_path = os.path.join(self.axial_images_dir, f'{idx}.npy')
        coronal_image_path = os.path.join(self.coronal_images_dir, f'{idx}.npy')

        data_axial = np.load(axial_image_path)
        data_coronal = np.load(coronal_image_path)

        va_logmar = self.VAlogmars[idx]
        label = self.labels[idx]

        if self.transform:
            data_axial = self.transform(data_axial)
            data_coronal = self.transform(data_coronal)
        if self.target_transform:
            label = self.target_transform(label)

        return {'axial': data_axial, 'coronal': data_coronal, 'VAlogmar': va_logmar, 'label': label}
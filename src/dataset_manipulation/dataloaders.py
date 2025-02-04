import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
#from torchnet.dataset import TensorDataset, ResampleDataset #because I (denis) don't have torchnet
# from .datasets_PolyMNIST import PolyMNISTDataset
# from .dataset_CUB import CUBSentences, resampler
#from dataset_manipulation.datasets_PolyMNIST import PolyMNISTDataset
from dataset_manipulation.datasets_Cityscapes import CityscapesDataset, CityscapesMultiModalDataset
#from dataset_manipulation.dataset_CUB import CUBSentences, resampler


# Constants
maxSentLen = 32  # max length of any description for birds dataset

class DataLoaderFactory:
    def __init__(self, datadir='../data', num_workers=2, pin_memory=True):
        """
        Initializes the DataLoaderFactory with the specified data directory and DataLoader parameters.
        
        Args:
            datadir (str): Path to the data directory.
            num_workers (int): Number of subprocesses to use for data loading.
            pin_memory (bool): If True, the data loader will copy tensors into CUDA pinned memory before returning them.
        """
        self.datadir = datadir
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def get_dataloader_polymnist_unimodal(self, batch_size, shuffle=True, device='cuda', modality=1):
        """
        Get PolyMNIST unimodal DataLoaders for a specific modality.
        
        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            device (str): Device to load data (e.g., 'cuda' or 'cpu').
            modality (int): Modality index (e.g., 1, 2, ...).
        
        Returns:
            tuple: (train_loader, test_loader)
        """
        unim_datapaths_train = [os.path.join(self.datadir, "PolyMNIST", "train", f"m{modality}")]
        unim_datapaths_test = [os.path.join(self.datadir, "PolyMNIST", "test", f"m{modality}")]

        kwargs = {'num_workers': self.num_workers, 'pin_memory': self.pin_memory} if device == 'cuda' else {}
        tx = transforms.ToTensor()

        train_dataset = PolyMNISTDataset(unim_datapaths_train, transform=tx)
        test_dataset = PolyMNISTDataset(unim_datapaths_test, transform=tx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train_loader, test_loader

    def get_dataloader_polymnist_multimodal(self, batch_size, shuffle=True, device='cuda'):
        """
        Get PolyMNIST multimodal DataLoaders combining multiple modalities.
        
        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            device (str): Device to load data (e.g., 'cuda' or 'cpu').
        
        Returns:
            tuple: (train_loader, test_loader)
        """
        modalities = [0, 1, 2, 3, 4]
        unim_train_datapaths = [os.path.join(self.datadir, "PolyMNIST", "train", f"m{i}") for i in modalities]
        unim_test_datapaths = [os.path.join(self.datadir, "PolyMNIST", "test", f"m{i}") for i in modalities]

        kwargs = {'num_workers': self.num_workers, 'pin_memory': self.pin_memory} if device == 'cuda' else {}
        tx = transforms.ToTensor()

        train_dataset = PolyMNISTDataset(unim_train_datapaths, transform=tx)
        test_dataset = PolyMNISTDataset(unim_test_datapaths, transform=tx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train_loader, test_loader

    def get_dataloader_cub_caption(self, batch_size, shuffle=True, device='cuda'):
        """
        Get CUB caption DataLoaders.
        
        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            device (str): Device to load data (e.g., 'cuda' or 'cpu').
        
        Returns:
            tuple: (train_loader, test_loader)
        """
        kwargs = {'num_workers': self.num_workers, 'pin_memory': self.pin_memory} if device == 'cuda' else {}
        tx = lambda data: torch.Tensor(data)

        t_data = CUBSentences(
            self.datadir,
            split='train',
            one_hot=True,
            transpose=False,
            transform=tx,
            max_sequence_length=maxSentLen
        )
        s_data = CUBSentences(
            self.datadir,
            split='test',
            one_hot=True,
            transpose=False,
            transform=tx,
            max_sequence_length=maxSentLen
        )

        train_loader = DataLoader(t_data, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(s_data, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train_loader, test_loader

    def get_dataloader_cub_image(self, batch_size, shuffle=True, device='cuda'):
        """
        Get CUB image DataLoaders.
        
        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            device (str): Device to load data (e.g., 'cuda' or 'cpu').
        
        Returns:
            tuple: (train_loader, test_loader)
        """
        kwargs = {'num_workers': self.num_workers, 'pin_memory': self.pin_memory} if device == 'cuda' else {}
        tx = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor()
        ])

        train_image_dir = os.path.join(self.datadir, 'cub', 'train')
        test_image_dir = os.path.join(self.datadir, 'cub', 'test')

        train_dataset = datasets.ImageFolder(train_image_dir, transform=tx)
        test_dataset = datasets.ImageFolder(test_image_dir, transform=tx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train_loader, test_loader

    def get_dataloader_cub_joint(self, batch_size, shuffle=True, device='cuda'):
        """
        Get joint CUB DataLoaders combining image and caption data.
        Note: This method assumes the existence of ResampleDataset and resampler.
        
        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            device (str): Device to load data (e.g., 'cuda' or 'cpu').
        
        Returns:
            tuple: (train_loader, test_loader)
        """
        # try:
        #     from torchnet.dataset import ResampleDataset, resampler
        # except ImportError:
        #     raise ImportError("Please define or import ResampleDataset and resampler from your_module.")

        # Get individual DataLoaders
        t1, s1 = self.get_dataloader_cub_image(batch_size, shuffle, device)
        t2, s2 = self.get_dataloader_cub_caption(batch_size, shuffle, device)

        kwargs = {'num_workers': self.num_workers, 'pin_memory': self.pin_memory} if device == 'cuda' else {}

        # Resample the image datasets
        train_resampled = ResampleDataset(t1.dataset, resampler, size=len(t1.dataset) * 10)
        test_resampled = ResampleDataset(s1.dataset, resampler, size=len(s1.dataset) * 10)

        # Create TensorDatasets combining resampled images and captions
        train_loader = DataLoader(
            TensorDataset([train_resampled, t2.dataset]),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
        test_loader = DataLoader(
            TensorDataset([test_resampled, s2.dataset]),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )

        return train_loader, test_loader

    def get_dataloader_cityscapes_unimodal(self, batch_size, shuffle=True, device='cuda', modality=0):
        """
        Get Cityscapes unimodal DataLoaders for a specific modality.
        
        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            device (str): Device to load data (e.g., 'cuda' or 'cpu').
            modality (int): Modality index (e.g., 1, 2, ...).
        
        Returns:
            tuple: (train_loader, test_loader)
        """
        #unim_datapaths_train = [os.path.join(self.datadir, "Cityscapes", "train")] # f"m{modality}")]
        #unim_datapaths_test = [os.path.join(self.datadir, "Cityscapes", "test")] # f"m{modality}")]

        #To change: here we work with preprocessed npy files:

        npy_datapath_train = os.path.join(self.datadir, "Cityscapes", "preprocessed", "train_32x64.npy")
        npy_datapath_test = os.path.join(self.datadir, "Cityscapes", "preprocessed", "test_32x64.npy")

        kwargs = {'num_workers': self.num_workers, 'pin_memory': self.pin_memory} if device == 'cuda' else {}
        tx = transforms.ToTensor()

        train_dataset = CityscapesDataset(npy_datapath_train, modality=modality, transform=tx)
        test_dataset = CityscapesDataset(npy_datapath_test, modality=modality, transform=tx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train_loader, test_loader

    def get_dataloader_cityscapes_multimodal(self, batch_size, shuffle=True, device='cuda', subset_percentage=1.0):
        """
        Get Cityscapes multimodal DataLoaders.
        
        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            device (str): Device to load data (e.g., 'cuda' or 'cpu').
        
        Returns:
            tuple: (train_loader, test_loader)
        """

        npy_datapath_train = os.path.join(self.datadir, "Cityscapes", "preprocessed", "train_32x64.npy")
        npy_datapath_test = os.path.join(self.datadir, "Cityscapes", "preprocessed", "test_32x64.npy")

        kwargs = {'num_workers': self.num_workers, 'pin_memory': self.pin_memory} if device == 'cuda' else {}

        train_dataset = CityscapesMultiModalDataset(npy_datapath_train, subset_percentage=subset_percentage)
        test_dataset = CityscapesMultiModalDataset(npy_datapath_test, subset_percentage=subset_percentage)
        print("------------------")
        print("------------------")
        print("------------------")
        print("------------------")
        print(f"bs {batch_size}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train_loader, test_loader

    def get_dataloader(self, dataset_name, batch_size, shuffle=True, device='cuda', **kwargs):
        """
        General method to get DataLoaders based on the dataset name.
        
        Args:
            dataset_name (str): Name of the dataset ('polymnist_unimodal', 'polymnist_multimodal', 'cub_caption', 'cub_image', 'cub_joint', 'cityscapaes_unimodal').
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            device (str): Device to load data (e.g., 'cuda' or 'cpu').
            **kwargs: Additional arguments for specific datasets.
        
        Returns:
            tuple: (train_loader, test_loader)
        """
        dataset_name = dataset_name.lower()
        if dataset_name == 'polymnist_unimodal':
            modality = kwargs.get('modality', 1)
            return self.get_dataloader_polymnist_unimodal(batch_size, shuffle, device, modality)
        elif dataset_name == 'polymnist_multimodal':
            return self.get_dataloader_polymnist_multimodal(batch_size, shuffle, device)
        elif dataset_name == 'cub_caption':
            return self.get_dataloader_cub_caption(batch_size, shuffle, device)
        elif dataset_name == 'cub_image':
            return self.get_dataloader_cub_image(batch_size, shuffle, device)
        elif dataset_name == 'cub_joint':
            return self.get_dataloader_cub_joint(batch_size, shuffle, device)
        elif dataset_name == 'cityscapes_unimodal':
            modality = kwargs.get('modality', 1)
            return self.get_dataloader_cityscapes_unimodal(batch_size, shuffle, device, modality=modality)
        elif dataset_name == 'cityscapes_multimodal':
            subset_percentage = kwargs.get('subset_percentage', 1.0)
            return self.get_dataloader_cityscapes_multimodal(batch_size, shuffle, device, subset_percentage=subset_percentage)
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported. Choose from 'polymnist_unimodal', 'polymnist_multimodal', 'cub_caption', 'cub_image', 'cub_joint', 'cityscapes_unimodal'.")

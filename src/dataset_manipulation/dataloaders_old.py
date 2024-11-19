import torch
from torch.utils.data import DataLoader
from datasets_PolyMNIST import PolyMNISTDataset
from dataset_CUB import CUBSentences

maxSentLen = 32

class getDataloaders():
    def __init__(self,  datadir = '../data')
        self.datadir = datadir
    # Polymnist dataloaders

    def getDataLoaders_unimodal_polymnist(self, batch_size, shuffle=True, device='cuda', modal = 1):
            """Get PolyMNIST modality dataloaders."""
            unim_datapaths_train = [self.datadir+"/PolyMNIST/train/" + "m" + str(modal)]
            unim_datapaths_test = [self.datadir+"/PolyMNIST/test/" + "m" + str(modal)]

            kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
            tx = transforms.ToTensor()
            train = DataLoader(PolyMNISTDataset(unim_datapaths_train, transform=tx),
                               batch_size=batch_size, shuffle=shuffle, **kwargs)
            test = DataLoader(PolyMNISTDataset(unim_datapaths_test, transform=tx),
                               batch_size=batch_size, shuffle=shuffle, **kwargs)
            return train, test


    def getDataSets_polymnist(self, batch_size, shuffle=True, device='cuda'):
            """Get PolyMNIST datasets."""
            tx = transforms.ToTensor()
            unim_train_datapaths = [self.datadir+"/PolyMNIST/train/" + "m" + str(i) for i in [0, 1, 2, 3, 4]]
            unim_test_datapaths = [self.datadir+"/PolyMNIST/test/" + "m" + str(i) for i in [0, 1, 2, 3, 4]]
            dataset_PolyMNIST_train = PolyMNISTDataset(unim_train_datapaths, transform=tx)
            dataset_PolyMNIST_test = PolyMNISTDataset(unim_test_datapaths, transform=tx)
            # kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
            # train = DataLoader(dataset_PolyMNIST_train, batch_size=batch_size, shuffle=shuffle, **kwargs)
            # test = DataLoader(dataset_PolyMNIST_test, batch_size=batch_size, shuffle=shuffle, **kwargs)
            return dataset_PolyMNIST_train, dataset_PolyMNIST_test

    # Cub Image Caption dataloaders


    def getDataLoaders_cub_caption(self, batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        tx = lambda data: torch.Tensor(data)
        t_data = CUBSentences(self.datadir, split='train', one_hot=True, transpose=False, transform=tx, max_sequence_length=maxSentLen)
        s_data = CUBSentences(self.datadir, split='test', one_hot=True, transpose=False, transform=tx, max_sequence_length=maxSentLen)

        train_loader = DataLoader(t_data, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(s_data, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train_loader, test_loader


    # remember that when combining with captions, this should be x10
    def getDataLoaders_cub_image(self, batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        tx = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(self.datadir+'/cub/train', transform=tx),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(self.datadir+'/cub/test', transform=tx),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader, test_loader


    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        # load base datasets
        t1, s1 = self.getDataLoaders_cub_image(batch_size, shuffle, device)
        t2, s2 = self.getDataLoaders_cub_caption(batch_size, shuffle, device)

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train_loader = DataLoader(TensorDataset([
            ResampleDataset(t1.dataset, resampler, size=len(t1.dataset) * 10),
            t2.dataset]), batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(TensorDataset([
            ResampleDataset(s1.dataset, resampler, size=len(s1.dataset) * 10),
            s2.dataset]), batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader, test_loader







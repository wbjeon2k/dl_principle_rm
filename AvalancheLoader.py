import random
import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from avalanche.benchmarks.classic import CORe50, SplitTinyImageNet, SplitCIFAR10, \
    SplitCIFAR100, SplitCIFAR110, SplitMNIST, RotatedMNIST, PermutedMNIST, SplitCUB200
    
from avalanche.benchmarks.datasets import MNIST, FashionMNIST, KMNIST, EMNIST, \
QMNIST, FakeData, CocoCaptions, CocoDetection, LSUN, ImageNet, CIFAR10, \
CIFAR100, STL10, SVHN, PhotoTour, SBU, Flickr8k, Flickr30k, VOCDetection, \
VOCSegmentation, Cityscapes, SBDataset, USPS, HMDB51, UCF101, \
CelebA, CORe50Dataset, TinyImagenet, CUB200, OpenLORIS


def divide_dataset(dataset, classes):
    data_dict = {class_idx: [] for class_idx in classes}

    for data, label, _ in dataset: 
        data_dict[label].append([data, label])

    return data_dict

def ready_mix_data(data_dict, fraction=0.5):
    '''
        data_dict : divided blurry dataset.
        fraction : M blurry level.
    '''
    mixed_dict = {}
    for label, data_list in data_dict.items():
        mixed_data = []
        origin_data = []
        num_samples = int(len(data_list) * fraction)
        
        indices = random.sample(range(len(data_list)), num_samples)
        mixed_samples = [data_list[i] for i in indices]
        origin_samples = [data_list[i] for i in range(len(data_list)) if i not in indices]
        
        origin_data.extend(origin_samples)
        mixed_data.extend(mixed_samples)
        
        random.shuffle(origin_data)
        random.shuffle(mixed_data)
        
        mixed_dict[label] = mixed_data # New mixed dataset config
        data_dict[label] = origin_data # Overwrite dataset config
        

    return data_dict, mixed_dict

def random_weights(num_weights):
    weights = np.random.random(num_weights)
    weights /= np.sum(weights)
    return weights

def blurry_data_weight(mixed_dict, blurry_classes):
    weight_dict = {}
    len_keys = len(mixed_dict.keys())
    
    for e_idx, key in enumerate(mixed_dict.keys()):
        weights = random_weights(len_keys - 1) # Becuase one class is itself.
        weights = np.insert(weights, e_idx, 0)
        weight_dict[key] = weights # Specific ratio in each class
    
    for key in weight_dict.keys():
        weight_dict[key] = (len(mixed_dict[key]) * weight_dict[key]).astype(int)
        
    blurry_ratio = []
    for e_idx, bidx in enumerate(blurry_classes):
        blurry_ratio.append([weight_dict[idx][e_idx] for idx in weight_dict.keys() if idx is not bidx])
        
    return blurry_ratio, weight_dict #for real, for checking

def export_blurryset(K, ratio, blurry_dataset_list, last = False):
    '''
        K : class
        ratio : classes ratio in other classes
        blurry_dataset_list : a partion of original blurry dataset for blurry set-up
    '''
    keys = list(blurry_dataset_list.keys())
    if last == False:
        keys.remove(K)  
    usage_list = []
    for e_idx, k_idx in enumerate(keys):
        K_list = blurry_dataset_list[k_idx]
        if last == False :
            count = ratio[e_idx]
            
            # Usage list for current blurry training
            usage_list.extend([K_list[i] for i in range(count)])
            
            # Delete the used indices from K_list
            K_list = [K_list[i] for i in range(len(K_list)) if i not in range(count)]

            # Update the dataset in the blurry_dataset_list
            blurry_dataset_list[k_idx] = K_list
        else :
            usage_list.extend(K_list)
            blurry_dataset_list[k_idx] = []
            
    return blurry_dataset_list, usage_list

class IBlurryDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data, label = self.dataset[index]
        data = torch.tensor(data).float()
        label = torch.tensor(label).long()
        return data, label
    
def ChooseDataset(M_percent:float, N_percent:float):
    '''
        We can call these classes.
        classes = {
            "mnist": 10,
            "KMNIST": 10,
            "EMNIST": 49,
            "FashionMNIST": 10,
            "SVHN": 10,
            "cifar10": 10,
            "cifar100": 100,
            "CINIC10": 10,
            "TinyImagenet": 200,
            "imagenet100": 100,
            "imagenet1000": 1000,
        }

    '''
    # creating the benchmark (scenario object)
    Dataset_caller = SplitCIFAR10(
        n_experiences=5,
        seed=1234,
    )

    # recovering the train and test streams
    train_stream = Dataset_caller.train_stream
    test_stream = Dataset_caller.test_stream

    # iterating over the train stream
    for experience in train_stream:
        print("Start of task ", experience.task_label)
        print('Classes in this task:', experience.classes_in_this_experience) #* 해당 exp에 사용되는 class를 확인할 수 있음. 여기서는 DI와 같은 방식으로 사용됨

        # The current Pytorch training set can be easily recovered through the 
        # experience
        current_training_set = experience.dataset
        # ...as well as the task_label
        print('Task {}'.format(experience.task_label))
        print('This task contains', len(current_training_set), 'training examples')

        # we can recover the corresponding test experience in the test stream
        current_test_set = test_stream[experience.current_experience].dataset #* indexing이 가능한 것을 말하고 있음
        print('This task contains', len(current_test_set), 'test examples')

    N_percent = 0.5 # Disjoint partition
    M_percent = 1 # Blurry level
    #assert (1-N_percent) > N_percent

    dataset = Dataset_caller
    dataset_length = len(dataset.class_mapping)
    reminder = dataset_length % int(1 / N_percent)

    #check divided available
    assert reminder == 0

    count_m = dataset_length * N_percent

    disjointed_classes = random.sample(dataset.classes_order, int(count_m))
    blurry_classes = list(set(dataset.class_mapping) - set(disjointed_classes))
    print(f"disjointed : {disjointed_classes}")
    print(f"blurry : {blurry_classes}")

    original_dataset = np.array(Dataset_caller.original_train_dataset)

    # Get the index of labels in the second column of original_dataset
    labels = original_dataset[:, 1]

    # Create a boolean mask for disjointed_classes
    disjointed_mask = np.isin(labels, disjointed_classes)

    # Create a boolean mask for blurry_classes
    blurry_mask = np.isin(labels, blurry_classes)

    # Extract the disjointed dataset using the disjointed_mask
    disjointed_dataset = original_dataset[disjointed_mask]

    # Extract the blurry dataset using the blurry_mask
    blurry_dataset = original_dataset[blurry_mask]

    # Divide disjointed and blurry setting
    disjointed_div_dataset = divide_dataset(disjointed_dataset, disjointed_classes)
    blurry_divided_dataset = divide_dataset(blurry_dataset, blurry_classes)

    # devide each classes in blurry classes: sum(original_blurry(3500), blurry_data(1500)) = original class counts(5000) -> first divided value.
    origin_blurry_dataset_list, blurry_dataset_list = ready_mix_data(blurry_divided_dataset, M_percent)
    blurry_ratio, checking = blurry_data_weight(blurry_dataset_list, blurry_classes)

    total_dataset = []
    last = False
    check_dict = {}
    batch_size = 32
    shuffle = True

    # I-Blurry Training start
    dataloader_list = []
    for e_idx, (didx, bidx) in enumerate(zip(disjointed_classes, blurry_classes)):
        if e_idx + 1 == len(blurry_classes):
            last = True
        disjoint_dataset = disjointed_div_dataset[didx]
        base_blurry_dataset = origin_blurry_dataset_list[bidx]
        
        blurry_dataset_list, blurry_dataset = export_blurryset(bidx, blurry_ratio[e_idx], blurry_dataset_list, last)
        
        print(f"e_idx : {e_idx}, each classes length : {[len(blurry_dataset_list[b_idx]) for b_idx in blurry_classes]}")
        
        # Dataset config complete
        total_dataset.extend(disjoint_dataset)
        total_dataset.extend(base_blurry_dataset)
        total_dataset.extend(blurry_dataset)
        
        my_dataset = IBlurryDataset(total_dataset)
        my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader_list.append(my_dataloader)
    
    # DataLoader_list has dataloaders as the number of experiences
    return dataloader_list
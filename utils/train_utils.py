"""
rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
import torch_optimizer
from easydict import EasyDict as edict
from torch import optim
import transformers
import fnmatch
from models import mnist, cifar, imagenet
from transformers import AutoImageProcessor, ViTForImageClassification

def select_optimizer(opt_name, lr, model, sched_name="cos"):
    if opt_name == "adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    elif opt_name == "radam":
        opt = torch_optimizer.RAdam(model.parameters(), lr=lr, weight_decay=0.00001)
    elif opt_name == "sgd":
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")

    if sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2, eta_min=lr * 0.01
        )
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, milestones=[30, 60, 80, 90], gamma=0.1
        )
    else:
        raise NotImplementedError(
            "Please select the sched_name [cos, anneal, multistep]"
        )

    return opt, scheduler


def select_model(model_name, dataset, num_classes=None, backbone=None):
    opt = edict(
        {
            "depth": 18,
            "num_classes": num_classes,
            "in_channels": 3,
            "bn": True,
            "normtype": "BatchNorm",
            "activetype": "ReLU",
            "pooltype": "MaxPool2d",
            "preact": False,
            "affine_bn": True,
            "bn_eps": 1e-6,
            "compression": 0.5,
        }
    )

    image_size = None
    patch_size = None
    dataset_name = None
    if "mnist" in dataset:
        dataset_name = "mnist"
        model_class = getattr(mnist, "MLP") if backbone=="basic" else getattr(mnist,"ViT_MLP")
        image_size=28
        patch_size = 4
    elif "cifar" in dataset:
        dataset_name = "cifar"
        model_class = getattr(cifar, "ResNet")
        if "vit" in backbone:
            image_size = 224
            patch_size = 16
            if model_name == "simplevit":
                model_class = getattr(cifar, "SimpleViT_CIFAR")
            elif model_name == "vit_pretrained":
                model_class = getattr(cifar, "PretrainedVit_CIFAR")
            elif model_name == "vit_vanilla":
                model_class = getattr(cifar, "Vit_CIFAR")
            else:
                raise NotImplementedError
        else:
            image_size = 32
            patch_size = 4
    elif "imagenet" in dataset:
        model_class = getattr(imagenet, "ResNet")
        image_size = 224
        patch_size = 16
    else:
        raise NotImplementedError(
            "Please select the appropriate datasets (mnist, cifar10, cifar100, imagenet)"
        )
    
    vit_config = transformers.ViTConfig(
        image_size= image_size,
        patch_size= patch_size,
        num_labels= num_classes
    )
    

    if backbone == "basic":
        if model_name == "resnet18":
            opt["depth"] = 18
        elif model_name == "resnet32":
            opt["depth"] = 32
        elif model_name == "resnet34":
            opt["depth"] = 34
        elif model_name == "mlp400":
            opt["width"] = 400
        else:
            raise NotImplementedError(
                "Please choose the model name in [resnet18, resnet32, resnet34]"
            )
            
    if "pretrained" in model_name:
        model = model_class(vit_config)
        assert isinstance(model, ViTForImageClassification)
        model = model.from_pretrained("google/vit-base-patch16-224")
        model.override_classifier(vit_config,opt)
    elif "vanilla" in model_name:
        model = model_class(vit_config)
        assert isinstance(model, ViTForImageClassification)
        model.override_classifier(vit_config, opt)
    else:
        model = model_class(opt)

    return model

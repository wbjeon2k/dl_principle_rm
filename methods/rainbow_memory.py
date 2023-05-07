"""
rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
import logging
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from methods.finetune import Finetune
from utils.data_loader import cutmix_data, ImageDataset

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class RM(Finetune):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )

        self.batch_size = kwargs["batchsize"]
        self.n_worker = kwargs["n_worker"]
        self.exp_env = kwargs["stream_env"]
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "uncertainty"
            
        #regularization term
        self.params = {
            n: p for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad
        }  # For convenience
        self.regularization_terms = {}
        self.task_count = 0
        self.reg_coef = kwargs["reg_coef"]
        
    #EWC regularization    
    def calculate_importance(self, dataloader):
        # Update the diag fisher information
        # There are several ways to estimate the F matrix.
        # We keep the implementation as simple as possible while maintaining a similar performance to the literature.
        logger.debug("Computing EWC")

        # Initialize the importance matrix
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(0)  # zero initialized

        # Sample a subset (n_fisher_sample) of data to estimate the fisher information (batch_size=1)
        # Otherwise it uses mini-batches for the estimation. This speeds up the process a lot with similar performance.
        if self.n_fisher_sample is not None:
            n_sample = min(self.n_fisher_sample, len(dataloader.dataset))
            logger.info("Sample", self.n_fisher_sample, "for estimating the F matrix.")
            rand_ind = random.sample(list(range(len(dataloader.dataset))), n_sample)
            subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind)
            dataloader = torch.utils.data.DataLoader(
                subdata, shuffle=True, num_workers=2, batch_size=1
            )

        self.model.eval()
        # Accumulate the square of gradients
        for data in dataloader:
            x = data["image"]
            y = data["label"]

            x = x.to(self.device)
            y = y.to(self.device)

            logit = self.model(x)

            pred = torch.argmax(logit, dim=-1)
            if self.empFI:  # Use groundtruth label (default is without this)
                pred = y

            loss = self.criterion(logit, pred)
            reg_loss = self.regularization_loss()
            loss += reg_loss

            self.model.zero_grad()
            loss.backward()

            for n, p in importance.items():
                # Some heads can have no grad if no loss applied on them.
                if self.params[n].grad is not None:
                    p += (self.params[n].grad ** 2) * len(x) / len(dataloader.dataset)

        return importance
    
    #EWC regularization loss
    def regularization_loss(
        self,
    ):
        reg_loss = 0
        if len(self.regularization_terms) > 0:
            # Calculate the reg_loss only when the regularization_terms exists
            for _, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term["importance"]
                task_param = reg_term["task_param"]

                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()

                max_importance = 0
                max_param_change = 0
                for n, p in self.params.items():
                    max_importance = max(max_importance, importance[n].max())
                    max_param_change = max(
                        max_param_change, ((p - task_param[n]) ** 2).max()
                    )
                if reg_loss > 1000:
                    logger.warning(
                        f"max_importance:{max_importance}, max_param_change:{max_param_change}"
                    )
                reg_loss += task_reg_loss
            reg_loss = self.reg_coef * reg_loss

        return reg_loss

    def train(self, cur_iter, n_epoch, batch_size, n_worker, n_passes=0):
        if len(self.memory_list) > 0:
            mem_dataset = ImageDataset(
                pd.DataFrame(self.memory_list),
                dataset=self.dataset,
                transform=self.train_transform,
            )
            memory_loader = DataLoader(
                mem_dataset,
                shuffle=True,
                batch_size=(batch_size // 2),
                num_workers=n_worker,
            )
            stream_batch_size = batch_size - batch_size // 2
        else:
            memory_loader = None
            stream_batch_size = batch_size

        # train_list == streamed_list in RM
        train_list = self.streamed_list
        test_list = self.test_list
        random.shuffle(train_list)
        # Configuring a batch with streamed and memory data equally.
        train_loader, test_loader = self.get_dataloader(
            stream_batch_size, n_worker, train_list, test_list
        )

        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list)+len(self.memory_list)}")
        logger.info(f"Test samples: {len(test_list)}")

        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
        self.model = self.model.to(self.device)
        for epoch in range(n_epoch):
            # initialize for each task
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                self.scheduler.step()

            train_loss, train_acc = self._train(train_loader=train_loader, memory_loader=memory_loader,
                                                optimizer=self.optimizer, criterion=self.criterion)
            eval_dict = self.evaluation(
                test_loader=test_loader, criterion=self.criterion
            )
            writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)
            writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
            writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)
            writer.add_scalar(
                f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )

            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )

            best_acc = max(best_acc, eval_dict["avg_acc"])

        # 2.Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()
            
        # 3.Calculate the importance of weights for current task
        importance = self.calculate_importance(train_loader)
        
        # Save the weight and importance of weights of current task
        self.task_count += 1

        # Use a new slot to store the task-specific information
        if self.online_reg and len(self.regularization_terms) > 0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {
                "importance": importance,
                "task_param": task_param,
            }
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {
                "importance": importance,
                "task_param": task_param,
            }
        logger.debug(f"# of reg_terms: {len(self.regularization_terms)}")
        
        return best_acc, eval_dict

    def update_model(self, x, y, criterion, optimizer):
        optimizer.zero_grad()

        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            logit = self.model(x)
            loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
                logit, labels_b
            )
        else:
            logit = self.model(x)
            loss = criterion(logit, y)
            
        reg_loss = self.regularization_loss()

        loss += reg_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        
        _, preds = logit.topk(self.topk, 1, True, True)
        total_loss += loss.item()
        
        return total_loss, torch.sum(preds == y.unsqueeze(1)).item(), y.size(0)

    def _train(
        self, train_loader, memory_loader, optimizer, criterion
    ):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        self.model.train()
        if memory_loader is not None and train_loader is not None:
            data_iterator = zip(train_loader, cycle(memory_loader))
        elif memory_loader is not None:
            data_iterator = memory_loader
        elif train_loader is not None:
            data_iterator = train_loader
        else:
            raise NotImplementedError("None of dataloder is valid")

        for data in data_iterator:
            if len(data) == 2:
                stream_data, mem_data = data
                x = torch.cat([stream_data["image"], mem_data["image"]])
                y = torch.cat([stream_data["label"], mem_data["label"]])
            else:
                x = data["image"]
                y = data["label"]

            x = x.to(self.device)
            y = y.to(self.device)

            l, c, d = self.update_model(x, y, criterion, optimizer)
            total_loss += l
            correct += c
            num_data += d

        if train_loader is not None:
            n_batches = len(train_loader)
        else:
            n_batches = len(memory_loader)

        return total_loss / n_batches, correct / num_data

    def allocate_batch_size(self, n_old_class, n_new_class):
        new_batch_size = int(
            self.batch_size * n_new_class / (n_old_class + n_new_class)
        )
        old_batch_size = self.batch_size - new_batch_size
        return new_batch_size, old_batch_size

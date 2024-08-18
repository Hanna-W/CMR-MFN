import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.cmr_mfn_net import CMR_MFNNet
from utils.toolkit import count_parameters, tensor2numpy

EPSILON = 1e-8
T = 2


class CMR_MFN(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._batch_size = args["batch_size"]
        self._num_workers = args["workers"]
        self._lr = args["lr"]
        self._epochs = args["epochs"]
        self._momentum = args["momentum"]
        self._weight_decay = args["weight_decay"]
        self._lr_steps = args["lr_steps"]
        self._modality = args["modality"]

        self._freeze = args["freeze"]
        self._clip_gradient = args["clip_gradient"]

        self._network = CMR_MFNNet(args["num_segments"], args["modality"], args["arch"],
                                dropout=args["dropout"], fusion_type=args["fusion_type"])
        
    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_task_size
        self.data_manager = data_manager

        self._network._gen_train_fc(self._cur_task_size * 2)

        # logging.info("All params: {}".format(count_parameters(self._network)))
        # logging.info(
        #     "Trainable params: {}".format(count_parameters(self._network, True))
        # )
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.fusion_networks[i].parameters():
                    p.requires_grad = False

                for p in self._network.fc_list[i].parameters():
                    p.requires_grad = False

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )       
        self.train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
        )
        self._train(self.train_loader, self.test_loader)

    def train(self):
        self._network.train()

        if self._cur_task > 0:
            for i in range(self._cur_task):
                self._network.fusion_networks[i].eval()
                self._network.fc_list[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()),
                                        self._lr,
                                        weight_decay=self._weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self._lr_steps, gamma=0.1)
        if self._cur_task == 0:
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
        self._network.save_parameter()

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)

                features = self._network.feature_extract_network(inputs)
                fake_inputs, fake_targets = self._confusion_mixup(features, targets)
                fusion_features = self._network.fusion_network(fake_inputs)["features"]
                fake_logits = self._network.fc(fusion_features)['logits']
                
                loss_clf = F.cross_entropy(fake_logits, fake_targets)
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(fake_logits, dim=1)
                correct += preds.eq(fake_targets.expand_as(preds)).cpu().sum()
                total += len(fake_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            self.training_iterations += 1

            if epoch % 5 == 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    train_acc,

                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                targets = targets - self._known_classes

                features = self._network.feature_extract_network(inputs)
                fake_inputs, fake_targets = self._confusion_mixup(features, targets)
                fusion_features = self._network.fusion_network(fake_inputs)["features"]
                fake_logits = self._network.fc(fusion_features)['logits']
                
                loss_clf = F.cross_entropy(fake_logits, fake_targets)
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(fake_logits, dim=1)
                correct += preds.eq(fake_targets.expand_as(preds)).cpu().sum()
                total += len(fake_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            self.training_iterations += 1
            if epoch % 5 == 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    train_acc,

                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _eval_cnn(self, loader):
        self._network.fusion_networks.to(self._device)
        self._network.fc_list.to(self._device)
        self._network.eval()
        y_pred, y_true = [], []
        results = []
        for _, (_, inputs, targets) in enumerate(loader):
            for m in self._modality:
                inputs[m] = inputs[m].to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, self._cur_task_size, mode='test')
                logits = outputs["logits"]
            predicts = torch.topk(
                logits, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            results.append({'features': {m: outputs['features'][m].cpu().numpy() for m in self._modality},
                            'fusion_features': outputs['fusion_features'].cpu().numpy(),
                            'logits': logits.cpu().numpy()})

        return np.concatenate(y_pred), np.concatenate(y_true), results  # [N, topk]

    def eval_task(self, scores_dir):
        y_pred, y_true, results = self._eval_cnn(self.test_loader)
        self.save_scores(results, y_true, y_pred, '{}/{}.pkl'.format(scores_dir, self._cur_task))
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def _map_targets(self, select_targets):
        mixup_targets = select_targets + self._cur_task_size
        return mixup_targets

    def _confusion_mixup(self, inputs, targets, alpha=0.2, mix_time=2):
        mixup_inputs = {}
        for m in self._modality:
            mixup_inputs[m] = []
        mixup_targets = []

        for _ in range(mix_time):
            index = torch.randperm(inputs[self._modality[0]].shape[0])
            perm_targets = targets[index]

            mask = perm_targets != targets
            
            for m in self._modality:
                select_inputs = inputs[m][mask]
                perm_inputs = inputs[m][index][mask]
                
                lams = np.random.beta(alpha, alpha, size=sum(mask))
                lams = np.where(lams < 0.5, 0.75, lams)
                lams = torch.from_numpy(lams).cuda(4)[:, None].float()
                
                if len(lams) != 0:
                    mixup_input = torch.cat(
                        [torch.unsqueeze(lams[i] * select_inputs[i] + (1 - lams[i]) * perm_inputs[i], 0) for i in
                         range(len(lams))], 0)
                    
                    mixup_inputs[m].append(mixup_input)

            if len(lams) != 0:
                select_targets = targets[mask]
                perm_targets = perm_targets[mask]
                mixup_targets.append(self._map_targets(select_targets))        

        for m in self._modality:
            if len(mixup_inputs[m]) != 0:
                mixup_inputs[m] = torch.cat(mixup_inputs[m], dim=0)
                inputs[m] = torch.cat([inputs[m], mixup_inputs[m]], dim=0)
        
        if len(mixup_targets) != 0:
            mixup_targets = torch.cat(mixup_targets, dim=0)
            targets = torch.cat([targets, mixup_targets], dim=0)

        return inputs, targets

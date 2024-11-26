import logging
import copy
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import FOSTERNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

EPSILON = 1e-8
batch_size = 64

class PLACEBO_FOSTER(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = FOSTERNet(args["convnet_type"], False)
        self._snet = None
        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        self.per_cls_weights = None
        self.is_teacher_wa = args["is_teacher_wa"]
        self.is_student_wa = args["is_student_wa"]
        self.lambda_okd = args["lambda_okd"]
        self.wa_value = args["wa_value"]
        self.oofc = args["oofc"].lower()

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager, aux_data_manager):
        self.data_manager = data_manager
        self.aux_data_manager = aux_data_manager
        self._cur_task += 1
        if self._cur_task > 1:
            self._network = self._snet
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for p in self._network.convnets[0].parameters():
                p.requires_grad = False
            for p in self._network.oldfc.parameters():
                p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        if self._cur_task >= 1:
            aux_data, _, aux_dataset = aux_data_manager.get_raw_dataset(ret_data=True)            
            self._eval_aux_data(aux_data, aux_dataset, data_manager)
            new_aux_dataset = aux_data_manager.get_selected_dataset(self.aux_data, self.aux_targets)
            self.aux_loader = DataLoader(
                new_aux_dataset,
                batch_size=self.args["batch_size"],
                shuffle=True,
                num_workers=self.args["num_workers"],
                pin_memory=True,
            )
        else:
            self.aux_loader = None

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            inc_mode=True, 
            appendent=self._get_memory(),
        )          
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
            pin_memory=True,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"],
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.aux_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network_module_ptr.train()
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            self._network_module_ptr.convnets[0].eval()

    def _eval_aux_data(self, aux_data, aux_dataset, data_manager):

        num_aux = self.args["num_aux"]
        self._network.to(self._device)
        logging.info("Selecting auxiliary data")

        # Old classes
        logging.info("Computing old class means")
        dummy_data = copy.deepcopy(self._data_memory)
        dummy_targets = copy.deepcopy(self._targets_memory)
        class_means = np.zeros((self._total_classes, self.feature_dim))
        _data_memory, _targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask], dummy_targets[mask]
            _data_memory = (
                np.concatenate((_data_memory, dd))
                if len(_data_memory) != 0
                else dd
            )
            _targets_memory = (
                np.concatenate((_targets_memory, dt))
                if len(_targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            #mean = mean / np.linalg.norm(mean)

            class_means[class_idx, :] = mean

        # New classes
        logging.info("Computing new class means")
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            #mean = mean / np.linalg.norm(mean)

            class_means[class_idx, :] = mean
        

        self.aux_loader = DataLoader(
            aux_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"],
            pin_memory=True,
        )

        vectors, _ = self._extract_vectors(self.aux_loader)

        _data_memory, _targets_memory = np.array([]), np.array([])

        for class_idx in range(self._total_classes):
            # Select

            this_cls_mean = class_means[class_idx]
            old_cls_means = class_means[:self._known_classes]
            new_cls_means = class_means[self._known_classes:]

            selected_aux = []
            for k in range(1, num_aux + 1):
                dis_this_cls = cdist(this_cls_mean.reshape(-1,1).T, vectors, self.args["score_dist_type"])

                dis_old_cls = np.zeros(dis_this_cls.shape)
                for old_class_idx in range(len(old_cls_means)):
                    this_old_cls_mean = old_cls_means[old_class_idx]
                    dis_old_cls += cdist(this_old_cls_mean.reshape(-1,1).T, vectors, self.args["score_dist_type"])
                dis_old_cls = dis_old_cls/len(old_cls_means)

                dis_new_cls = np.zeros(dis_this_cls.shape)
                for new_class_idx in range(len(new_cls_means)):
                    this_new_cls_mean = new_cls_means[new_class_idx]
                    dis_new_cls += cdist(this_new_cls_mean.reshape(-1,1).T, vectors, self.args["score_dist_type"])
                dis_new_cls = dis_new_cls/len(new_cls_means)

                total_dis = dis_this_cls + self.args["score_old"]*dis_old_cls + self.args["score_new"]*dis_new_cls


                i = np.argmin(total_dis)
                selected_aux.append(
                    np.array(aux_data[i])
                )  # New object to avoid passing by inference
                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_aux_samples = np.array(selected_aux)
            aux_targets = np.full(num_aux, class_idx)
            _data_memory = (
                np.concatenate((_data_memory, selected_aux_samples))
                if len(_data_memory) != 0
                else selected_aux_samples
            )
            _targets_memory = (
                np.concatenate((_targets_memory, aux_targets))
                if len(_targets_memory) != 0
                else aux_targets
            )

        self.aux_data = _data_memory
        self.aux_targets = _targets_memory


    def _train(self, train_loader, test_loader, aux_loader):
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["init_epochs"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:

            cls_num_list = [self.samples_old_class] * self._known_classes + [
                self.samples_new_class(i)
                for i in range(self._known_classes, self._total_classes)
            ]

            effective_num = 1.0 - np.power(self.beta1, cls_num_list)
            per_cls_weights = (1.0 - self.beta1) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )

            logging.info("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)

            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args["lr"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["boosting_epochs"]
            )
            if self.oofc == "az":
                for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                    if i == 0:
                        p.data[
                            self._known_classes :, : self._network_module_ptr.out_dim
                        ] = torch.tensor(0.0)
            elif self.oofc != "ft":
                assert 0, "not implemented"
            self._feature_boosting(train_loader, test_loader, aux_loader, optimizer, scheduler)
            if self.is_teacher_wa:
                self._network_module_ptr.weight_align(
                    self._known_classes,
                    self._total_classes - self._known_classes,
                    self.wa_value,
                )
            else:
                logging.info("do not weight align teacher!")

            cls_num_list = [self.samples_old_class] * self._known_classes + [
                self.samples_new_class(i)
                for i in range(self._known_classes, self._total_classes)
            ]
            effective_num = 1.0 - np.power(self.beta2, cls_num_list)
            per_cls_weights = (1.0 - self.beta2) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )
            logging.info("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)
            self._feature_compression(train_loader, test_loader, aux_loader)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)

    def _feature_boosting(self, train_loader, test_loader, aux_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["boosting_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_fe = 0.0
            losses_kd = 0.0
            correct, total = 0, 0

            aux_iter = iter(aux_loader)

            for i, (_, inputs, targets) in enumerate(train_loader):

                try:
                    (_, inputs_aux, targets_aux) = next(aux_iter)
                except:
                    aux_iter = iter(aux_loader)
                    (_, inputs_aux, targets_aux) = next(aux_iter)

                if self.args["aux_boosting"]:
                    inputs_overall = torch.cat((inputs, inputs_aux), 0) 
                else:
                    inputs_overall = inputs
                inputs_overall = inputs_overall.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)

                outputs = self._network(inputs_overall)
                logits, fe_logits, old_logits = (outputs["logits"], outputs["fe_logits"], outputs["old_logits"].detach(),)
                loss_clf = F.cross_entropy(logits[0:len(targets)] / self.per_cls_weights, targets)
                loss_fe = F.cross_entropy(fe_logits[0:len(targets)], targets)
                loss_kd = self.lambda_okd * _KD_loss(logits[:, : self._known_classes], old_logits, self.args["T"])
                loss = loss_clf + loss_fe + loss_kd
                optimizer.zero_grad()
                loss.backward()
                if self.oofc == "az":
                    for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                        if i == 0:
                            p.grad.data[
                                self._known_classes :,
                                : self._network_module_ptr.out_dim,
                            ] = torch.tensor(0.0)
                elif self.oofc != "ft":
                    assert 0, "not implemented"
                optimizer.step()
                losses += loss.item()
                losses_fe += loss_fe.item()
                losses_clf += loss_clf.item()
                losses_kd += (
                    self._known_classes / self._total_classes
                ) * loss_kd.item()
                _, preds = torch.max(logits[0:len(targets)], dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["boosting_epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["boosting_epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)

    def _feature_compression(self, train_loader, test_loader, aux_loader):
        self._snet = FOSTERNet(self.args["convnet_type"], False)
        self._snet.update_fc(self._total_classes)
        if len(self._multiple_gpus) > 1:
            self._snet = nn.DataParallel(self._snet, self._multiple_gpus)
        if hasattr(self._snet, "module"):
            self._snet_module_ptr = self._snet.module
        else:
            self._snet_module_ptr = self._snet
        self._snet.to(self._device)
        self._snet_module_ptr.convnets[0].load_state_dict(
            self._network_module_ptr.convnets[0].state_dict()
        )
        self._snet_module_ptr.copy_fc(self._network_module_ptr.oldfc)
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._snet.parameters()),
            lr=self.args["lr"],
            momentum=0.9,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.args["compression_epochs"]
        )
        self._network.eval()
        prog_bar = tqdm(range(self.args["compression_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._snet.train()
            losses = 0.0
            correct, total = 0, 0

            aux_iter = iter(aux_loader)

            for i, (_, inputs, targets) in enumerate(train_loader):

                try:
                    (_, inputs_aux, targets_aux) = next(aux_iter)
                except:
                    aux_iter = iter(aux_loader)
                    (_, inputs_aux, targets_aux) = next(aux_iter)

                if self.args["aux_compression"]:
                    inputs_overall = torch.cat((inputs, inputs_aux), 0) 
                else:
                    inputs_overall = inputs

                inputs_overall = inputs_overall.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)

                dark_logits = self._snet(inputs_overall)["logits"]
                with torch.no_grad():
                    outputs = self._network(inputs_overall)
                    logits, old_logits, fe_logits = (
                        outputs["logits"],
                        outputs["old_logits"],
                        outputs["fe_logits"],
                    )
                loss_dark = self.BKD(dark_logits, logits, self.args["T"])
                loss = loss_dark
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(dark_logits[: targets.shape[0]], dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._snet, test_loader)
                info = "SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["compression_epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["compression_epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)
        if len(self._multiple_gpus) > 1:
            self._snet = self._snet.module
        if self.is_student_wa:
            self._snet.weight_align(
                self._known_classes,
                self._total_classes - self._known_classes,
                self.wa_value,
            )
        else:
            logging.info("do not weight align student!")

        self._snet.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self._device, non_blocking=True)
            with torch.no_grad():
                outputs = self._snet(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        cnn_accy = self._evaluate(y_pred, y_true)
        logging.info("darknet eval: ")
        logging.info("CNN top1 curve: {}".format(cnn_accy["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_accy["top5"]))

    @property
    def samples_old_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._known_classes

    def samples_new_class(self, index):
        if self.args["dataset"] == "cifar100":
            return 500
        else:
            return self.data_manager.getlen(index)

    def BKD(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        soft = soft * self.per_cls_weights
        soft = soft / soft.sum(1)[:, None]
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

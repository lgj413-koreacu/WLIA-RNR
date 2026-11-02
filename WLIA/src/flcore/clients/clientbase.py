import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data, read_public_data

from flcore.trainmodel.models import *
from flcore.trainmodel.models import BaseHeadSplit, BaseHeadSplit_list
# from flcore.trainmodel.models import BaseHeadSplit

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

import utils.load_flcore


from opacus.validators import ModuleValidator




class Client(object):
    """
    Base class for clients in federated learning.
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        # torch.manual_seed(args.seed)
        self.device = args.device
        self.args = args

        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.id = id  # integer

        if isinstance(args.model, list):
            self.model = BaseHeadSplit_list(args, self.id).to(self.device)
            self.aux_model = BaseHeadSplit_list(args, 0).to(self.device)
        else:
            self.model = copy.deepcopy(args.model)

        if self.args.privacy:
            self.model = ModuleValidator.fix(self.model)


        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.distribution_inference_attack = args.distribution_inference_attack

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

        self.train_data = []
        self.test_data = []

        self.data_dist = args.data_dist[id]
        gt = [element / sum(self.data_dist) for element in self.data_dist]
        self.gt = torch.tensor(gt)

        self.l1 = []
        self.l2 = []
        self.l_inf = []
        self.tv = []
        self.wasserstein = []
        self.kl_div = []


    def load_public_data(self, subset_idx, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        public_data = read_public_data(self.dataset, self.id, subset_idx, is_train=True)
        return DataLoader(public_data, batch_size, drop_last=True, shuffle=False)

    def load_train_data(self, batch_size=None, is_attack=False):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, is_attack=is_attack)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None, is_attack=False):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, is_attack=is_attack)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def calc_consensus(self, public_loader):
        self.model.eval()
        tmp_consensus = []
        tmp_consensus_indices = []
        with torch.no_grad():
            for x, _, indices in public_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                tmp_consensus.append(self.model(x).clone())
                tmp_consensus_indices.append(indices)
        return tmp_consensus, tmp_consensus_indices

    def set_consensus(self, received_consensus):
        self.consensus = received_consensus

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, target in testloaderfull:
                # if type(x) == type([]):
                #     x[0] = x[0].to(self.device)
                # else:
                #     x = x.to(self.device)
                # y = y.to(self.device)

                if type(target) == type([]):
                    y, lens = target
                else:
                    y = target

                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        self.model.cpu()
        torch.cuda.empty_cache()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        # auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        auc = 1

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, target in trainloader:
                # if type(x) == type([]):
                #     x[0] = x[0].to(self.device)
                # else:
                #     x = x.to(self.device)
                # y = y.to(self.device)

                if type(target) == type([]):
                    y, lens = target
                else:
                    y = target

                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        self.model.cpu()
        torch.cuda.empty_cache()
        # self.save_model(self.model, 'model')

        return losses, train_num

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def eval_distribution_inference_attack(self, infer_dist):
        l1 = torch.norm(self.gt - infer_dist, p=1)
        l2 = torch.norm(self.gt - infer_dist, p=2)
        l_inf = torch.norm(self.gt - infer_dist, p=float("inf"))

        # Total Variation distance
        tv = 0.5 * torch.norm(self.gt - infer_dist, p=1)

        # Wasserstein distance (1D discrete case)
        p_sorted, _ = torch.sort(self.gt)
        q_sorted, _ = torch.sort(infer_dist)
        cdf_p = torch.cumsum(p_sorted, dim=0)
        cdf_q = torch.cumsum(q_sorted, dim=0)
        wasserstein = torch.sum(torch.abs(cdf_p - cdf_q))

        eps = 1e-12
        kl_div = F.kl_div((infer_dist + eps).log(), self.gt, reduction="sum")

        self.l1.append(l1.item())
        self.l2.append(l2.item())
        self.l_inf.append(l_inf.item())
        self.tv.append(tv.item())
        self.wasserstein.append(wasserstein.item())
        self.kl_div.append(kl_div.item())

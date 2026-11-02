import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.random_weights = None

        self.l1_rw = []
        self.l2_rw = []
        self.l_inf_rw = []
        self.tv_rw = []
        self.wasserstein_rw = []

    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        self.random_weights = self.random_weights.to(
            self.device) if self.args.random_weights is not None else self.random_weights

        # differential privacy
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, target) in enumerate(trainloader):
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

                lens = (x != 0).sum(dim=1)
                lens = torch.clamp(lens, min=1).cpu()

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)

                ## ========== random_weight_module ================
                if self.args.random_weights is not None:
                    if hasattr(self.model, 'head'):
                        fc_weight_norm = torch.norm(self.model.head.weight, dim=1).unsqueeze(0)
                    else:
                        fc_weight_norm = torch.norm(self.model.fc.weight, dim=1).unsqueeze(0)

                    fc_weight_norm = fc_weight_norm.view(-1)
                    if self.args.random_weights_normalization:
                        fc_weight = fc_weight_norm / fc_weight_norm.sum()
                        random_weights = self.random_weights / self.random_weights.sum()
                    else:
                        fc_weight = fc_weight_norm
                        random_weights = self.random_weights

                    rw_regularizer = torch.norm(random_weights - fc_weight, p=2)
                    loss += 0.5 * self.args.lamda * rw_regularizer

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.model.cpu()

        ## ========== random_weight_module ================
        self.random_weights = self.random_weights.cpu() \
            if self.args.random_weights is not None else self.random_weights

        torch.cuda.empty_cache()

        # compute correlation btw fc_weight_norm and class distribution
        if self.distribution_inference_attack:
            if hasattr(self.model, 'head'):
                fc_weight_norm_final = torch.norm(self.model.head.weight, dim=1).view(-1).detach()
            else:
                fc_weight_norm_final = torch.norm(self.model.fc.weight, dim=1).view(-1).detach()

            attack_result = fc_weight_norm_final / sum(fc_weight_norm_final)

            if self.args.random_weights is not None:
                random_weights_dist = self.random_weights / self.random_weights.sum()
                self.eval_random_weight(attack_result, random_weights_dist)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)


    def eval_random_weight(self, infer_dist, rw_dist):
        l1 = torch.norm(rw_dist - infer_dist, p=1)
        l2 = torch.norm(rw_dist - infer_dist, p=2)
        l_inf = torch.norm(rw_dist - infer_dist, p=float("inf"))

        # Total Variation distance
        tv = 0.5 * torch.norm(rw_dist - infer_dist, p=1)

        # Wasserstein distance (1D discrete case)
        p_sorted, _ = torch.sort(rw_dist)
        q_sorted, _ = torch.sort(infer_dist)
        cdf_p = torch.cumsum(p_sorted, dim=0)
        cdf_q = torch.cumsum(q_sorted, dim=0)
        wasserstein = torch.sum(torch.abs(cdf_p - cdf_q))

        self.l1_rw.append(l1.item())
        self.l2_rw.append(l2.item())
        self.l_inf_rw.append(l_inf.item())
        self.tv_rw.append(tv.item())
        self.wasserstein_rw.append(wasserstein.item())



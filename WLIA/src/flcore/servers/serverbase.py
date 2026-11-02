from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import math
import numpy as np
import csv
import h5py
import copy
import time
import random

from torch._C import device
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data, read_public_data, read_aux_data, read_data
from utils.dlg import DLG
from utils.privacy import PGD_attack, hc_noise_attack


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.round_cnt = 0
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate

        if isinstance(args.model, list):
            pass
        else:
            self.global_model = copy.deepcopy(args.model)

        self.fine_tune = False

        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.batches_scores = []
        self.batches_indices = []
        self.client_fc_norm = {}
        self.consensus = []
        self.consensus_indices = []
        self.subset_idx = []

        self.uploaded_weights = []
        self.uploaded_weights_cw = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.rs_cw_train_loss = []
        self.rs_attack = defaultdict(list)

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        self.diverge = False

        self.fl_log = []
        self.attack_result_log = []

        self.fl_log_name = (f'{self.args.dataset}_{self.args.model_family}_{self.args.algorithm}_'
                            f'nc_{self.args.num_clients}_lbs_{self.args.local_epochs}_'
                            f'lr_{self.args.local_learning_rate}_gr_{self.args.global_rounds}_'
                            f'ls_{self.args.local_epochs}_seed_{self.args.seed}_fl_log.pt')

        if self.args.random_weights is not None:
            self.fl_log_name = (f'{self.args.dataset}_{self.args.model_family}_{self.args.algorithm}_'
                                f'nc_{self.args.num_clients}_lbs_{self.args.local_epochs}_'
                                f'lr_{self.args.local_learning_rate}_gr_{self.args.global_rounds}_'
                                f'ls_{self.args.local_epochs}_seed_{self.args.seed}_rw_{self.args.random_weights}_fl_log.pt')


    def set_clients(self, clientObj):
        c_ids = []
        c_models = []
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)

            c_ids.append(client.id)
            c_models.append(client.model)

        if self.args.distribution_inference_attack:
            self.fl_log.append({'ids': c_ids, 'models': c_models, 'global': self.global_model})

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = \
            np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def receive_consensus(self):
        def load_public_data(subset_idx, batch_size=None):
            if batch_size == None:
                batch_size = self.batch_size
                batch_size = 128
            public_data = read_public_data(self.dataset, subset_idx, is_train=True)
            return DataLoader(public_data, batch_size, drop_last=True, shuffle=False)

        self.batches_scores = []
        self.batches_indices = []
        self.subset_idx = random.sample(range(50000), 5000)  # 55,000: MNIST, 50,000: Cifar 10/100
        self.public_loader = load_public_data(self.subset_idx)
        for c in self.selected_clients:
            tmp_consensus, tmp_consensus_indices = c.calc_consensus(self.public_loader)
            self.batches_scores.append(tmp_consensus)
            self.batches_indices.append(tmp_consensus_indices)

    def aggregate_consensus(self):
        self.consensus = []
        for scores in zip(*self.batches_scores):
            self.consensus.append(torch.stack(scores, dim=-1).mean(dim=-1).cpu())
        print('debug')

    def send_consensus(self):
        for c in self.selected_clients:
            c.set_consensus(self.consensus)

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        if self.diverge:
            print("\nDiverge")
        else:
            if self.args.distribution_inference_attack:
                data_to_add = {
                    "exp_name": self.args.goal,
                }

                print('\nLabel Distribution Inference Attack Result')
                for metric, log in self.rs_attack.items():
                    min_value = min(log)
                    min_index = log.index(min_value)
                    max_value = max(log)
                    max_index = log.index(max_value)
                    print(
                        f"{metric}: {min_value:.3f} (round: {(min_index * self.args.attack_gap) + self.args.attack_gap}), "
                        f"max: {max_value:.3f} (round: {(max_index * self.args.attack_gap) + self.args.attack_gap})")

                    if len(log) > 1:
                        data_to_add[metric] = round(min_value, 3)
                    else:
                        data_to_add[metric] = None

                if self.rs_test_acc:
                    ############## add to .h5 file
                    file_path = self.goal + "_result.h5"
                    print("\nFile path: " + file_path)

                    with h5py.File(file_path, 'w') as hf:
                        hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                        hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                        hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

                    data_to_add["accuracy"] = max(self.rs_test_acc)
                    data_to_add["avg_time"] = sum(self.Budget[1:]) / len(self.Budget[1:])

                    print(f"\nBest accuracy: {data_to_add['accuracy']:.4f} (Round: {np.argmax(self.rs_test_acc)})")
                    print(f"Average Process time: {data_to_add['avg_time']:.4f}")
            else:
                ############## add to .h5 file
                file_path = self.goal + "_result.h5"
                print("\nFile path: " + file_path)

                with h5py.File(file_path, 'w') as hf:
                    hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                    hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                    hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

                data_to_add = {
                    "exp_name": self.args.goal,
                    "accuracy": max(self.rs_test_acc),
                    "avg_time": sum(self.Budget[1:]) / len(self.Budget[1:]),
                }

                print(f"\nBest accuracy: {data_to_add['accuracy']:.4f} (Round: {np.argmax(self.rs_test_acc)})")
                print(f"Average Process time: {data_to_add['avg_time']:.4f}")

    def save_proto(self):
        if not self.diverge:
            tmp_protos = {}
            tmp_protos['global'] = self.global_protos
            for client in self.selected_clients:
                tmp_protos[f'local_{client.id}'] = client.protos

            tmp_name = self.args.goal + f'_result_prototype'
            torch.save(tmp_protos, os.path.join(tmp_name + ".pt"))

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        self.round_cnt += 1

        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])

        cw_train_loss = [a / b if b != 0 else None for a, b in zip(stats_train[2], stats_train[1])]
        self.rs_cw_train_loss.append(cw_train_loss)

        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print(f"Averaged Test Accuracy: {test_acc:.4f}(Max: {max(self.rs_test_acc):.4f} | round: {np.argmax(self.rs_test_acc)})")
        print("Std Test Accuracy: {:.4f}\n".format(np.std(accs)))

        if math.isnan(train_loss):
            self.diverge = True

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')


    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=False,
                               send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc

    def load_aux_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_aux_data(self.dataset)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=False)


    def label_distribution_inference_attack(self, round):
        attack_result_all = {}

        if "WLIA" in self.args.label_dist_attack:
            for c in self.selected_clients:
                attack_id = c.id
                tgt_log = self.fl_log[round]
                pos_idx = tgt_log['ids'].index(attack_id)
                tgt_weight = tgt_log['models'][pos_idx]

                ref_log = self.fl_log[round-1]  # difference from previous model
                ref_weight = ref_log['global']

                if self.args.algorithm in ('LG_FedAvg', 'FedClassAvg', 'CHFL'):
                    diff = tgt_weight.weight.detach() - ref_weight.weight.detach()
                else:
                    if hasattr(tgt_weight, 'head'):
                        diff = tgt_weight.head.weight.detach() - ref_weight.head.weight.detach()
                    else:
                        diff = tgt_weight.fc.weight.detach() - ref_weight.fc.weight.detach()

                ##### use only positive diff
                mask = diff > 0
                selected_diff = diff * mask
                row_norm = torch.norm(selected_diff, dim=1)

                refine_row_norm = row_norm - row_norm.min()
                attack_result = refine_row_norm / refine_row_norm.sum()
                attack_result_all[attack_id] = attack_result

        elif "ldpm" in self.args.label_dist_attack:
            # reproduce code of the paper named Protecting Label Distribution in Cross Silo Federated Learning
            auxloader = self.load_aux_data()
            agg_local_model = {}
            g_weight = self.fl_log[round]['global']

            # step 1: local model aggregation
            for client in self.selected_clients:
                attack_id = client.id
                tmp_model = []
                agg_weight_list = []
                for num_round, tgt_log in enumerate(self.fl_log[1:]): # except for initial weight
                    pos_idx = tgt_log['ids'].index(attack_id)
                    tgt_weight = tgt_log['models'][pos_idx]
                    tmp_model.append(tgt_weight)

                    # Compute L2 distance between local and global parameters
                    agg_weight = sum(
                        torch.norm(c - g).item() for c, g in zip(tgt_weight.parameters(), g_weight.parameters()))
                    agg_weight_list.append(agg_weight)

                agg_weight_tensor = torch.tensor(agg_weight_list, dtype=torch.float32)
                agg_weight_soft = F.softmax(agg_weight_tensor, dim=0)

                agg_tgt_model = copy.deepcopy(tmp_model[0])
                for param in agg_tgt_model.parameters():
                    param.data.zero_()

                for tgt_param, w in zip(tmp_model, agg_weight_soft):
                    for param, source_param in zip(agg_tgt_model.parameters(), tgt_param.parameters()):
                        param.data += source_param.data.clone() * w

                agg_local_model[attack_id] = agg_tgt_model

            # step 2: Allocate auxiliary data
            all_output_g = []
            all_output_c = defaultdict(list)

            with torch.no_grad():
                for x, y in auxloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    g_weight.to(self.device)
                    g_weight.eval()
                    output_g = g_weight(x)
                    all_output_g.append(F.softmax(output_g, dim=1).detach().cpu())

                    for c_id, agg_model in agg_local_model.items():
                        agg_model.to(self.device)
                        agg_model.eval()

                        output_c = agg_model(x)
                        all_output_c[c_id].append(F.softmax(output_c, dim=1).detach().cpu())

                        agg_model.cpu()
                    g_weight.cpu()

            all_output_g_cat = torch.cat(all_output_g, dim=0)
            all_output_dist = {}
            for tmp_id, soft_output in all_output_c.items():
                tmp_c_cat = torch.cat(soft_output, dim=0)
                tmp_l2_dist = torch.norm(tmp_c_cat - all_output_g_cat, dim=1)
                all_output_dist[tmp_id] = 1.0 / (tmp_l2_dist + 1e-12)

            inv_dist_sum = sum(all_output_dist.values())
            for sub_id, inv_dist in all_output_dist.items():
                p_i = inv_dist / inv_dist_sum
                rand_threshold = torch.rand(p_i.size())
                tmp_subset = torch.nonzero(p_i > rand_threshold, as_tuple=True)[0].tolist()
                if len(tmp_subset) == 0:
                    attack_result_all[sub_id] = torch.zeros(self.args.num_classes)
                else:
                    tmp_l_value = torch.cat(all_output_c[sub_id], dim=0)
                    tmp_l_value_selected = tmp_l_value[tmp_subset]

                    argmax_labels = torch.argmax(tmp_l_value_selected, dim=1)
                    counts = torch.bincount(argmax_labels, minlength=self.args.num_classes)
                    attack_result_all[sub_id] = counts.float() / counts.sum()
                    if "hard" in self.args.label_dist_attack:
                        # attack by argmax(hard) label
                        argmax_labels = torch.argmax(tmp_l_value_selected, dim=1)
                        counts = torch.bincount(argmax_labels, minlength=self.args.num_classes)
                        attack_result_all[sub_id] = counts.float() / counts.sum()
                    else:
                        # attack by normalized soft-label
                        tmp_l_value_sum = tmp_l_value_selected.sum(dim=0)
                        attack_result_all[sub_id] = tmp_l_value_sum / tmp_l_value_sum.sum()

        elif "pgda" in self.args.label_dist_attack:
            # reproduce code of the paper named On Data Distribution Leakage in Cross-Silo Federated Learning
            agg_local_model = {}
            g_weight = self.fl_log[round]['global']

            fake_sample_size_per_client = int(self.args.fake_sample_size / len(self.selected_clients))

            for client in self.selected_clients:
                attack_id = client.id
                # -------------------------------------------------------------------------
                # Step 1: Get data shape and projection bound
                # -------------------------------------------------------------------------
                raw_data = read_data(self.dataset, attack_id, is_train=False)['x']

                if 'dbpedia' in self.args.dataset:
                    test_data = raw_data
                    if isinstance(test_data[0], list) or isinstance(test_data[0], np.ndarray):
                        seq_len = len(test_data[0])
                        sample_shape = (seq_len,)
                    else:
                        sample_shape = test_data[0].shape

                    r_min, r_max = None, None
                else:
                    test_data = raw_data
                    example = test_data[0]

                    if isinstance(example, np.ndarray):
                        sample_shape = example.shape
                    else:
                        sample_shape = (test_data.shape[1],)

                    data_sample = test_data
                    cliped_test_data = np.clip(data_sample, -3, 3)
                    r_min = np.min(cliped_test_data)
                    r_max = np.max(cliped_test_data)

                # -------------------------------------------------------------------------
                # Step 2: Local model aggregation (Historical Snapshots)
                # -------------------------------------------------------------------------
                tmp_model = []
                agg_weight_list = []
                for num_round, tgt_log in enumerate(self.fl_log[1:]): # except for initial weight
                    pos_idx = tgt_log['ids'].index(attack_id)
                    tgt_weight = tgt_log['models'][pos_idx]
                    tmp_model.append(tgt_weight)

                    # Compute L2 distance between local and global parameters
                    agg_weight = sum(
                        torch.norm(c - g).item() for c, g in zip(tgt_weight.parameters(), g_weight.parameters()))
                    agg_weight_list.append(agg_weight)

                agg_weight_tensor = torch.tensor(agg_weight_list, dtype=torch.float32)
                agg_weight_soft = F.softmax(agg_weight_tensor, dim=0)

                agg_tgt_model = copy.deepcopy(tmp_model[0])
                for param in agg_tgt_model.parameters():
                    param.data.zero_()

                for tgt_param, w in zip(tmp_model, agg_weight_soft):
                    for param, source_param in zip(agg_tgt_model.parameters(), tgt_param.parameters()):
                        param.data += source_param.data.clone() * w

                agg_local_model[attack_id] = agg_tgt_model

                # -------------------------------------------------------------------------
                # Step 3: Generate synthetic sample setup
                # -------------------------------------------------------------------------
                logit_sample_list = []

                # default setting from the original paper
                var_factor = 0.1
                lr=0.1
                pgd_steps = 25
                eps = 0.5

                if 'dbpedia' in self.args.dataset:
                    agg_tgt_model.to(self.device)
                    with torch.no_grad():
                        if hasattr(agg_tgt_model, 'embedding'):
                            norms = agg_tgt_model.base.embedding.weight.norm(p=2, dim=1)
                            MAX_EMB_NORM = norms.mean() + 3 * norms.std() # average + 3*std
                        else:
                            MAX_EMB_NORM = 25.0  # Fallback value for FastText

                # input_perturb_sample = 100
                input_perturb_sample = 50
                var_random_noise_std = 0.003

                g_weight.to(self.device)
                agg_tgt_model.to(self.device)

                g_weight.eval()
                agg_tgt_model.eval()

                while len(logit_sample_list) < fake_sample_size_per_client:
                    # generate random samples
                    if 'dbpedia' in self.args.dataset:
                        seq_len = test_data.shape[1]
                        input_ids = torch.randint(
                            low=0,
                            high=agg_tgt_model.base.embedding.num_embeddings,
                            size=(1, seq_len),
                            device=self.device
                        )
                        emb = agg_tgt_model.base.embedding(input_ids)
                        emb = emb.detach().requires_grad_()

                        optimizer = torch.optim.Adam([emb], lr=lr)
                        # emb_orig = emb.detach().clone()

                        for _ in range(pgd_steps):
                            yg_feat = g_weight.base.forward_from_embedding(emb)
                            yg_logit = g_weight.head(yg_feat)
                            yg = F.softmax(yg_logit, dim=1)

                            yc_feat = agg_tgt_model.base.forward_from_embedding(emb)
                            yc_logit = agg_tgt_model.head(yg_feat)
                            yc = F.softmax(yc_logit, dim=1)

                            loss = F.mse_loss(yg, yc) + var_factor * emb.var()

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            # 3. Projection (L∞-ball)
                            with torch.no_grad():
                                current_norm = emb.norm(p=2, dim=1, keepdim=True)
                                clip_coef = torch.clamp(MAX_EMB_NORM / (current_norm + 1e-6), max=1.0)
                                emb.mul_(clip_coef)

                        ##### 4. Stability Verification (Batch Processing)
                        with torch.no_grad():
                            emb_result = emb.detach()  # [1, Seq_Len, Dim]

                            emb_expanded = emb_result.repeat(input_perturb_sample, 1, 1)

                            noise = torch.randn_like(emb_expanded) * var_random_noise_std
                            emb_hat_batch = emb_expanded + noise

                            curr_norm_batch = emb_hat_batch.norm(p=2, dim=2, keepdim=True)  # dim 주의
                            clip_coef_batch = torch.clamp(MAX_EMB_NORM / (curr_norm_batch + 1e-6), max=1.0)
                            emb_hat_batch.mul_(clip_coef_batch)

                            f_local_batch_feat = agg_tgt_model.base.forward_from_embedding(emb_hat_batch)
                            f_local_batch_logit = agg_tgt_model.head(f_local_batch_feat)
                            f_local_batch = F.softmax(f_local_batch_logit, dim=1)

                            f_global_batch_feat = g_weight.base.forward_from_embedding(emb_hat_batch)
                            f_global_batch_logit = g_weight.head(f_global_batch_feat)
                            f_global_batch = F.softmax(f_global_batch_logit, dim=1)

                            diff_tensor = F.mse_loss(f_local_batch, f_global_batch, reduction='none').mean(
                                dim=1)
                            diff_list = diff_tensor.cpu().numpy()

                            delta = np.mean(diff_list)
                            delta_tau = np.median(diff_list)

                            if delta < delta_tau:
                                delta_feat = agg_tgt_model.base.forward_from_embedding(emb_result)
                                delta_logit = agg_tgt_model.head(delta_feat)
                                logit = F.softmax(delta_logit, dim=1).squeeze(0).cpu()
                                logit_sample_list.append(logit)

                    else:
                        x_np = np.random.uniform(r_min, r_max, size=sample_shape).astype(np.float32)
                        x = torch.tensor(x_np, device=self.device)
                        x = x.unsqueeze(0).detach().requires_grad_()

                        opt = torch.optim.Adam([x], lr=lr)

                        for _ in range(pgd_steps):
                            # algorithm line 5-6
                            yg = F.softmax(g_weight(x).squeeze(0), dim=0)
                            yc = F.softmax(agg_tgt_model(x).squeeze(0), dim=0)

                            loss = torch.nn.MSELoss()(yg, yc) + var_factor * x.var()

                            opt.zero_grad()
                            loss.backward()
                            opt.step()

                            # projection | # algorithm line 7
                            with torch.no_grad():
                                x.clamp_(r_min, r_max)

                        with torch.no_grad():
                            x_result = x.detach()

                        #### 4. Stability Verification (Batch Processing)
                        with torch.no_grad():
                            x_result = x.detach()

                            if x_result.dim() == 4:
                                x_expanded = x_result.repeat(input_perturb_sample, 1, 1, 1)
                            else:
                                x_expanded = x_result.repeat(input_perturb_sample, 1)

                            noise = torch.randn_like(x_expanded) * var_random_noise_std
                            x_hat_batch = (x_expanded + noise).clamp(r_min, r_max)

                            f_local_batch = F.softmax(agg_tgt_model(x_hat_batch), dim=1)
                            f_global_batch = F.softmax(g_weight(x_hat_batch), dim=1)

                            diff_tensor = F.mse_loss(f_local_batch, f_global_batch, reduction='none').mean(dim=1)
                            diff_list = diff_tensor.cpu().numpy()

                            delta = np.mean(diff_list)
                            delta_tau = np.median(diff_list)

                            if delta < delta_tau:
                                sample_logit = F.softmax(agg_tgt_model(x_result).squeeze(0), dim=0).cpu().detach()
                                logit_sample_list.append(sample_logit)

                # step 4: estimate label distribution
                tot_attack_logit = (torch.stack(logit_sample_list, dim=0)).sum(dim=0)
                attack_result = tot_attack_logit / tot_attack_logit.sum()

                attack_result_all[attack_id] = attack_result

                g_weight.cpu()

        elif "decaf" in self.args.label_dist_attack:
            # reproduce code of the paper named Decaf
            auxloader = self.load_aux_data()

            for client in self.selected_clients:
                attack_id = client.id

                tgt_log = self.fl_log[round]
                ref_log = self.fl_log[round-1]

                pos_idx = tgt_log['ids'].index(attack_id)
                tgt_weight = tgt_log['models'][pos_idx]
                ref_weight = ref_log['global']

                if self.args.algorithm in ('LG_FedAvg', 'FedClassAvg', 'CHFL'):
                    tgt_weight_layer = tgt_weight.weight.data
                    ref_weight_layer = ref_weight.weight.data
                else:
                    if hasattr(tgt_weight, 'head'):
                        tgt_weight_layer = tgt_weight.head.weight.data
                        ref_weight_layer = ref_weight.head.weight.data
                    else:
                        tgt_weight_layer = tgt_weight.fc.weight.data
                        ref_weight_layer = ref_weight.fc.weight.data

                grad_diff = (tgt_weight_layer - ref_weight_layer) / self.learning_rate

                # step 2. Null classes removal
                selected_classes = [1] * self.num_classes
                for cls in range(self.num_classes):
                    if grad_diff[cls].max().item() == 0.0:
                        selected_classes[cls] = 0

                # step 3. Gradient Bases Construction
                grad_base = {}
                grad_base_lr = self.args.grad_base_lr
                for cls, flag in enumerate(selected_classes):
                    if flag == 0:
                        continue

                    ref_g_model = copy.deepcopy(ref_weight)
                    ref_g_model.to(self.device)
                    ref_g_model.train()

                    ref_g_loss = nn.CrossEntropyLoss()
                    ref_g_optimizer = torch.optim.SGD(ref_g_model.parameters(), lr=grad_base_lr)

                    for i, (x, y) in enumerate(auxloader):
                        mask = (y == cls)
                        if mask.sum() == 0:
                            continue

                        if type(x) == type([]):
                            x = [t[mask] for t in x]
                        else:
                            x = x[mask]
                        y = y[mask]

                        if isinstance(x, list):
                            x[0] = x[0].to(self.device)
                        else:
                            x = x.to(self.device)
                        y = y.to(self.device)

                        # forward & backward
                        output = ref_g_model(x)
                        loss = ref_g_loss(output, y)
                        ref_g_optimizer.zero_grad()
                        loss.backward()
                        ref_g_optimizer.step()

                    ref_g_model.cpu()

                    if hasattr(ref_weight, 'head'):
                        tmp_tgt_layer = ref_g_model.head.weight.data
                    else:
                        tmp_tgt_layer = ref_g_model.head.fc.data

                    tmp_grad_diff = (tmp_tgt_layer - ref_weight_layer) / self.learning_rate

                    grad_base[cls] = tmp_grad_diff

                # calculate unified gradient diff
                ref_g_model = copy.deepcopy(ref_weight)
                ref_g_model.to(self.device)
                ref_g_model.train()

                ref_g_loss = nn.CrossEntropyLoss()
                ref_g_optimizer = torch.optim.SGD(ref_g_model.parameters(), lr=self.learning_rate)

                for i, (x, y) in enumerate(auxloader):
                    if isinstance(x, list):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    # forward & backward
                    output = ref_g_model(x)
                    loss = ref_g_loss(output, y)
                    ref_g_optimizer.zero_grad()
                    loss.backward()
                    ref_g_optimizer.step()

                ref_g_model.cpu()

                if hasattr(ref_weight, 'head'):
                    tmp_tgt_layer = ref_g_model.head.weight.data
                else:
                    tmp_tgt_layer = ref_g_model.head.fc.data

                tmp_grad_diff = (tmp_tgt_layer - ref_weight_layer) / self.learning_rate
                grad_base_u = tmp_grad_diff

                # step 4. Remaining Classes Decomposition
                decompose_lr = self.args.decompose_lr
                decompose_epochs = 1000

                eta_c = torch.rand(self.num_classes, requires_grad=True)
                eta_u = torch.rand(1, requires_grad=True)

                optimizer = torch.optim.SGD([eta_c, eta_u], lr=decompose_lr)
                mse_loss = nn.MSELoss()

                for epoch in range(decompose_epochs):
                    optimizer.zero_grad()

                    h = torch.zeros_like(grad_diff)
                    for cls, g_diff_c in grad_base.items():
                        h += eta_c[cls] * g_diff_c
                    h += eta_u * grad_base_u

                    loss = mse_loss(h, grad_diff)

                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        eta_c.clamp_(min=0)
                        eta_u.clamp_(min=0)

                with torch.no_grad():
                    eta_total = eta_c + eta_u
                    mask = torch.tensor(selected_classes, dtype=torch.bool, device=eta_c.device)

                    attack_result = torch.zeros_like(eta_c)
                    attack_result[mask] = eta_total[mask]


                    attack_result = attack_result / attack_result.sum()

                attack_result_all[attack_id] = attack_result

        elif "random" in self.args.label_dist_attack:
            n_iters = 100
            samples_per_iter = 2000
            num_classes = self.args.num_classes

            for c in self.selected_clients:
                attack_id = c.id
                total_counts = np.zeros(num_classes, dtype=np.int64)

                for _ in range(n_iters):
                    rv = np.random.random(num_classes)
                    p = rv / rv.sum()
                    counts = np.random.multinomial(samples_per_iter, p)
                    total_counts += counts

                final_dist = total_counts.astype(np.float64) / total_counts.sum()
                attack_result_all[attack_id] = torch.tensor(final_dist, dtype=torch.float32)

        return attack_result_all


    def eval_attack(self, attack_result):
        eval_result = ["L1", "L2", "L_inf", "TV", "Wasserstein", "kl_div"]

        self.attack_result_log.append(attack_result)

        if self.args.visualization:
            csv_filename = self.args.goal + '_result_file.csv'

            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header_written = False

                for idx in range(self.args.num_clients):
                    tmp_gt_label_dist = self.clients[idx].gt.numpy()
                    tmp_infer_label_dist = attack_result[idx].numpy()

                    if not header_written:
                        num_classes = len(tmp_gt_label_dist)
                        header = ['method'] + [str(i) for i in range(num_classes)]
                        writer.writerow(header)
                        header_written = True

                    formatted_gt_list = [f"{x:.4f}" for x in tmp_gt_label_dist]
                    formatted_atk_list = [f"{x:.4f}" for x in tmp_infer_label_dist]

                    writer.writerow(['True'] + formatted_gt_list)
                    writer.writerow(['Attack'] + formatted_atk_list)

                    formatted_gt_str = np.array2string(tmp_gt_label_dist,
                                                       formatter={'float_kind': lambda x: f"{x:.4f}"})
                    formatted_atk_str = np.array2string(tmp_infer_label_dist,
                                                        formatter={'float_kind': lambda x: f"{x:.4f}"})
                    print(f"Client {idx} Saved.")
                    print(f"True  : {formatted_gt_str}")
                    print(f"Attack: {formatted_atk_str}\n")

        for c in self.selected_clients:
            c.eval_distribution_inference_attack(attack_result[c.id])

        for metric in eval_result:
            tmp_list = {}
            for client in self.selected_clients:
                if metric == 'L1':
                    tmp_list[client.id] = client.l1[-1]
                elif metric == 'L2':
                    tmp_list[client.id] = client.l2[-1]
                elif metric == 'L_inf':
                    tmp_list[client.id] = client.l_inf[-1]
                elif metric == 'TV':
                    tmp_list[client.id] = client.tv[-1]
                elif metric == 'Wasserstein':
                    tmp_list[client.id] = client.wasserstein[-1]
                elif metric == 'kl_div':
                    tmp_list[client.id] = client.kl_div[-1]

            values = list(tmp_list.values())
            avg_value = sum(values) / len(values)

            self.rs_attack[metric].append(avg_value)
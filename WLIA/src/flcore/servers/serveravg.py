from tqdm import tqdm
import time
import torch
import copy
import os
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        if self.args.random_weights is not None:
            self.set_random_weights()

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if self.args.distribution_inference_attack:
                fl_log_path = os.path.join('models', self.fl_log_name)
                if not os.path.exists(fl_log_path):
                    self.send_models()

                    if i % self.eval_gap == 0:
                        print(f"\n-------------Round number: {i}-------------")
                        print("\nEvaluate global model")
                        self.evaluate()
                        if self.diverge:
                            break

                    for client in self.selected_clients:
                        client.train()

                    self.receive_models()
                    if self.dlg_eval and i%self.dlg_gap == 0:
                        self.call_dlg(i)
                    self.aggregate_parameters()

                    if self.args.distribution_inference_attack:
                        # update local and global model log
                        self.fl_log.append({
                            'ids': copy.deepcopy(self.uploaded_ids),
                            'models': [copy.deepcopy(m) for m in self.uploaded_models],
                            'global': copy.deepcopy(self.global_model),
                        })

                        if i > 0 and i % self.args.attack_gap == 0:
                            attack_start = time.time()
                            attack_result = self.label_distribution_inference_attack(round=i+1)
                            attack_time = time.time() - attack_start
                            print(f'attack_time for round {i}: {attack_time}')
                            self.eval_attack(attack_result)

                    self.Budget.append(time.time() - s_t)
                    eta = int(self.Budget[-1] * (self.global_rounds + 1 - i))
                    print('-' * 50, self.Budget[-1], f'eta: {eta // 3600}h {(eta % 3600) // 60}m {eta % 60}s')

                    if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                        break

                else:
                    break
            else:
                self.send_models()

                if i % self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    self.evaluate()
                    if self.diverge:
                        break

                for client in self.selected_clients:
                    client.train()

                self.receive_models()
                if self.dlg_eval and i % self.dlg_gap == 0:
                    self.call_dlg(i)
                self.aggregate_parameters()

                self.Budget.append(time.time() - s_t)
                eta = int(self.Budget[-1] * (self.global_rounds + 1 - i))
                print('-' * 50, self.Budget[-1], f'eta: {eta // 3600}h {(eta % 3600) // 60}m {eta % 60}s')

                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break

        if self.args.distribution_inference_attack:
            if not os.path.exists(fl_log_path):
                pass
            else:
                self.fl_log = torch.load(fl_log_path, weights_only=False)
                for i in tqdm(range(self.global_rounds + 1)):
                    if i > 0 and i % self.args.attack_gap == 0:
                        attack_result = self.label_distribution_inference_attack(round=i + 1)
                        self.eval_attack(attack_result)

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def set_random_weights(self, num_round=None):
        if num_round is not None:
            torch.manual_seed(num_round)
        else:
            torch.manual_seed(0)

        if "uniform" in self.args.random_weights:
            self.random_weights = {
                i: (
                        torch.rand(self.num_classes)
                        * self.args.random_weights_mean
                )
                for i in range(self.num_clients)
            }
        else:
            self.random_weights = {
                i: torch.normal(
                    mean=self.args.random_weights_mean,
                    std=self.args.random_weights_std,
                    size=(self.num_classes,)
                ) for i in range(self.num_clients)}

        for c in self.clients:
            if "server" in self.args.random_weights:
                c.random_weights = self.random_weights[0]
            elif "RNR" in self.args.random_weights:
                c.random_weights = self.random_weights[c.id]


    def eval_rw(self):
        eval_result = ["L1", "L2", "L_inf", "TV", "Wasserstein"]

        for metric in eval_result:
            tmp_list = {}
            for client in self.selected_clients:
                if metric == 'L1':
                    tmp_list[client.id] = client.l1_rw[-1]
                elif metric == 'L2':
                    tmp_list[client.id] = client.l2_rw[-1]
                elif metric == 'L_inf':
                    tmp_list[client.id] = client.l_inf_rw[-1]
                elif metric == 'TV':
                    tmp_list[client.id] = client.tv_rw[-1]
                elif metric == 'Wasserstein':
                    tmp_list[client.id] = client.wasserstein_rw[-1]

            values = list(tmp_list.values())
            avg_value = sum(values) / len(values)

            max_id = max(tmp_list, key=tmp_list.get)
            min_id = min(tmp_list, key=tmp_list.get)

            max_value = tmp_list[max_id]
            min_value = tmp_list[min_id]

            print(f"Average {metric}: {avg_value:.3f}(Max: {max_value:.3f}/id_{max_id} | Min: {min_value:.3f}/id_{min_id})")


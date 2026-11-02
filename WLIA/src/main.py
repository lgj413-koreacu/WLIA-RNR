import torch
import os
import time
import warnings
import numpy as np
import logging
import random

from utils.result_utils import average_data, plotting_trial_result
from utils.mem_utils import MemReporter
from utils.load_models import *
from utils.load_flcore import *
from utils.load_arguments import *
from utils.class_distribution_metric import *

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model_family

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        args.model = load_select_models(args, model_str)

        if isinstance(args.model, list):
            for model in args.model:
                print(model)
        else:
            print(args.model)

        # assign FL framework core
        server = assign_fl_core(args, i)
        server.train()

        time_list.append(time.time() - start)

        if not (args.distribution_inference_attack or args.unlearn or args.membership_inference):
            plotting_trial_result(args)

    avg_time = np.average(time_list)
    minutes, seconds = divmod(avg_time, 60)
    hours, minutes = divmod(minutes, 60)
    print(f"\nAverage time cost: {int(hours)}h {int(minutes)}m {round(seconds, 2)}s.")

    print("All done!")

    if args.device == "cuda" and not torch.cuda.is_available():
        reporter.report()


if __name__ == "__main__":
    total_start = time.time()
    parser = load_arguments()
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        if platform.system() == 'Linux':
            if not torch.cuda.is_available():
                args.device = "cpu"
        else:
            if torch.backends.mps.is_available():
                args.device = "mps"

    DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    file_path = os.path.join(dataset_path, 'config.json')
    with open(file_path, 'r') as file:
        data = json.load(file)
        args.num_classes = data['num_classes']
        args.dataset_type = data['partition']
        data_statistic = data['Size of samples for labels in clients']

    data_dist = np.zeros((args.num_clients, args.num_classes))
    for i in range(args.num_clients):
        client_data_dist_np = np.array(data_statistic[i])
        data_dist[i, client_data_dist_np[:, 0]] = client_data_dist_np[:, 1]

    algo = args.dataset + "_" + args.algorithm
    result_path = "./results/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"Directory '{result_path}' created.")

    algo = algo + "_" + args.goal
    file_path = result_path + "{}_data_distribution.npy".format(algo)

    np.save(file_path, data_dist)
    args.data_dist = data_dist.tolist()

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Data Heterogeneity: {}".format(args.dataset_type))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model_family))
    print("Using device: {}".format(args.device))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))

    if args.algorithm == 'cwFedAvg':
        print("Class-wise aggregation: {}".format(args.add_cw))
        print("True class distribution based aggregation: {}".format(args.use_true_dist))
        print("Use weight regularizer: {}".format(args.add_wdr))
        print("Weigh regularizer decay: {}".format(args.weight_decay))

    if args.algorithm == "ProtoNorm":
        print(f'prototype_alignment: {args.prototype_alignment}')
        print(f'prototype_upscaling: {args.prototype_upscaling}')
        if args.prototype_upscaling:
            if args.constant_upscale:
                print(f'Constant Upscale: {args.constant_upscale}')
            elif args.trainable_upscale:
                print(f'Trainable Upscale: {args.trainable_upscale}')
            else:  # fixed scale factor
                print(f'Fixed Upscale: {args.fixed_upscale}')

        print(f'Thomson Iteration Number: {args.thomson_iteration_number}')
        print(f'Thomson Iteration Number: {args.thomson_iteration_number}')
        print(f'Thomson Momentum: {args.thomson_iteration_number}')
        print(f'Thomson Learning Rate: {args.thomson_momentum}')

    if args.algorithm == 'FedProto' or args.algorithm == 'PFL' or args.algorithm == 'TinyFP' or args.algorithm == 'TinyFT':
        print("prototype MSE weight: {}".format(args.lamda))
        print("TGP: margin threshold: {}".format(args.margin_threthold))
        print("TGP: server epoch: {}".format(args.server_epochs))
        print("Class-Wise Structured Sparsity Regularizer: {}".format(args.add_csr))
        print("Number of CSR: {}".format(args.csr_ratio))
        print("Privacy-Preserving Prototype Distillation: {}".format(args.add_ppa))
        print("Constant Scaling Factor: {}".format(args.constant_scale_factor))
    print("=" * 50)

    print('Class Distribution')
    print(data_dist.astype(int))

    print("=" * 50)

    args_dict = vars(args)
    file_name = args.goal + '_args.json'
    with open(file_name, "w") as f:
        json.dump(args_dict, f, indent=4)

    run(args)
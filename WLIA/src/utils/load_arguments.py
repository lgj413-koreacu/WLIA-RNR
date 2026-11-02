import argparse

def load_arguments():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-seed', "--seed", type=int, default=1)

    parser.add_argument('-go', "--goal", type=str, default="test")
    parser.add_argument('-dev', "--device", type=str, default="cuda")
    parser.add_argument('-did', "--device_id", type=str, default="0")

    parser.add_argument('-data', "--dataset", type=str, default="Cifar10")

    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model_family", type=str, default="resnet18")

    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005)
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)

    parser.add_argument('-gr', "--global_rounds", type=int, default=20)
    parser.add_argument('-ls', "--local_epochs", type=int, default=10)

    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")

    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0, help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False, help="Random ratio of clients per round")

    parser.add_argument('-nc', "--num_clients", type=int, default=10)

    parser.add_argument('-pv', "--prev", type=int, default=0, help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1, help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1, help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='temp')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=98635)
    parser.add_argument('-ml', "--max_len", type=int, default=200)

    parser.add_argument('-dp', "--privacy", action='store_true')
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.5)

    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)

    parser.add_argument('-dbg', "--debug", action='store_true', help='for debug')
    parser.add_argument('-vis', "--visualization", action='store_true', help='for visualization')
    parser.add_argument('-vis_hyper', "--hypersphere_visualization", action='store_true', help='for visualization')

    ''' ----------------------- WLIA+RNR -----------------------'''
    parser.add_argument('-dia', "--distribution_inference_attack", action='store_true')

    parser.add_argument('-ag', "--attack_gap", type=int, default=5, help="Rounds gap for evaluation")

    parser.add_argument('-lda', "--label_dist_attack", type=str, default=None)

    parser.add_argument('-pgds', "--fake_sample_size", type=int, default=1000)
    parser.add_argument('-dlr1', "--grad_base_lr", type=float, default=1e-3)
    parser.add_argument('-dlr2', "--decompose_lr", type=float, default=1e-1)

    parser.add_argument('-rw', "--random_weights", type=str, default=None)
    parser.add_argument('-rwn', "--random_weights_normalization", action='store_true')
    parser.add_argument('-rwm', "--random_weights_mean", type=float, default=3.0)
    parser.add_argument('-rws', "--random_weights_std", type=float, default=1.0)

    parser.add_argument('-lam', "--lamda", type=float, default=1.0)


    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    return parser
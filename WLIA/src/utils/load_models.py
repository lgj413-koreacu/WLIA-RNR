import copy
import torch
import platform
import argparse
import os
import json
import time
import warnings
import numpy as np
import torchvision
import logging

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *
from flcore.trainmodel.mobilevit import *

# from station.pfl.stand_alone_test import model_name

vocab_size = 98635  # 98635 for AG_News and 399198 for Sogou_News
max_len = 200
emb_dim = 32


def load_select_models(args, model_str):
    in_feat = 3
    if 'mnist' in args.dataset or 'MNIST' in args.dataset:
        in_feat = 1
        dim = 1024
    elif 'Cifar' in args.dataset or 'GTSRB' in args.dataset:
        dim = 1600
    elif 'fer2013' in args.dataset:
        dim = 5184
    elif 'skin' in args.dataset:
        dim = 1024
    elif 'dbpedia' in args.dataset:
        vocab_size = 30522
        max_len = 300
    else:
        dim = 10816

    # Generate args.model
    if model_str == "mlr":  # convex
        if "mnist" in args.dataset:
            args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes)
        elif "Cifar10" in args.dataset:
            args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes)
        else:
            args.model = Mclr_Logistic(60, num_classes=args.num_classes)

    elif model_str == "cnn":  # non-convex
        # args.model = FedAvgCNN(in_features=in_feat, num_classes=args.num_classes, dim=dim).to(args.device)
        args.model = FedAvgCNN(in_features=in_feat, num_classes=args.num_classes, dim=dim)
        if "omniglot" in args.dataset:
            args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
        elif "Digit5" in args.dataset:
            args.model = Digit5CNN().to(args.device)

    elif model_str == "cnn_toy":  # non-convex
        # args.model = FedAvgCNN(in_features=in_feat, num_classes=args.num_classes, dim=dim).to(args.device)
        args.model = FedAvgCNN_toy(in_features=in_feat, num_classes=args.num_classes, dim=dim, fd=args.feature_dim)

    elif model_str == "cnn_spc":  # non-convex
        # args.model = FedAvgCNN(in_features=in_feat, num_classes=args.num_classes, dim=dim).to(args.device)
        args.model = FedAvgCNN_spc(in_features=in_feat, num_classes=args.num_classes, dim=dim, fd=args.feature_dim)

    elif model_str == "dnn":  # non-convex
        if "mnist" in args.dataset:
            args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
        else:
            args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

    elif model_str == "NN":  # non-convex
        if "mnist" in args.dataset:
            args.model = NN(1 * 28 * 28, num_classes=args.num_classes)
        elif "Cifar10" in args.dataset:
            args.model = NN(3 * 32 * 32, num_classes=args.num_classes)
        elif "fer2013" in args.dataset:
            args.model = NN(3 * 48 * 48, num_classes=args.num_classes)
        elif "skin" in args.dataset:
            args.model = NN(3 * 28 * 28, num_classes=args.num_classes)
        elif "wine_quality" in args.dataset:
            args.model = NN(11, num_classes=args.num_classes)
        elif "Sensorless_drive_diagnosis" in args.dataset:
            args.model = NN(48, num_classes=args.num_classes)
        elif "satimage" in args.dataset:
            args.model = NN(36, num_classes=args.num_classes)
        else:
            args.model = NN(60, num_classes=args.num_classes)

    elif model_str == "VGG16":  # non-convex
        args.model = VGG(vgg_name=model_str, in_features=in_feat, num_classes=args.num_classes)

    elif model_str == "resnet":
        args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

    elif model_str == "resnet8":
        args.model = resnet8(num_classes=args.num_classes).to(args.device)

    elif model_str == "resnet10":
        args.model = resnet10(num_classes=args.num_classes).to(args.device)

    elif model_str == "resnet18":
        # args.model = resnet18(num_classes=args.num_classes).to(args.device)
        if args.privacy:
            args.model = resnet18(num_classes=args.num_classes, inplace=False)
        else:
            args.model = resnet18(num_classes=args.num_classes)

    elif model_str == "resnet34":
        args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

    elif model_str == "alexnet":
        args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)

    elif model_str == "googlenet":
        args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                  num_classes=args.num_classes).to(args.device)

    elif model_str == "mobilenet_v2":
        args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)

    elif model_str == "shufflenet":
        args.model = torchvision.models.shufflenet_v2_x1_0(pretrained=False, num_classes=args.num_classes).to(args.device)

    elif model_str == "mobilevit":
        args.model = mobilevit_s(num_classes=args.num_classes)

    elif model_str == "lstm":
        args.model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
            args.device)

    elif model_str == "bilstm":
        args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim,
                                               output_size=args.num_classes,
                                               num_layers=1, embedding_dropout=0, lstm_dropout=0,
                                               attention_dropout=0,
                                               embedding_length=emb_dim).to(args.device)

    elif model_str == "fastText":
        # args.model = fastText(
        #     hidden_dim=emb_dim,
        #     vocab_size=vocab_size,
        #     num_classes=args.num_classes
        # ).to(args.device) # original version

        args.model = fastText(
            input_dim=emb_dim,
            # hidden_dim=emb_dim,
            hidden_dim=args.feature_dim, # for PBFL
            vocab_size=vocab_size,
            num_classes=args.num_classes
        )


    elif model_str == "TextCNN":
        args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size,
                             num_classes=args.num_classes).to(args.device)

    elif model_str == "Transformer":
        args.model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=8, d_hid=emb_dim, nlayers=2,
                                      num_classes=args.num_classes).to(args.device)

    elif model_str == "AmazonMLP":
        args.model = AmazonMLP().to(args.device)

    elif model_str == "harcnn":
        if args.dataset == 'har':
            args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                pool_kernel_size=(1, 2)).to(args.device)
        elif args.dataset == 'pamap':
            args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                pool_kernel_size=(1, 2)).to(args.device)

    elif args.model_family == "Ht0":
            args.model = [
                'resnet8(num_classes=args.num_classes)',
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.shufflenet_v2_x1_0(pretrained=False)',
                'torchvision.models.efficientnet_b0(pretrained=False)',
            ]

    elif args.model_family == "HtFE2":
        args.model = [
            f'FedAvgCNN(in_features={in_feat}, num_classes=args.num_classes, dim={dim})',
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
        ]

    elif args.model_family == "HtFE8":
        args.model = [
            f'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim={dim})',
            'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
            'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
        ]

    elif args.model_family == "HmCNN":
        args.model = [
            f'FedAvgCNN(in_features={in_feat}, num_classes=args.num_classes, dim={dim})'
        ]

    elif args.model_family == "HmMobile":
        args.model = [
            'mobilenet_v2(pretrained=False, num_classes=args.num_classes)'
        ]

    elif model_str == "HmRes8":
        args.model = ['resnet8(num_classes=args.num_classes)']

    elif args.model_family == "HmRes18":
        args.model = [
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)'
        ]

    elif args.model_family == "HtText":
        args.model = [
            'fastText(input_dim=emb_dim, hidden_dim=args.feature_dim, vocab_size=vocab_size, num_classes=args.num_classes)',
            'TextCNN(input_dim=emb_dim, hidden_dim=args.feature_dim, max_len=max_len, vocab_size=vocab_size, num_classes=args.num_classes)'
        ]

    # elif model_str == "HtFE3":
    #     args.model = [
    #         'resnet10(num_classes=args.num_classes)',
    #         'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
    #     ]
    #
    # elif model_str == "HtFE4":
    #     args.model = [
    #         'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',
    #         'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
    #         'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)'
    #     ]
    #
    # elif model_str == "HtFE9":
    #     args.model = [
    #         'resnet4(num_classes=args.num_classes)',
    #         'resnet6(num_classes=args.num_classes)',
    #         'resnet8(num_classes=args.num_classes)',
    #         'resnet10(num_classes=args.num_classes)',
    #         'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)',
    #     ]
    #
    # elif model_str == "HtFE8-HtC4":
    #     args.model = [
    #         'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',
    #         'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
    #         'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
    #     ]
    #     args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)'
    #     args.heads = [
    #         'Head(hidden_dims=[512], num_classes=args.num_classes)',
    #         'Head(hidden_dims=[512, 512], num_classes=args.num_classes)',
    #         'Head(hidden_dims=[512, 256], num_classes=args.num_classes)',
    #         'Head(hidden_dims=[512, 128], num_classes=args.num_classes)',
    #     ]
    #
    # elif model_str == "Res34-HtC4":
    #     args.model = [
    #         'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
    #     ]
    #     args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)'
    #     args.heads = [
    #         'Head(hidden_dims=[512], num_classes=args.num_classes)',
    #         'Head(hidden_dims=[512, 512], num_classes=args.num_classes)',
    #         'Head(hidden_dims=[512, 256], num_classes=args.num_classes)',
    #         'Head(hidden_dims=[512, 128], num_classes=args.num_classes)',
    #     ]
    #
    # elif model_str == "HCNNs8":
    #     args.model = [
    #         'CNN(num_cov=1, hidden_dims=[], in_features=1, num_classes=args.num_classes)',
    #         'CNN(num_cov=2, hidden_dims=[], in_features=1, num_classes=args.num_classes)',
    #         'CNN(num_cov=1, hidden_dims=[512], in_features=1, num_classes=args.num_classes)',
    #         'CNN(num_cov=2, hidden_dims=[512], in_features=1, num_classes=args.num_classes)',
    #         'CNN(num_cov=1, hidden_dims=[1024], in_features=1, num_classes=args.num_classes)',
    #         'CNN(num_cov=2, hidden_dims=[1024], in_features=1, num_classes=args.num_classes)',
    #         'CNN(num_cov=1, hidden_dims=[1024, 512], in_features=1, num_classes=args.num_classes)',
    #         'CNN(num_cov=2, hidden_dims=[1024, 512], in_features=1, num_classes=args.num_classes)',
    #     ]
    #
    # elif model_str == "ViTs":
    #     args.model = [
    #         'torchvision.models.vit_b_16(image_size=32, num_classes=args.num_classes)',
    #         'torchvision.models.vit_b_32(image_size=32, num_classes=args.num_classes)',
    #         'torchvision.models.vit_l_16(image_size=32, num_classes=args.num_classes)',
    #         'torchvision.models.vit_l_32(image_size=32, num_classes=args.num_classes)',
    #     ]
    #
    # elif model_str == "HtM10":
    #     args.model = [
    #         'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',
    #         'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
    #         'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.vit_b_16(image_size=32, num_classes=args.num_classes)',
    #         'torchvision.models.vit_b_32(image_size=32, num_classes=args.num_classes)'
    #     ]
    #
    # elif model_str == "NLP_all":
    #     args.model = [
    #         'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
    #         'LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
    #         'BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, output_size=args.num_classes, num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, embedding_length=args.feature_dim)',
    #         'TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, num_classes=args.num_classes)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)'
    #     ]
    #
    # elif model_str == "NLP_Transformers-nhead=8":
    #     args.model = [
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
    #     ]
    #
    # elif model_str == "NLP_Transformers-nlayers=4":
    #     args.model = [
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
    #     ]
    #
    # elif model_str == "NLP_Transformers":
    #     args.model = [
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
    #     ]
    #
    # elif model_str == "MLPs":
    #     args.model = [
    #         'AmazonMLP(feature_dim=[])',
    #         'AmazonMLP(feature_dim=[200])',
    #         'AmazonMLP(feature_dim=[500])',
    #         'AmazonMLP(feature_dim=[1000, 500])',
    #         'AmazonMLP(feature_dim=[1000, 500, 200])',
    #     ]
    #
    # elif model_str == "MLP_1layer":
    #     args.model = [
    #         'AmazonMLP(feature_dim=[200])',
    #         'AmazonMLP(feature_dim=[500])',
    #     ]
    #
    # elif model_str == "MLP_layers":
    #     args.model = [
    #         'AmazonMLP(feature_dim=[500])',
    #         'AmazonMLP(feature_dim=[1000, 500])',
    #         'AmazonMLP(feature_dim=[1000, 500, 200])',
    #     ]
    #
    # elif model_str == "HtFE-img-2":
    #     args.models = [
    #         'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',  # for 32x32 img
    #         'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
    #     ]
    #
    # elif model_str == "HtFE-img-3":
    #     args.models = [
    #         'resnet10(num_classes=args.num_classes)',
    #         'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
    #     ]
    #
    # elif model_str == "HtFE-img-4":
    #     args.models = [
    #         'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',  # for 32x32 img
    #         'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
    #         'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)'
    #     ]
    #
    # elif model_str == "HtFE-img-5":
    #     args.models = [
    #         'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
    #         'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
    #     ]
    #
    # elif model_str == "HtFE-img-8":
    #     args.models = [
    #         'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',  # for 32x32 img
    #         # 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816)', # for 64x64 img
    #         'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
    #         'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
    #     ]
    #
    # elif model_str == "HtFE-img-9":
    #     args.models = [
    #         'resnet4(num_classes=args.num_classes)',
    #         'resnet6(num_classes=args.num_classes)',
    #         'resnet8(num_classes=args.num_classes)',
    #         'resnet10(num_classes=args.num_classes)',
    #         'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)',
    #     ]
    #
    # elif model_str == "HtFE-img-8-HtC-img-4":
    #     args.models = [
    #         'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',  # for 32x32 img
    #         'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
    #         'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
    #     ]
    #     args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)'  # for 32x32 img
    #     args.heads = [
    #         'Head(hidden_dims=[512], num_classes=args.num_classes)',
    #         'Head(hidden_dims=[512, 512], num_classes=args.num_classes)',
    #         'Head(hidden_dims=[512, 256], num_classes=args.num_classes)',
    #         'Head(hidden_dims=[512, 128], num_classes=args.num_classes)',
    #     ]
    #
    # elif model_str == "Res34-HtC-img-4":
    #     args.models = [
    #         'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
    #     ]
    #     args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)'  # for 32x32 img
    #     args.heads = [
    #         'Head(hidden_dims=[512], num_classes=args.num_classes)',
    #         'Head(hidden_dims=[512, 512], num_classes=args.num_classes)',
    #         'Head(hidden_dims=[512, 256], num_classes=args.num_classes)',
    #         'Head(hidden_dims=[512, 128], num_classes=args.num_classes)',
    #     ]
    #
    # elif model_str == "HtM-img-10":
    #     args.models = [
    #         'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',  # for 32x32 img
    #         'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
    #         'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)',
    #         'torchvision.models.vit_b_16(image_size=32, num_classes=args.num_classes)',
    #         'torchvision.models.vit_b_32(image_size=32, num_classes=args.num_classes)'
    #     ]
    #
    # elif model_str == "HtFE-txt-2":
    #     args.models = [
    #         'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
    #         'TextLogisticRegression(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)'
    #     ]
    #
    # elif model_str == "HtFE-txt-4":
    #     args.models = [
    #         'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
    #         'TextLogisticRegression(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
    #         'LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
    #         'BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, output_size=args.num_classes, num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, embedding_length=args.feature_dim)'
    #     ]
    #
    # elif model_str == "HtFE-txt-5-1":
    #     args.models = [
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
    #     ]
    #
    # elif model_str == "HtFE-txt-5-2":
    #     args.models = [
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
    #     ]
    #
    # elif model_str == "HtFE-txt-5-3":
    #     args.models = [
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
    #     ]
    #
    # elif model_str == "HtFE-txt-6":
    #     args.models = [
    #         'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
    #         'LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
    #         'BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, output_size=args.num_classes, num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, embedding_length=args.feature_dim)',
    #         'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
    #         'TextLogisticRegression(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
    #         'GRUNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)'
    #     ]
    #
    # elif model_str == "MLPs":
    #     args.models = [
    #         'AmazonMLP(feature_dim=[])',
    #         'AmazonMLP(feature_dim=[500])',
    #         'AmazonMLP(feature_dim=[1000, 500])',
    #         'AmazonMLP(feature_dim=[1000, 500, 200])',
    #     ]
    #
    # elif model_str == "MLP_1layer":
    #     args.models = [
    #         'AmazonMLP(feature_dim=[200])',
    #         'AmazonMLP(feature_dim=[500])',
    #     ]
    #
    # elif model_str == "MLP_layers":
    #     args.models = [
    #         'AmazonMLP(feature_dim=[500])',
    #         'AmazonMLP(feature_dim=[1000, 500])',
    #         'AmazonMLP(feature_dim=[1000, 500, 200])',
    #     ]
    #
    # elif model_str == "HtFE-sen-2":
    #     args.models = [
    #         'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
    #         'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
    #     ]
    #
    # elif model_str == "HtFE-sen-3":
    #     args.models = [
    #         'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
    #         'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
    #         'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=3)',
    #     ]
    #
    # elif model_str == "HtFE-sen-5":
    #     args.models = [
    #         'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
    #         'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
    #         'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=3)',
    #         'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=1)',
    #         'HARCNN3(9, dim_hidden=3328, num_classes=args.num_classes, stride=1)',
    #     ]
    #
    # elif model_str == "HtFE-sen-8":
    #     args.models = [
    #         'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
    #         'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
    #         'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=3)',
    #         'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=1)',
    #         'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=2)',
    #         'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=3)',
    #         'HARCNN3(9, dim_hidden=3328, num_classes=args.num_classes, stride=1)',
    #         'HARCNN3(9, dim_hidden=3328, num_classes=args.num_classes, stride=2)',
    #     ]
    else:
        raise NotImplementedError

    return args.model
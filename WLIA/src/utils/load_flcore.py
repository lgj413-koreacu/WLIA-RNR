from flcore.servers.serveravg import FedAvg

import copy
from torch import nn


# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out


def assign_fl_core(args, i):
    # select algorithm

    if args.algorithm == "Local":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = Local(args, i)

    elif args.algorithm == "FedAvg":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedAvg(args, i)

    elif args.algorithm == "cwFedAvg":
        if args.partial_layer_train:
            layer_names = []
            for name, _ in args.model.named_children():
                layer_names.append(name)

            args.layer_groups = {
                "common": layer_names[:-args.cw_layer_num],
                "cw": layer_names[-args.cw_layer_num:]
            }

            for key, value in args.layer_groups.items():
                print(f'{key}: {value}')
        else:
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)

        server = cwFedAvg(args, i)

    elif args.algorithm == "FedRWR":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedRWR(args, i)

    elif args.algorithm == "FedClassAvg":
        if isinstance(args.model, list):
            pass
        else:
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
        server = FedClassAvg(args, i)

    elif args.algorithm == "CHFL":
        if isinstance(args.model, list):
            pass
        else:
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
        server = CHFL(args, i)

    elif args.algorithm == "FedSPC":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedSPC(args, i)

    elif args.algorithm == "LG_FedAvg":
        if isinstance(args.model, list):
            pass
        else:
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
        server = LG_FedAvg(args, i)

    elif args.algorithm == "FedDistill":
        server = FedDistill(args, i)

    elif args.algorithm == "FML":
        server = FML(args, i)

    elif args.algorithm == "FedKD":
        if isinstance(args.model, list):
            pass
        else:
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
        server = FedKD(args, i)

    elif args.algorithm == "FedNH":
        server = FedNH(args, i)

    elif args.algorithm == "FedUV":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedUV(args, i)

    elif args.algorithm == 'FedAS':
        if isinstance(args.model, list):
            pass
        else:
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
        server = FedAS(args, i)

    elif args.algorithm == "FedDBE":
        if isinstance(args.model, list):
            pass
        else:
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
        server = FedDBE(args, i)

    # elif args.algorithm == "FedProto":
    #     server = FedProto(args, i)

    elif args.algorithm == "FedProto":
        if isinstance(args.model, list):
            pass
        else:
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
        server = FedProto(args, i)

    elif args.algorithm == "FedTGP":
        if isinstance(args.model, list):
            pass
        else:
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
        server = FedTGP(args, i)

    elif args.algorithm == "TinyFP":
        if isinstance(args.model, list):
            pass
        else:
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
        server = TinyFP(args, i)

    elif args.algorithm == "TinyFT":
        if isinstance(args.model, list):
            pass
        else:
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
        server = TinyFT(args, i)

    elif args.algorithm == "ProtoNorm":
        if isinstance(args.model, list):
            pass
        else:
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
        server = ProtoNorm(args, i)

    else:
        raise NotImplementedError

    return server
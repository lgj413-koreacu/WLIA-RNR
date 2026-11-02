

from opacus import PrivacyEngine
import torch
import numpy as np
import copy

MAX_GRAD_NORM = 1.0
DELTA = 1e-5

def initialize_dp(
        model,
        optimizer,
        data_loader,
        dp_sigma=0.3):
    privacy_engine = PrivacyEngine(secure_mode=False)

    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier = dp_sigma, 
        max_grad_norm = MAX_GRAD_NORM
    )
    return model, optimizer, data_loader, privacy_engine


def get_dp_params(privacy_engine):
    return privacy_engine.get_epsilon(delta=DELTA), DELTA


def apply_dp_to_prototypes(prototypes, max_norm=1.0, noise_multiplier=0.5):
    """
    prototypes: dict[class_id] = [feature_dim]
    """
    dp_prototypes = {}
    for cls, proto in prototypes.items():
        norm = proto.norm()
        clipped = proto * min(1.0, max_norm / (norm + 1e-6))
        noise = torch.randn_like(clipped) * noise_multiplier * max_norm
        dp_prototypes[cls] = clipped + noise
    return dp_prototypes


def PGD_attack(c, feature_dis, num_feature, ml, g, lr, var_factor):
    r_min = 0
    r_max = 1
    x = torch.tensor(np.random.uniform(r_min, r_max, num_feature)).float().requires_grad_()
    opt = torch.optim.Adam([x], lr=lr)

    for _ in range(25):
        yg = torch.softmax(g(x), 0)
        yc = torch.softmax(ml[c](x), 0)

        loss = torch.nn.MSELoss()(yg, yc) + var_factor * x.var()

        opt.zero_grad()
        loss.backward()
        opt.step()

        # projection
        x = torch.clamp(x, r_min, r_max).clone().detach().numpy()
        for f in range(num_feature):
            # discrete data
            if type(feature_dis[f][0]) is np.ndarray:
                x[f] = np.random.choice(feature_dis[f][0], 1)[0]
        x = torch.tensor(x).requires_grad_()
        opt = torch.optim.Adam([x], lr=lr)
    return x.detach().numpy()


def hc_noise_attack(feature_dis, num_feature, num_client, fake_sample_size, ml, g, prior_dist):
    fake_sample_size = int(fake_sample_size / num_client)
    sample_list = []
    subsample_list = []

    for c in range(num_client):
        hc_sample = []
        hc_subsample = []
        threshold = 1e-3

        d_max = -1
        d_min = 9999

        cnt = 0
        d_list = []
        d_threshold = 0
        while len(hc_subsample) < fake_sample_size:
            x = hc_attack(c, feature_dis, num_feature, ml, g, prior_dist, threshold)
            j = 0

            # generate a valid sample
            while x is None:
                x = hc_attack(c, feature_dis, num_feature, ml, g, prior_dist, threshold)
                j += 1
                if j > 10:
                    threshold *= 2
                    # print("threshold =", threshold)
                    j = 0

            if len(hc_sample) < fake_sample_size:
                hc_sample.append(copy.deepcopy(x))
                hc_subsample.append(copy.deepcopy(x))

        sample_list.append(copy.deepcopy(np.array(hc_sample)))
        subsample_list.append(copy.deepcopy(np.array(hc_subsample)))
    return (sample_list, subsample_list)


def hc_attack(c, feature_dist, num_feature, ml, g, prior_dist, threshold):
    """
    c: client
    kmax = 128
    rejmax = 10
    kmin = 4
    confmin = 0.2
    """

    # def g(x):
    #    pred_prob = np.mean([local_ml.predict_proba(x.reshape(1, -1)) for local_ml in ml], 0)[0]
    #    return pred_prob

    # continues feature ~ N(0, 1)
    # we only need to modify discrete data
    def random_value(prior_dist):
        if prior_dist == 'uniform':
            return np.random.uniform(r_min, r_max)
        elif prior_dist == 'gaussian':
            return np.random.normal(0.5, 0.25)

    r_min = 0
    r_max = 1
    x = np.random.uniform(r_min, r_max, num_feature)
    for f in range(num_feature):
        x[f] = random_value(prior_dist)
        # discrete data
        if type(feature_dist[f][0]) is np.ndarray:
            x[f] = np.random.choice(feature_dist[f][0], 1)[0]

    y_star = 0
    diff_star = 999
    j = 0
    k = int(num_feature / 2)  # 128
    rejmax = 10
    kmin = 4
    # threshold = threshold

    for _ in range(1000):
        tmpx = torch.tensor(x).float()
        yg = torch.softmax(g(tmpx), 0)  # .detach().numpy()
        yc = torch.softmax(ml[c](tmpx), 0)  # .detach().numpy()
        diff = torch.nn.MSELoss()(yg, yc) + 0.1 * tmpx.var()

        if diff <= diff_star:
            if diff <= threshold:
                # print("iter=", _)
                return x
            x_star = x
            diff_star = diff
            j = 0
        j += 1
        if j > rejmax:
            k = int(max(kmin, np.ceil(k / 2)))
            j = 0
        # x <-- randrecord(x*, k)
        # select k features randomly
        feature_id = np.random.choice(num_feature, k, replace=False)
        x_star_tmp = copy.deepcopy(x_star)
        for f in feature_id:
            x_star_tmp[f] = random_value(prior_dist)
            # discrete data
            if type(feature_dist[f][0]) is np.ndarray:
                x_star_tmp[f] = np.random.choice(feature_dist[f][0], 1)[0]
        x = x_star_tmp
    return None
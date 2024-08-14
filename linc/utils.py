import torch
import numpy as np
import random
import torch.nn.functional as F
from scipy.stats import rankdata
import pickle
from torch.utils.data import Dataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class LINCDataset(Dataset):
    def __init__(self, data_path, split=1, key="train"):
        super(LINCDataset, self).__init__()
        X, y = self.load_data(data_path, key)
        self.y = y[::split]
        self.X = X[::split]

    def load_data(self, data_path, key):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        return data[key]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx] * 2 - 1  # [0, 1] -> [-1. 1]
        y = self.y[idx]
        return x.astype("float32"), y.astype("int64")


def quantity(x, num_bit):
    out = np.zeros(num_bit)
    xbin = str(bin(int(x)))[2:]
    bin_len = len(xbin)
    min_len = min(num_bit, bin_len)

    for i in range(min_len):
        out[i] = xbin[bin_len - i - 1]
    return out[::-1]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def generate_rules(model, dataset):
    Keys = []
    Masks = []
    Scores = []
    Values = []

    for i, nnus in enumerate(model.ensemble_nnus):
        scores = []
        masks = []
        keys = dataset.X
        y = dataset.y
        index = np.argwhere(y == i)[:, -1]
        keys = keys[index]
        in_fea = len(keys[0])

        for nnu in nnus.nnus:
            mask = (list(nnu.children())[0].get_mask() > 0).cpu().numpy()
            masks.append(mask)
            use_keys = keys * mask  # (N, in_fea)
            X = use_keys * 2 - 1
            X = torch.from_numpy(X.astype("float32")).cuda()
            with torch.no_grad():
                scores.append(nnu(X).cpu().numpy())

        scores = np.concatenate(scores, axis=1)  # (Bs, num_ensemble)
        index = np.argmax(scores, axis=-1)
        scores = np.max(scores, axis=-1)
        full_masks = []
        for k in index:
            full_masks.append(masks[k])
        masks = np.concatenate(full_masks, axis=0)  # (Bs, 112)
        use_keys = keys * masks
        keys_masks = np.concatenate([use_keys, masks], axis=1)  # (Bs, 224)
        keys_masks, index = np.unique(
            keys_masks, axis=0, return_index=True
        )  # (M, in_fea)
        scores = scores[index]
        keys, masks = keys_masks[:, :in_fea], keys_masks[:, in_fea:]

        for k in range(len(keys)):
            Keys.append(keys[k].tolist())
            Masks.append(masks[k].tolist())
            Scores.append(scores[k])
            Values.append(i)

    Scores = np.array(Scores)
    Scores = rankdata(Scores)
    Keys = np.array(Keys)
    Masks = np.array(Masks)
    Values = np.array(Values)
    index = np.argsort(Scores)

    return Keys[index], Masks[index], Values[index], Scores[index]


def merge_neighbor_mask(keys, scores, masks, values):
    def cover_(k1, k2, m1, m2):
        m = m1 + m2
        index = np.argwhere(m == 2)[:, -1]
        if len(index) == 0:
            return False
        if (k1[index] == k2[index]).sum() == len(index):
            return False
        return True

    def diff(k1, k2):
        index = np.argwhere(k1 != k2)[:, -1]
        if len(index) == 1:
            return True, index[0]
        else:
            return False, 0

    del_id = []
    for i in range(len(keys) - 1):
        cover = True
        for j in range(i + 1, len(keys)):
            if not cover:
                break
            if values[i] == values[j] and (masks[i] == masks[j]).sum() == len(masks[i]):
                is_match, index = diff(keys[i], keys[j])
                if not is_match:
                    continue
                del_id.append(i)
                masks[j, index] = 0
                break
            if values[i] == values[j]:
                continue
            cover = cover_(keys[i], keys[j], masks[i], masks[j])

    masks = np.delete(masks, del_id, axis=0)
    keys = np.delete(keys, del_id, axis=0)
    scores = np.delete(scores, del_id, axis=0)
    values = np.delete(values, del_id, axis=0)

    return keys, scores, masks, values


def toInt(array):
    return int("0b" + "".join(str(int(i)) for i in array), 2)


def resort_rules(keys, scores, masks, values):
    scores = rankdata(-1 * scores)
    ind = np.argsort(scores)
    return keys[ind], masks[ind], values[ind], scores[ind]


def load_from_txt(path, num_bit=112, min_score=1000):
    with open(path, "r") as f:
        rules = f.readlines()
    Keys, Masks, Values, Scores = [], [], [], []
    for r in rules:
        m, k, value, score = r.split()
        Keys.append(quantity(k, num_bit))
        Masks.append(quantity(m, num_bit))
        Values.append(int(value))
        Scores.append(float(score))
    Keys = np.vstack(Keys)
    Masks = np.vstack(Masks)
    Values = np.array(Values)
    Scores = np.array(Scores)

    return resort_rules(Keys, Scores, Masks, Values)


def load_from_mousika(path, num_bit=112, min_score=1000):
    with open(path, "r") as f:
        rules = f.readlines()
    Keys, Masks, Values = [], [], []
    for r in rules:
        m, k, value = r.split()
        Keys.append(quantity(k, num_bit))
        Masks.append(quantity(m, num_bit))
        Values.append(int(value))

    Keys = np.vstack(Keys)
    Masks = np.vstack(Masks)
    Values = np.array(Values)
    Scores = np.ones(len(Values)) * min_score
    return Keys, Masks, Values, Scores


def load_positive_samples(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

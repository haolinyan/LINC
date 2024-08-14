import argparse
from linc.utils import (
    load_from_txt,
    LINCDataset,
    setup_seed,
    AverageMeter,
    load_positive_samples,
    merge_neighbor_mask,
    rankdata,
    toInt,
    load_from_mousika,
)
from linc.mousika import Mousika
from linc.mousika_utils import rule2entry
from torch.utils.data import ConcatDataset
from loguru import logger
from linc.simulator import Switch, Simulator
from torch.utils.data import DataLoader
import pickle
import numpy as np
import torch
import time
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
import os

RATIO = 1  # 0.05% of new dat


def load_iscx_binary(path, key):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data[key]


setup_seed(0)


class LINCDataset_New(LINCDataset):
    def load_data(self, data_path, key):
        X, y = load_iscx_binary(data_path, key=key)
        mask = np.isin(y, [4])
        X = X[mask]
        y = np.ones_like(y[mask]) * 3
        if key != "test":
            X = X[: int(X.shape[0] * RATIO)]
            y = y[: int(y.shape[0] * RATIO)]

        return X, y


class LINCDataset_Old(LINCDataset):
    def load_data(self, data_path, key):
        X, y = load_iscx_binary(data_path, key=key)
        mask = np.isin(y, [0, 1, 2])
        X = X[mask]
        y = y[mask]
        return X, y


def validate_filter_err(switch, loader):
    batch_time = AverageMeter("Tps", ":6.3f")
    allocate_err_x, allocate_err_y, allocate_err_id = [], [], []
    y_list = []
    pred_list = []
    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(loader):
            X = batch[0].cuda()
            y = batch[1].cuda()
            pred, rules_id = switch((X + 1) / 2, y, return_key=True)
            y_list += y.cpu().numpy().tolist()
            pred_list += pred.cpu().numpy().tolist()
            index = torch.argwhere(pred != y).flatten()
            allocate_err_x.append(torch.index_select(X, 0, index).cpu().numpy())
            allocate_err_y += torch.index_select(y, 0, index).cpu().numpy().tolist()
            allocate_err_id += (
                torch.index_select(rules_id, 0, index).cpu().numpy().tolist()
            )
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

    return (
        np.vstack(allocate_err_x),
        np.array(allocate_err_y),
        np.array(allocate_err_id),
        accuracy_score(y_list, pred_list),
    )


def fit_mousika(X, y, total_bit=96, out="./outputs/mousika.txt"):
    data = np.concatenate([X, y[..., None]], axis=1)
    model = Mousika()
    model.fit(data)
    model.export(out)
    rule2entry(out, total_bit=total_bit - 1)
    keys, masks, values, scores = load_from_mousika(
        "tmp_rule2entry.txt", num_bit=total_bit
    )
    return keys, masks, values, scores


def insert_rules(index, old_rules, new_rules):
    keys, masks, values, scores = old_rules
    keys_, masks_, values_, scores_ = new_rules

    base_mask = masks[index]
    base_key = keys[index]

    ind = base_mask == 1
    for i in range(len(keys_)):
        keys_[i][ind] = base_key[ind]
        masks_[i][ind] = base_mask[ind]

    new_keys = np.vstack([keys[: index + 1], keys_, keys[index + 1 :]])
    new_masks = np.vstack([masks[: index + 1], masks_, masks[index + 1 :]])
    new_values = np.array(
        values[: index + 1].tolist() + values_.tolist() + values[index + 1 :].tolist()
    )
    new_scores = np.arange(len(new_values))
    return new_keys, new_masks, new_values, new_scores


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    keys, masks, values, scores = load_from_txt(args.path, num_bit=args.in_features)
    logger.info(f"Loaded {len(keys)} rules from {args.path}")
    switch = Switch(
        args.in_features, keys, masks, values, scores, num_classes=args.num_classes - 1
    )
    switch = switch.cuda()
    switch.eval()
    train_data_n = LINCDataset_New(args.data, key="train")
    val_data_n = LINCDataset_New(args.data, key="val")
    train_data_n = ConcatDataset([train_data_n, val_data_n])
    test_data_n = LINCDataset_New(args.data, key="test")
    logger.info(
        f"Loaded {len(train_data_n)} training samples and {len(test_data_n)} test samples"
    )
    train_loader_n = DataLoader(
        train_data_n, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader_n = DataLoader(
        test_data_n, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_data_o = LINCDataset_Old(args.data, key="test")
    val_data_o = LINCDataset_Old(args.data, key="val")
    test_loader_o = DataLoader(
        test_data_o, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    val_data_all = ConcatDataset([train_data_n, val_data_o])
    val_loader_all = DataLoader(
        val_data_all, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_overall = ConcatDataset([test_data_n, test_data_o])
    test_loader_overall = DataLoader(
        test_overall, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    allocate_err_x, allocate_err_y, allocate_err_id, _ = validate_filter_err(
        switch, train_loader_n
    )
    allocate_err_x = (allocate_err_x + 1) / 2  # [-1, 1] -> [0, 1]
    uni_id, id_cnt = np.unique(allocate_err_id, return_counts=True)
    allocate_err_dict = {}
    for i in range(len(allocate_err_id)):
        if allocate_err_id[i] not in allocate_err_dict:
            allocate_err_dict[allocate_err_id[i]] = [
                [allocate_err_x[i]],
                [allocate_err_y[i]],
            ]
        else:
            allocate_err_dict[allocate_err_id[i]][0].append(allocate_err_x[i])
            allocate_err_dict[allocate_err_id[i]][1].append(allocate_err_y[i])
    positive_samples = load_positive_samples(args.pos_data)
    for ind in np.sort(list(allocate_err_dict.keys()))[::-1]:
        pos_X = positive_samples[ind][str(int(values[ind]))]
        pos_y = np.zeros(len(pos_X))
        if len(pos_X) < 2:
            logger.info(
                f"no positive samples, change to new class: {values[ind]} -> {args.new_cls}"
            )
            values[ind] = args.new_cls
            continue
        neg_X = allocate_err_dict[ind][0]
        neg_y = np.ones(len(allocate_err_dict[ind][1]))
        X = np.vstack([pos_X, neg_X])
        y = np.hstack([pos_y, neg_y])
        runs = RandomOverSampler(random_state=42)
        X, y = runs.fit_resample(X, y)
        keys_, masks_, values_, scores_ = fit_mousika(
            X,
            y,
            total_bit=args.in_features,
            out=os.path.join(args.save_dir, "mousika.txt"),
        )
        m = values_ == 1
        keys_ = keys_[m]
        masks_ = masks_[m]
        values_ = np.ones(len(masks_)) * args.new_cls
        scores_ = values_
        keys, masks, values, scores = insert_rules(
            ind, [keys, masks, values, scores], [keys_, masks_, values_, scores_]
        )
    raw_n_keys = len(keys)
    min_num_rules = len(keys)
    round = 0
    while True:
        keys, scores, masks, values = merge_neighbor_mask(keys, scores, masks, values)
        round += 1
        if len(keys) == min_num_rules:
            break
        else:
            min_num_rules = len(keys)
    switch = Switch(
        args.in_features, keys, masks, values, scores, num_classes=args.num_classes
    )
    switch = switch.cuda()
    switch.eval()
    _, _, _, test_o_acc = validate_filter_err(switch, test_loader_o)
    _, _, _, test_n_acc = validate_filter_err(switch, test_loader_n)
    _, _, _, test_all_acc = validate_filter_err(switch, test_loader_overall)
    logger.info(
        f"test_o_acc: {test_o_acc}, test_n_acc: {test_n_acc}, test_all_acc: {test_all_acc}"
    )
    logger.info(f"round: {round}, raw_n_keys: {raw_n_keys}, n_keys: {len(keys)}")

    simulator = Simulator(len(keys[0]), keys, masks, values, scores, num_classes=6)
    simulator.validate(val_loader_all, cache=True)
    simulator.save_allocate_samples(os.path.join(args.save_dir, "cache.pkl"))

    scores = rankdata(-1 * scores)
    rule_string = ""
    for m, k, v, s in zip(masks, keys, values, scores):
        rule_string += "%d %d %d %d\n" % (toInt(m), toInt(k), v, s)
    with open(os.path.join(args.save_dir, "rules.txt"), "w") as f:
        f.write(rule_string[:-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="checkpoints/iscx_c3/rules.txt")
    parser.add_argument("--in_features", type=int, default=96)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--new_cls", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="checkpoints/iscx_c4")
    parser.add_argument("--data", type=str, default="iscx.pkl")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--pos_data",
        default="checkpoints/iscx_c3/cache.pkl",
        type=str,
        help="path to positive samples",
    )
    args = parser.parse_args()
    begin = time.time()
    main(args)
    logger.info(f"Update Time: {time.time() - begin}")

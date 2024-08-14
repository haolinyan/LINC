import torch
import torch.nn as nn
from .utils import *
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time
import pickle
from loguru import logger


class Switch(nn.Module):
    def __init__(self, in_fea, keys, mask, values, scores, num_classes=2):
        super(Switch, self).__init__()
        self.raw_memory = [in_fea, keys, mask, values, scores, num_classes]
        self.set_memory(*self.raw_memory)

    def set_memory(self, in_fea, keys, mask, values, scores, num_classes):
        self.num_keys = len(keys)
        self.register_buffer(
            "keys", torch.from_numpy(keys.astype("float32")).unsqueeze(0)
        )  # (1, N, in_fea)
        self.register_buffer(
            "masks",
            torch.abs(torch.from_numpy(mask.astype("float32")).unsqueeze(0) - 1),
        )  # (1, N, in_fea)
        self.register_buffer("values", torch.from_numpy(values))
        self.register_buffer(
            "scores", torch.from_numpy(scores.astype("float32")).unsqueeze(0)
        )  # (1, N)
        self.register_buffer("in_fea", torch.ones(1, self.num_keys) * in_fea)
        self.allocate_samples = [
            {"%d" % d: [] for d in range(num_classes)} for _ in range(len(scores) + 1)
        ]
        self.default_value = 0

    def forward(self, query, label, num_drop=0, cache=False, return_key=False):
        """
        query: (Batchsize, in_fea) bitarray
        """
        query = query.unsqueeze(1)
        out = 1 - (
            query + self.keys[:, num_drop:, :] - 2 * query * self.keys[:, num_drop:, :]
        )
        out = (
            out + self.masks[:, num_drop:, :] - out * self.masks[:, num_drop:, :]
        )  # (1, N, in_fea)
        out = out.sum(dim=-1)  # (Batchsize, N)

        mask = out == self.in_fea[:, num_drop:].expand_as(out)
        index = torch.argmax(
            mask * self.scores[:, num_drop:].expand_as(out), dim=-1
        )  # (Batchsize,)
        pred = torch.index_select(self.values[num_drop:], 0, index)

        index_default = torch.sum(mask, dim=-1) == 0
        index_match = torch.sum(mask, dim=-1) != 0
        pred = pred * index_match + index_default * self.default_value
        index = index * index_match.long() + index_default.long() * len(self.scores)
        if cache:
            for q, label, ind in zip(
                query[:, 0, ...].cpu().numpy(), label.cpu().numpy(), index.cpu().numpy()
            ):
                self.allocate_samples[ind]["%d" % label].append(q)
        if return_key:
            return pred, index
        return pred


class Simulator:
    def __init__(self, in_fea, keys, masks, values, scores, num_classes=2):
        self.switch = Switch(
            in_fea, keys, masks, values, scores, num_classes=num_classes
        )
        self.switch = self.switch.cuda()
        self.switch.eval()
        self.num_rules = len(keys)

    def validate(self, loader, num_drop=0, cache=False):
        top1_switch = AverageMeter("Switch Acc@1", ":6.2f")
        batch_time = AverageMeter("Tps", ":6.3f")
        progress = ProgressMeter(len(loader), [batch_time], prefix="Validate: ")
        allocate_gt_y = []
        allocate_pred_y = []

        with torch.no_grad():
            end = time.time()
            for i, batch in enumerate(loader):
                allocate_gt_y += batch[1].numpy().tolist()
                X = batch[0].cuda()
                y = batch[1].cuda()
                pred_label = (
                    self.switch((X + 1) / 2, y, num_drop=num_drop, cache=cache)
                    .cpu()
                    .numpy()
                    .flatten()
                )
                allocate_pred_y += pred_label.tolist()
                n = X.size(0)
                acc = accuracy_score(batch[1].numpy().flatten(), pred_label) * 100
                top1_switch.update(acc, n)
                torch.cuda.synchronize()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 100 == 0:
                    progress.display(i)
        return top1_switch.avg, classification_report(
            allocate_gt_y, allocate_pred_y, digits=8
        )

    def save_allocate_samples(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.switch.allocate_samples, f)

    def check_allocate_samples(self):
        num_sample = []
        for i, samples in enumerate(self.switch.allocate_samples):
            n = np.sum([len(v) for v in samples.values()])
            if n < 2:
                print("rule %d, no samples" % i, n)
            else:
                num_sample.append(n)
        print("average number of samples: %f" % np.mean(num_sample))
        print("max number of samples: %d" % np.max(num_sample))
        print("min number of samples: %d" % np.min(num_sample))

    def search(self, loader, search_step_ratio=0.1, delta=0.01):
        step = int(self.num_rules * search_step_ratio)
        num_rules = self.num_rules
        rules, top1 = [], []
        best_acc = 0
        searched_rules = 0
        while num_rules > 0:
            num_drop = self.num_rules - num_rules
            acc, _ = self.validate(loader, num_drop=num_drop)
            if acc > best_acc:
                best_acc = acc
                searched_rules = num_rules
            elif best_acc - acc <= delta:
                searched_rules = num_rules
            if best_acc - acc > delta and searched_rules != 0:
                break

            rules.append(num_rules)
            top1.append(acc)
            num_rules -= step
        # plot the result
        plt.plot(rules, top1)
        plt.xlabel("Number of rules")
        plt.ylabel("Accuracy")
        plt.savefig("search.png")

        # save as csv
        with open("search.csv", "w") as f:
            for r, t in zip(rules, top1):
                f.write("%d, %f\n" % (r, t))

        return searched_rules

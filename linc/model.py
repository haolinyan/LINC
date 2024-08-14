import os
import torch
import torch.nn as nn
from loguru import logger
from .ops import EnsembleNeuralNetworkUnits
from .utils import (
    setup_seed,
    AverageMeter,
    ProgressMeter,
    accuracy,
    generate_rules,
    merge_neighbor_mask,
    toInt,
    rankdata,
)
from .simulator import Simulator
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import datetime
import time

use_lab = os.environ.get("LINC_LAB_UTIL", False)
if use_lab == "swanlab":
    import swanlab as lab

    assert lab is not None, "Please install swanlab to manager the experiments."
elif use_lab == "wandb":
    import wandb as lab


class EnsembleNNModel(nn.Module):
    def __init__(self, num_classes, in_features, k, T, E):
        super(EnsembleNNModel, self).__init__()
        self.ensemble_nnus = nn.ModuleList(
            [
                EnsembleNeuralNetworkUnits(in_features, k, T, E)
                for _ in range(num_classes)
            ]
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out = []
        for nnus in self.ensemble_nnus:
            out.append(nnus(x))
        return torch.concat(out, dim=-1)

    def loss(self, X, y):
        logits = self.forward(X)
        return logits, self.criterion(logits, y)


class LINCModel:
    def __init__(
        self,
        dataset,
        num_classes,
        in_features,
        k,
        T,
        E,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=32,
        num_epochs=32,
        save_dir=None,
        save_best_only=True,
        split=1,
        data=None,
    ):
        now = datetime.datetime.now()
        formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.num_classes = num_classes
        self.in_features = in_features
        self.k = k
        self.T = T
        self.E = E
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_dir = (
            save_dir if save_dir is not None else "checkpoints/{}".format(formatted_now)
        )
        self.split = split
        self.data = data
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        logger.add(os.path.join(self.save_dir, "runtime.log"))
        logger.info("------- LINC Config -------\n{}".format(self.__dict__))
        if use_lab:
            lab.init(project="LINC", config=self.__dict__)
        setup_seed(2023)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        logger.info("Using device: {}".format(self.device))
        self.dataset = dataset
        self.save_best_only = save_best_only
        self._build_model()
        self._build_dataset()
        self.training_dur = 0
        self.conversion_dur = 0

    def _build_model(self):
        self.model = EnsembleNNModel(
            self.num_classes, self.in_features, self.k, self.T, self.E
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        logger.info("------- Building Model -------\n{}".format(self.model))

    def _build_dataset(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        self.train_data = self.dataset(self.data, split=self.split, key="train")
        self.valid_data = self.dataset(self.data, key="val")
        self.test_data = self.dataset(self.data, key="test")
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=3,
        )
        self.valid_loader = DataLoader(
            self.valid_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=3,
        )
        self.test_loader = DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=3,
        )

    def _save_checkpoint(self, epoch, best_valid_acc=None):
        if best_valid_acc is None:
            save_path = os.path.join(self.save_dir, "epoch_{}.pth".format(epoch))
            torch.save(self.model.state_dict(), save_path)
        else:
            save_path = os.path.join(self.save_dir, "best.pth")
            torch.save(self.model.state_dict(), save_path)

    def train(self):
        begin = time.time()
        best_valid_acc = 0
        start_epoch = 0
        for epoch in range(start_epoch, self.num_epochs):
            train_loss, train_acc = self._train_epoch(epoch)
            valid_loss, valid_acc = self._valid_epoch(epoch)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                if self.save_best_only:
                    self._save_checkpoint(epoch, best_valid_acc)
                    logger.info(
                        "Save the best checkpoint. Acc@1: {:.4f}".format(best_valid_acc)
                    )
                else:
                    self._save_checkpoint(epoch)
            if use_lab:
                lab.log(
                    {
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "valid_loss": valid_loss,
                        "valid_acc": valid_acc,
                        "best_valid_acc": best_valid_acc,
                    },
                    step=epoch,
                )
            logger.info(
                "Epoch: {}/{}, Train Loss: {:.4f}, Train Acc: {:.4f}, Valid Loss: {:.4f}, Valid Acc: {:.4f}".format(
                    epoch + 1,
                    self.num_epochs,
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc,
                )
            )
        self.training_dur = time.time() - begin
        logger.info("Best Valid Acc: {:.4f}".format(best_valid_acc))
        self._load_model()
        cls_report = self._test_epoch()
        logger.info("Classification Report (Test): \n{}".format(cls_report))

    def _load_model(self):
        ckpt = torch.load(os.path.join(self.save_dir, "best.pth"))
        self.model.load_state_dict(ckpt)

    def _valid_epoch(self, epoch, loader=None):
        batch_time = AverageMeter("Tps", ":6.3f")
        loss_value = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        self.model.eval()
        if loader is None:
            loader = self.valid_loader
        progress = ProgressMeter(
            len(loader), [batch_time, loss_value, top1], prefix="Valid: "
        )
        end = time.time()
        num_batch = len(loader)
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                logits, loss = self.model.loss(x, y)
                acc = accuracy(logits, y)
                loss_value.update(loss.item(), x.size(0))
                top1.update(acc[0].item(), x.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
                if i % (num_batch // 1) == 0:
                    progress.display(i)
        return loss_value.avg, top1.avg

    def _test_epoch(self):
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                y_true += y.numpy().tolist()
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                y_pred += torch.argmax(logits, dim=-1).cpu().numpy().tolist()

        cls_report = classification_report(y_true, y_pred, digits=4)
        return cls_report

    def _train_epoch(self, epoch):
        batch_time = AverageMeter("Tps", ":6.3f")  # Time per step
        data_time = AverageMeter("Data", ":6.3f")
        loss_value = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        self.model.train()
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, loss_value, top1],
            prefix="Epoch: [{}] ".format(epoch),
        )

        end = time.time()
        self.optimizer.zero_grad()
        num_batch = len(self.train_loader)

        for i, (x, y) in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            x, y = x.to(self.device), y.to(self.device)
            logits, loss = self.model.loss(x, y)
            loss.backward()
            self.optimizer.step()
            acc = accuracy(logits, y, topk=(1,))
            n = x.size(0)
            loss_value.update(loss.item(), n)
            top1.update(acc[0].item(), n)
            self.optimizer.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()
            if i % (num_batch // 10) == 0:
                progress.display(i)
        return loss_value.avg, top1.avg

    def export(
        self,
        search_plot=False,
        search_step_ratio=0.1,
        delta=0.01,
        out="rules.txt",
        export_cache="cache.pkl",
        split=20,
    ):
        out = os.path.join(self.save_dir, out)
        begin = time.time()
        self._load_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.split = split
        self._build_dataset(batch_size=1024)
        keys, masks, values, scores = generate_rules(self.model, self.train_data)
        logger.info("Stage1: Export Num rules: %d" % len(keys))
        min_num_rules = len(keys)
        round = 0
        while True:
            keys, scores, masks, values = merge_neighbor_mask(
                keys, scores, masks, values
            )
            logger.info("Stage2-Round%d: Export Num rules: %d" % (round, len(keys)))
            round += 1
            if len(keys) == min_num_rules:
                break
            else:
                min_num_rules = len(keys)
        simulator = Simulator(
            len(keys[0]), keys, masks, values, scores, num_classes=self.num_classes
        )
        acc, report = simulator.validate(self.test_loader)
        logger.info("Classification Report: \n{}".format(report))
        if search_plot:
            searched_rules = simulator.search(
                self.valid_loader, search_step_ratio=search_step_ratio, delta=delta
            )
            num_drop = len(keys) - searched_rules
            logger.info(
                "Num rules: %d, Num rules (Searched): %d" % (len(keys), searched_rules)
            )
            _, report = simulator.validate(self.test_loader, num_drop=num_drop)
            logger.info("Classification Report (Searched): \n{}".format(report))

        scores = rankdata(-1 * scores)
        rule_string = ""
        for m, k, v, s in zip(masks, keys, values, scores):
            rule_string += "%d %d %d %d\n" % (toInt(m), toInt(k), v, s)
        with open(out, "w") as f:
            f.write(rule_string[:-1])

        if export_cache is not None:
            simulator.validate(self.valid_loader, cache=True)
            simulator.save_allocate_samples(os.path.join(self.save_dir, export_cache))
        self.conversion_dur = time.time() - begin
        logger.info("Export finished, total rules: %d" % len(keys))
        if use_lab:
            lab.log({"entries": len(keys), "test_acc": acc})
        return acc, len(keys)

    def finish(self):
        logger.info("------- Training finished -------")
        logger.info("Training time: %d s" % self.training_dur)
        logger.info("Conversion time: %d s" % self.conversion_dur)
        if use_lab:
            lab.finish()


if __name__ == "__main__":
    num_classes = 10
    in_features = 10
    k = 4
    T = 1.0
    E = 8
    model = EnsembleNNModel(num_classes, in_features, k, T, E)
    X = torch.randn(32, in_features)
    y = torch.randint(0, num_classes, (32,))
    logits, loss = model.loss(X, y)
    loss.backward()

from linc import LINCDataset, LINCModel
import argparse
import numpy as np
import pickle

class LINCDataset_(LINCDataset):
    def __init__(self, data_path, split=1, key="train"):
        super().__init__(data_path, split, key)
    def load_data(self, data_path, key):
        subset = [0, 1, 2]
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        X, y = data[key]
        mask = np.isin(y, subset)
        X = X[mask]
        y = y[mask]
        return X, y
    

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--in_features", type=int, default=96)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--T", type=float, default=0.2219338905926658)
    parser.add_argument("--E", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01700159092363334)
    parser.add_argument("--weight_decay", type=float, default=0.005160903944914268)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--num_epochs", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="checkpoints/iscx_c3")
    parser.add_argument("--save_best_only", type=bool, default=True)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--data", type=str, default="iscx.pkl")
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parser_args()
    trainer = LINCModel(LINCDataset_, **vars(args))
    trainer.train()
    trainer.export()
    trainer.finish()


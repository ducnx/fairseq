"""
Modify vq-wav2vec_featurize to use mimic3 pkl dataset

Returns:
    [type]: [description]
"""

import pickle
import pprint
import os
import argparse
import torch
import fairseq
import pandas as pd
from icecream import ic
from torch import nn
from torch.utils.data import DataLoader


class ArgTypes:
    @staticmethod
    def existing_path(arg):
        arg = str(arg)
        assert os.path.exists(arg), f"File {arg} does not exist"
        return arg

    @staticmethod
    def mkdir(arg):
        arg = str(arg)
        os.makedirs(arg, exist_ok=True)
        return arg


class PKL_Dataset:
    def __init__(self, datapath, labelpath):
        self.data = torch.FloatTensor(pickle.load(open(datapath, 'rb')))
        # ic(type(self.data), self.data.shape)
        df = pd.read_pickle(open(labelpath, 'rb'))
        self.labels = df[['mort_hosp', 'los_3']]
        self.labels.replace({False: 0, True: 1}, inplace=True)
        # ic(type(self.labels), self.labels.shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.labels.iloc[index]

    def collate(self, batch):
        return batch


class PKL_DatasetWriter():
    def __init__(self):
        self.args = self.load_config()
        pprint.pprint(self.args.__dict__)

        self.model = self.load_model()

    def __getattr__(self, attr):
        return getattr(self.args, attr)

    def load_config(self):

        parser = argparse.ArgumentParser("Vector Quantized wav2vec features")

        # Model Arguments
        parser.add_argument("--checkpoint", type=ArgTypes.existing_path, required=True)
        parser.add_argument("--data-parallel", action="store_true")

        # Output Arguments
        parser.add_argument("--output-dir", type=ArgTypes.mkdir, required=True)

        # Data Arguments
        parser.add_argument("--data-dir", type=ArgTypes.existing_path, required=True)
        parser.add_argument("--splits", type=str, nargs="+", required=True)
        parser.add_argument("--extension", type=str, required=True)
        parser.add_argument("--labels", type=str, required=False)

        parser.add_argument("--shard", type=int, default=None)
        parser.add_argument("--num-shards", type=int, default=None)
        parser.add_argument("--max-size", type=int, default=1300000)

        # Logger Arguments
        parser.add_argument(
            "--log-format", type=str, choices=["none", "simple", "tqdm"]
        )

        return parser.parse_args()

    def label_file(self, name):
        shard_part = "" if self.args.shard is None else f".{self.args.shard}"
        return os.path.join(self.output_dir, f"{name}.lbl{shard_part}")

    def data_file(self, name):
        shard_part = "" if self.args.shard is None else f".{self.args.shard}"
        return os.path.join(self.output_dir, f"{name}.src{shard_part}")

    def var_file(self):
        return os.path.join(self.output_dir, f"vars.pt")

    def iterate(self, loader):
        for batch in loader:
            for x, y in batch:
                x = x.unsqueeze(0).float().cuda()
                div = 1
                while x.size(-1) // div > self.args.max_size:
                    div += 1
                xs = x.chunk(div, dim=-1)

                result = []
                for x in xs:
                    torch.cuda.empty_cache()
                    x = self.model.feature_extractor(x)
                    if self.quantize_location == "encoder":
                        with torch.no_grad():
                            _, idx = self.model.vector_quantizer.forward_idx(x)
                            # ic(idx)
                            idx = idx.squeeze(0).cpu()
                    else:
                        with torch.no_grad():
                            z = self.model.feature_aggregator(x)
                            _, idx = self.model.vector_quantizer.forward_idx(z)
                            idx = idx.squeeze(0).cpu()
                    result.append(idx)
                idx = torch.cat(result, dim=0)
                yield " ".join("-".join(map(str, a.tolist())) for a in idx)

    def process_splits(self):

        for split in self.splits:
            print(split)

            if self.extension == "pkl":
                datapath = os.path.join(self.data_dir, f'{split}.{self.extension}')
                labelpath = os.path.join(self.data_dir, f'{split}_label.{self.extension}')
                loader = self.load_data(datapath, labelpath)
                with open(self.data_file(split), "w") as srcf:
                    for line in self.iterate(loader):
                        print(line, file=srcf)

    def load_data(self, datapath, labelpath):
        dataset = PKL_Dataset(datapath, labelpath)
        loader = DataLoader(
            dataset, batch_size=64, collate_fn=dataset.collate, num_workers=8
        )
        return loader

    def load_model(self):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.checkpoint])
        model = model[0]

        self.quantize_location = getattr(cfg.model, "vq", "encoder")

        model.eval().float()
        model.cuda()

        if self.data_parallel:
            model = nn.DataParallel(model)

        return model

    def __call__(self):

        self.process_splits()

        if hasattr(self.model.feature_extractor, "vars") and (
            self.args.shard is None or self.args.shard == 0
        ):
            vars = (
                self.model.feature_extractor.vars.view(
                    self.model.feature_extractor.banks,
                    self.model.feature_extractor.num_vars,
                    -1,
                )
                .cpu()
                .detach()
            )
            print("writing learned latent variable embeddings: ", vars.shape)
            torch.save(vars, self.var_file())


if __name__ == '__main__':
    write_dataset = PKL_DatasetWriter()
    write_dataset()

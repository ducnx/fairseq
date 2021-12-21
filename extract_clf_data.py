"""
Extract classification data
python -m extract_clf_data
"""
import torch
import pickle
import os
from icecream import ic
from tqdm import tqdm
from fairseq.models.roberta import RobertaModel


def main():
    datadir = '/home/dxng/datasets/MIMIC-III/processed_48h_percent0.05'
    # ckpt = torch.load('/home/dxng/runs/mimic3_roberta/3layers_6heads_embed192_update40000_lr0.0005/checkpoint_best.pt')
    # ic(ckpt.keys())
    # ic(ckpt['args'])
    # roberta = RobertaModel.build_model(ckpt['args'], task=None)
    # roberta.load_state_dict(ckpt['model'])
    # roberta.eval()
    roberta = RobertaModel.from_pretrained(
        '/home/dxng/runs/mimic3_roberta/3layers_6heads_embed192_update40000_lr0.0005',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=datadir,
        bpe=None
    )
    roberta.eval()

    splits = ['train', 'valid', 'test']
    # split = 'test'
    for split in splits:
        # Features
        feats = []
        with open(os.path.join(datadir, f'{split}.src')) as fin:
            for idx, line in tqdm(enumerate(fin)):
                line = line.strip()
                # tokens = roberta.encode(line)
                tokens = roberta.task.source_dictionary.encode_line(line, append_eos=False, add_if_not_exist=False)
                last_layer_features = roberta.extract_features(tokens)
                avg_features = last_layer_features.mean(1).squeeze()
                feats.append(avg_features.clone())
                # if idx > 100:
                #     break
        feats = torch.stack(feats, dim=0)
        ic(feats.shape)
        pickle.dump(feats, open(os.path.join(datadir, f'clf_full/{split}.pkl'), 'wb'))


if __name__ == '__main__':
    main()

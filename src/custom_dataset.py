import os
import torch
import pickle
import json
from itertools import chain
from tqdm import tqdm
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, prefix, args):
        self.args = args
        assert prefix == args.train_prefix or prefix == args.valid_prefix
        
        data_path = os.path.join(args.data_dir, f"multi_{prefix}_data.pkl")
        context_path = os.path.join(args.data_dir, f"context_label_{prefix}_data.pkl")
        
        print(f"Loading main data from {data_path}...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            # Using slicing for debugging. For a full run, remove the '[:1]'
            texts, videos, audios, targets = data["txt"][:1], data["img"][:1], data["aud"][:1], data["label"][:1]

        print(f"Loading context and emotion labels from {context_path}...")
        with open(context_path, 'rb') as f:
            context_label = pickle.load(f)
            # Using slicing for debugging. For a full run, remove the '[:1]'
            contexts_data = context_label["context"][:1]
            emotion_labels_data = context_label["label"][:1]

        self.input_ids = []
        self.token_type_ids = []
        self.labels = []
        self.imgs = []
        self.auds = []
        self.contexts = []
        self.emotion_labels = []

        for i in tqdm(range(len(texts)), desc=f"Processing {prefix} data"):
            dialogue_texts = texts[i]
            dialogue_targets = targets[i]
            dialogue_contexts = contexts_data[i]
            dialogue_emotion_labels = emotion_labels_data[i]
            
            # Ensure all parts of a dialogue have the same length
            assert len(dialogue_texts) == len(dialogue_targets) == len(dialogue_contexts) == len(dialogue_emotion_labels)

            for j in range(len(dialogue_texts)):
                utterance_tokens = dialogue_texts[j]
                input_ids_list = list(chain.from_iterable(utterance_tokens))

                if len(input_ids_list) >= 1024:
                    continue

                sp1_id, sp2_id = args.sp1_id, args.sp2_id
                current_token_types = [[sp1_id] * len(ctx) if c % 2 == 0 else [sp2_id] * len(ctx) for c, ctx in enumerate(utterance_tokens)]
                current_token_types_list = list(chain.from_iterable(current_token_types))
                assert len(input_ids_list) == len(current_token_types_list)
                
                current_lm_target = dialogue_targets[j]
                current_lm_labels = current_lm_target[2:-2] + [args.eos_id] # Slice to remove special tokens

                len_gap = len(input_ids_list) - len(current_lm_labels)
                if len_gap > 0:
                    current_lm_labels = [-100] * len_gap + current_lm_labels
                elif len_gap < 0:
                    gap_to_add = abs(len_gap)
                    input_ids_list.extend([args.eos_id] * gap_to_add)
                    current_token_types_list.extend([current_token_types_list[-1]] * gap_to_add)

                assert len(input_ids_list) == len(current_lm_labels)

                self.input_ids.append(input_ids_list)
                self.token_type_ids.append(current_token_types_list)
                self.labels.append(current_lm_labels)
                self.emotion_labels.append(dialogue_emotion_labels[j])
                
                v_tmp = [videos[i][0]] * len(input_ids_list)
                a_tmp = [audios[i][0]] * len(input_ids_list)
                self.imgs.append(v_tmp)
                self.auds.append(a_tmp)
                self.contexts.append(dialogue_contexts[j])
        
        assert len(self.input_ids) == len(self.token_type_ids) == len(self.labels) == \
               len(self.imgs) == len(self.auds) == len(self.contexts) == len(self.emotion_labels)
        
        print(f"Finished processing. Total samples: {len(self.input_ids)}")

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return (
            self.input_ids[index],
            self.token_type_ids[index],
            self.labels[index],
            self.imgs[index],
            self.auds[index],
            self.contexts[index],
            self.emotion_labels[index],
        )

class PadCollate():
    def __init__(self, eos_id, args):
        self.args = args
        self.eos_id = eos_id

    def pad_collate(self, batch):
        input_ids, token_type_ids, labels, imgs, auds, contexts, emotion_labels = [], [], [], [], [], [], []

        for seqs in batch:
            input_ids.append(torch.LongTensor(seqs[0]))
            token_type_ids.append(torch.LongTensor(seqs[1]))
            labels.append(torch.LongTensor(seqs[2]))
            imgs.append(seqs[3]) 
            auds.append(seqs[4])
            contexts.append(seqs[5]) 
            emotion_labels.append(seqs[6])

        # Pad the sequence-like tensors
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.eos_id)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=self.eos_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return (
            input_ids,
            token_type_ids,
            labels,
            imgs, 
            auds,
            contexts, 
            emotion_labels
        )

import os
import sys
import argparse
import copy
import math
import random
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

from tqdm import tqdm
from transformers import (
    GPT2Tokenizer,
    get_polynomial_decay_schedule_with_warmup,
)

from model import * 
from custom_dataset import *
from eval import Evaluator

def print_custom(context, ref, sentence):
    """Formats the context, reference, and generated sentence for printing."""
    res = ""
    res += f"Context: {context}\n"
    res += f"GPT-2: {sentence}\n"
    res += f"Ref: {ref}\n"
    res += "---------------------------------------------------------------\n"
    return res


class Manager:
    def __init__(self, args):
        self.args = args

        if torch.cuda.is_available():
            self.args.device = torch.device(f"cuda:{self.args.gpu}")
        else:
            self.args.device = torch.device("cpu")

        print("Loading the tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model_type)
        special_tokens = {
            "bos_token": self.args.bos_token,
            "additional_special_tokens": [self.args.sp1_token, self.args.sp2_token],
        }
        self.args.eos_token = self.tokenizer.eos_token
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.args.bos_id = vocab[self.args.bos_token]
        self.args.eos_id = vocab[self.args.eos_token]
        self.args.sp1_id = vocab[self.args.sp1_token]
        self.args.sp2_id = vocab[self.args.sp2_token]
        
        print("Loading the model...")
        self.fix_seed(self.args.seed)
        self.model = GPT2LMHeadModel.from_pretrained(self.args.model_type).to(self.args.device)
        self.model.resize_token_embeddings(self.args.vocab_size)
        self.args.max_len = min(self.args.max_len, self.model.config.n_ctx)

        if self.args.mode in ["train", "infer"]:
            print("Loading the optimizer...")
            self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
            self.best_ppl = sys.float_info.max
            self.last_epoch = 0

            print("Loading train & valid data...")
            train_set = CustomDataset(self.args.train_prefix, self.args)
            valid_set = CustomDataset(self.args.valid_prefix, self.args)
            
            ppd = PadCollate(eos_id=self.args.eos_id, args=self.args)

            self.train_loader = DataLoader(
                train_set, collate_fn=ppd.pad_collate, shuffle=True,
                batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
            )
            self.valid_loader = DataLoader(
                valid_set, collate_fn=ppd.pad_collate,
                batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
            )

            os.makedirs(self.args.ckpt_dir, exist_ok=True)

            num_batches = len(self.train_loader)
            args.total_train_steps = args.num_epochs * num_batches
            args.warmup_steps = int(args.warmup_ratio * args.total_train_steps)

            self.sched = get_polynomial_decay_schedule_with_warmup(
                self.optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_train_steps, power=2,
            )
            self.writer = SummaryWriter()

        if self.args.ckpt_name is not None:
            ckpt_path = os.path.join(self.args.ckpt_dir, f"{self.args.ckpt_name}.ckpt")
            if os.path.exists(ckpt_path):
                print("Loading the trained checkpoint...")
                ckpt = torch.load(ckpt_path, map_location=self.args.device)
                self.model.load_state_dict(ckpt["model_state_dict"], strict=False) # Use strict=False to handle the new emotion_head

                if self.args.mode == "train":
                    print(f"Training will resume from checkpoint: {self.args.ckpt_name}.ckpt")
                    self.optim.load_state_dict(ckpt["optim_state_dict"])
                    self.sched.load_state_dict(ckpt["sched_state_dict"])
                    self.best_ppl = ckpt.get('ppl', sys.float_info.max) # Load best ppl if available
                    self.last_epoch = ckpt["epoch"]
                else:
                    print("Inference will start with the specified checkpoint.")
            else:
                print(f"Cannot find the specified checkpoint: {ckpt_path}")
                if self.args.mode == "train":
                    print("Training will start with an initialized model.")
                else:
                    print("Cannot run inference without a valid checkpoint.")
                    exit()

        print("Setting finished.")

    def train(self):
        self.fix_seed(self.args.seed)
        print("Training starts.")
        start_epoch = self.last_epoch + 1
        
        for epoch in range(start_epoch, start_epoch + self.args.num_epochs):
            self.model.train()
            print(f"-" * 35 + f"Epoch: {epoch}" + "-" * 35)
            
            train_total_losses = []
            train_lm_losses = [] # For PPL calculation
            train_correct_emotions = 0
            train_total_emotions = 0

            for i, batch in enumerate(tqdm(self.train_loader)):
                input_ids, token_type_ids, lm_labels, imgs, auds, contexts, emotion_labels = batch
                
                input_ids, token_type_ids, lm_labels, emotion_labels = (
                    input_ids.to(self.args.device),
                    token_type_ids.to(self.args.device),
                    lm_labels.to(self.args.device),
                    torch.LongTensor(emotion_labels).to(self.args.device), # Ensure emotion labels are tensors
                )
                
                outputs = self.model(
                    input_ids=input_ids, token_type_ids=token_type_ids, 
                    labels=lm_labels, emotion_labels=emotion_labels
                )
                
                loss = outputs.loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.sched.step()
                
                train_total_losses.append(loss.item())
                
                with torch.no_grad():
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = lm_labels[..., 1:].contiguous()
                    loss_fct_lm = nn.CrossEntropyLoss()
                    lm_loss = loss_fct_lm(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    train_lm_losses.append(lm_loss.item())

                    preds = torch.argmax(outputs.emotion_logits, dim=-1)
                    train_correct_emotions += (preds == emotion_labels).sum().item()
                    train_total_emotions += emotion_labels.size(0)

            avg_train_loss = np.mean(train_total_losses)
            avg_lm_loss = np.mean(train_lm_losses)
            train_ppl = math.exp(avg_lm_loss)
            train_acc = (train_correct_emotions / train_total_emotions) * 100

            print(f"Train Loss: {avg_train_loss:.4f} | Train PPL: {train_ppl:.4f} | Train Emotion Acc: {train_acc:.2f}%")
            self.writer.add_scalar("Loss/train", avg_train_loss, epoch)
            self.writer.add_scalar("PPL/train", train_ppl, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)

            self.last_epoch += 1
            valid_loss, valid_ppl, valid_acc = self.validation()

            if valid_ppl < self.best_ppl:
                self.best_ppl = valid_ppl
                state_dict = {
                    "model_state_dict": self.model.state_dict(),
                    "optim_state_dict": self.optim.state_dict(),
                    "sched_state_dict": self.sched.state_dict(),
                    "ppl": self.best_ppl,
                    "epoch": self.last_epoch,
                }
                save_path = os.path.join(self.args.ckpt_dir, f"best_ckpt_epoch={epoch}_valid_ppl={self.best_ppl:.4f}.ckpt")
                torch.save(state_dict, save_path)
                print("*" * 10 + " Current best checkpoint is saved. " + "*" * 10)
                print(save_path)

            print(f"Best valid PPL: {self.best_ppl:.4f}")
            print(f"Current valid loss: {valid_loss:.4f} | Current valid PPL: {valid_ppl:.4f} | Current valid Emotion Acc: {valid_acc:.2f}%")
            self.writer.add_scalar("Loss/valid", valid_loss, epoch)
            self.writer.add_scalar("PPL/valid", valid_ppl, epoch)
            self.writer.add_scalar("Accuracy/valid", valid_acc, epoch)

        print("Training finished!")

    def validation(self):
        print("Validation processing...")
        self.model.eval()
        
        valid_total_losses = []
        valid_lm_losses = []
        valid_correct_emotions = 0
        valid_total_emotions = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, token_type_ids, lm_labels, imgs, auds, contexts, emotion_labels = batch
                
                input_ids, token_type_ids, lm_labels, emotion_labels = (
                    input_ids.to(self.args.device),
                    token_type_ids.to(self.args.device),
                    lm_labels.to(self.args.device),
                    torch.LongTensor(emotion_labels).to(self.args.device),
                )

                outputs = self.model(
                    input_ids=input_ids, token_type_ids=token_type_ids, 
                    labels=lm_labels, emotion_labels=emotion_labels
                )
                
                valid_total_losses.append(outputs.loss.item())
                
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = lm_labels[..., 1:].contiguous()
                loss_fct_lm = nn.CrossEntropyLoss()
                lm_loss = loss_fct_lm(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                valid_lm_losses.append(lm_loss.item())

                preds = torch.argmax(outputs.emotion_logits, dim=-1)
                valid_correct_emotions += (preds == emotion_labels).sum().item()
                valid_total_emotions += emotion_labels.size(0)

        avg_valid_loss = np.mean(valid_total_losses)
        avg_lm_loss = np.mean(valid_lm_losses)
        valid_ppl = math.exp(avg_lm_loss)
        valid_acc = (valid_correct_emotions / valid_total_emotions) * 100
            
        if math.isnan(valid_ppl):
            valid_ppl = 1e8

        return avg_valid_loss, valid_ppl, valid_acc
    
    def nucleus_sampling(self, input_ids, token_type_ids, input_len):
        output_ids = []
        for pos in range(input_len, self.args.max_len):
            outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
            next_token_logits = outputs.logits[:, pos - 1, :]
            
            probs = F.softmax(next_token_logits, dim=-1)
            
            sorted_probs, sorted_idxs = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            idx_remove = cumsum_probs > self.args.top_p
            idx_remove[:, 1:] = idx_remove[:, :-1].clone()
            idx_remove[:, 0] = False
            sorted_probs[idx_remove] = 0.0
            sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)
            
            probs = torch.zeros(probs.shape, device=self.args.device).scatter_(-1, sorted_idxs, sorted_probs)
            idx = torch.multinomial(probs, 1)
            idx_item = idx.squeeze(-1).squeeze(-1).item()
            output_ids.append(idx_item)

            if idx_item == self.args.eos_id:
                break
            
            input_ids = torch.cat((input_ids, idx), dim=-1)
            next_type_id = torch.LongTensor([[self.args.sp2_id]]).to(self.args.device)
            token_type_ids = torch.cat((token_type_ids, next_type_id), dim=-1)
            assert input_ids.shape == token_type_ids.shape

        return output_ids

    def fix_seed(self, seed):
        """Sets a random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    def test(self):
        print("Test processing: Collecting generated texts and references...")
        self.model.eval()
        self.fix_seed(self.args.seed)
        
        all_hypotheses = []
        all_references = []
        all_true_labels = []
        all_losses = [] # For overall test PPL
        
        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, token_type_ids, lm_labels, imgs, auds, contexts, emotion_labels = batch
                
                input_ids, token_type_ids, lm_labels, emotion_labels = (
                    input_ids.to(self.args.device),
                    token_type_ids.to(self.args.device),
                    lm_labels.to(self.args.device),
                    torch.LongTensor(emotion_labels).to(self.args.device),
                )
                
                for i in range(input_ids.size(0)):
                    current_input = input_ids[i].unsqueeze(0)
                    current_token_types = token_type_ids[i].unsqueeze(0)
                    
                    input_len = (current_input != self.args.eos_id).sum().item()

                    output_ids = self.nucleus_sampling(current_input[:, :input_len], current_token_types[:, :input_len], input_len)
                    hypothesis_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    all_hypotheses.append(hypothesis_text)

                    ref_ids = lm_labels[i][lm_labels[i] != -100] # Filter out padding
                    reference_text = self.tokenizer.decode(ref_ids, skip_special_tokens=True)
                    all_references.append(reference_text)
                    
                    all_true_labels.append(emotion_labels[i].item())

                outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, labels=lm_labels)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = lm_labels[..., 1:].contiguous()
                loss_fct_lm = nn.CrossEntropyLoss()
                lm_loss = loss_fct_lm(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                all_losses.append(lm_loss.item())

        return all_hypotheses, all_references, all_true_labels, all_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="The random seed.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer"], help="The running mode: train or infer.")
    parser.add_argument("--data_dir", type=str, default="data", help="The parent directory where data files are stored.")
    parser.add_argument("--train_prefix", type=str, default="train", help="The prefix of the train data files' name.")
    parser.add_argument("--valid_prefix", type=str, default="valid", help="The prefix of the validation data files' name.")
    parser.add_argument("--model_type", type=str, default="gpt2", help="The model type of GPT-2.")
    parser.add_argument("--bos_token", type=str, default="<bos>", help="The BOS token.")
    parser.add_argument("--sp1_token", type=str, default="<sp1>", help="The speaker1 token.")
    parser.add_argument("--sp2_token", type=str, default="<sp2>", help="The speaker2 token.")
    parser.add_argument("--gpu", type=str, default="0", help="The index of GPU to use.")
    parser.add_argument("--lr", type=float, default=2e-5, help="The learning rate.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="The ratio of warmup steps to the total training steps.")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="The number of workers for data loading.")
    parser.add_argument("--num_epochs", type=int, default=100, help="The number of total epochs.")
    parser.add_argument("--max_len", type=int, default=1024, help="The maximum length of input sequence.")
    parser.add_argument("--max_turns", type=int, default=10, help="The maximum number of dialogue histories to include.")
    parser.add_argument("--top_p", type=float, default=0.95, help="The top-p value for nucleus sampling decoding.")
    parser.add_argument("--ckpt_dir", type=str, default="saved_models", help="The directory name for saved checkpoints.")
    parser.add_argument("--ckpt_name", type=str, default=None, help="The name of the trained checkpoint (without extension).")
    
    args = parser.parse_args()

    args.data_dir = os.path.join(args.data_dir, args.model_type)
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.model_type)
    
    if args.mode == 'train':
        manager = Manager(args)
        manager.train()
    elif args.mode == 'infer':
        assert args.ckpt_name is not None, "Please specify the trained model checkpoint using --ckpt_name."
        manager = Manager(args)
        
        hypotheses, references, true_labels, losses = manager.test()
        
        evaluator = Evaluator(device=manager.args.device)
        
        final_metrics = evaluator.evaluate_all(
            hypotheses=hypotheses,
            references=references,
            true_label_ids=true_labels,
            losses=losses
        )

        print("\n--- Final Evaluation Results ---")
        for metric, value in final_metrics.items():
            if isinstance(value, float):
                print(f"{metric.upper():<10}: {value:.4f}")
            else:
                print(f"{metric.upper():<10}: {value}")
        print("--------------------------------")
        
        results_file_path = os.path.join(args.data_dir, f"{args.ckpt_name}_evaluation_results.txt")
        with open(results_file_path, "w", encoding="utf-8") as f:
            for metric, value in final_metrics.items():
                f.write(f"{metric}: {value}\n")
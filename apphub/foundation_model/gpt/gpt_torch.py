# Copyright 2023 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import fastestimator as fe
from fastestimator.dataset.data import wikitext_103
from fastestimator.op.numpyop import LambdaOp, NumpyOp
from fastestimator.op.tensorop import LambdaOp as TLambdaOp
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver


class TextDataset(Dataset):
    def __init__(self, file_path, num_chars=5000):
        super().__init__()
        self.texts = self._read_file(file_path)
        self.num_chars = num_chars

    @staticmethod
    def _read_file(path):
        text = ''.join(pd.read_parquet(path, engine='fastparquet')['text'].to_list())
        return text

    def __len__(self):
        # this is just a placeholder, we use 'train_steps_per_epoch' to control training length
        return 10000

    def __getitem__(self, idx):
        start_idx = random.randint(0, len(self.texts) - self.num_chars - 1)
        random_text = self.texts[start_idx:start_idx + self.num_chars]
        return {"x": random_text[random_text.index(" ") + 1:]}  # always start from a new word


class Encode(NumpyOp):
    def __init__(self, tokenizer, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.tokenizer = tokenizer

    def forward(self, data, state):
        return np.array(self.tokenizer(data, truncation=True)['input_ids'])


class MultiHeadAttention(nn.Module):
    # Multi-head attention is like group convolution, but for attention.
    def __init__(self, context_len, em_dim, num_heads=4, p_drop=0.2, use_mask=True):
        super().__init__()
        self.num_heads = num_heads
        self.use_mask = use_mask
        self.key = nn.Linear(em_dim, em_dim, bias=False)
        self.query = nn.Linear(em_dim, em_dim, bias=False)
        self.value = nn.Linear(em_dim, em_dim, bias=False)
        self.projection = nn.Linear(em_dim, em_dim)
        self.register_buffer('tril', torch.tril(torch.ones(context_len, context_len)))  # lookahead mask
        self.dropout_attn = nn.Dropout(p_drop)
        self.dropout_proj = nn.Dropout(p_drop)

    def forward(self, x):
        B, T, _ = x.shape  # input shape: B, seq, em_dim
        k, q, v = self.key(x), self.query(x), self.value(x)  # B, seq, em_dim
        # split the head and move the head dimension next to batch so heads are indepenent
        k = k.reshape(B, T, self.num_heads, -1).permute(0, 2, 1, 3)  # B, head, seq, em_dim//head
        q = q.reshape(B, T, self.num_heads, -1).permute(0, 2, 1, 3)  # B, head, seq, em_dim//head
        v = v.reshape(B, T, self.num_heads, -1).permute(0, 2, 1, 3)  # B, head, seq, em_dim//head
        # attention
        attention = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # B, head, seq, seq
        if self.use_mask:
            attention = attention.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # apply lookahead mask
        attention = attention.softmax(dim=-1)
        attention = self.dropout_attn(attention)
        x = (attention @ v).permute(0, 2, 1, 3)  # B, seq, head, em_dim//head
        x = x.reshape(B, T, -1)  # B, seq, em_dim
        # projection
        x = self.projection(x)
        x = self.dropout_proj(x)
        return x


class AttentionBlock(nn.Module):
    """multi-attention  + feedforward skip connection"""
    def __init__(self, context_len, em_dim, num_heads, ffwd_dim, p_drop=0.2, use_mask=True):
        super().__init__()
        self.self_attention = MultiHeadAttention(context_len,
                                                 em_dim,
                                                 num_heads=num_heads,
                                                 p_drop=p_drop,
                                                 use_mask=use_mask)
        self.ffwd = nn.Sequential(nn.Linear(em_dim, ffwd_dim),
                                  nn.ReLU(),
                                  nn.Linear(ffwd_dim, em_dim),
                                  nn.Dropout(p_drop))
        self.norm1 = nn.LayerNorm(em_dim)
        self.norm2 = nn.LayerNorm(em_dim)

    def forward(self, x):
        x = x + self.self_attention(self.norm1(x))
        x = x + self.ffwd(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, num_blocks, vocab_size, context_len, em_dim, num_heads, ffwd_dim, p_drop=0.2, use_mask=True):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, em_dim)
        self.position_embedding = nn.Embedding(context_len, em_dim)
        self.blocks = nn.Sequential(
            *[AttentionBlock(context_len, em_dim, num_heads, ffwd_dim, p_drop, use_mask) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(em_dim)
        self.lm_head = nn.Linear(em_dim, vocab_size)
        self.register_buffer('pos_idx', torch.arange(context_len))  # position index
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        token_em = self.token_embedding(x)
        position_em = self.position_embedding(self.pos_idx[:x.shape[-1]])
        x = token_em + position_em
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


class CrossEntropy(TensorOp):
    def forward(self, data, state):
        logits, targets = data
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.reshape(B * T)
        loss = F.cross_entropy(logits, targets)
        return loss


def generate_response(question, model, tokenizer, max_response_token=128, context_len=512):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    tokens = torch.Tensor(tokenizer.encode(question)).long().to(device)
    num_input_tokens = tokens.shape[0]
    assert num_input_tokens <= context_len, "question exceeding maximum input tokens"
    tokens = tokens[None, ...]  # add batch dimension
    responses = None
    for _ in range(max_response_token):
        input_tokens = tokens[:, -context_len:]
        # get prediction
        logits = model(input_tokens)
        # focus only on the last time step
        logits = logits[:, -1, :]  # becomes (B, C)
        probs = F.softmax(logits, dim=-1)  # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        if responses is None:
            responses = idx_next
        else:
            responses = torch.cat((responses, idx_next), dim=1)  # (B, T+1)
        tokens = torch.cat((tokens, idx_next), dim=1)  # (B, T+1)
        if idx_next[0, 0] == 102:
            break
    responses = responses.to('cpu').numpy()
    responses = tokenizer.decode(responses[0])
    return responses


def get_estimator(data_dir=None,
                  epochs=50,
                  batch_size=32,
                  context_len=512,
                  num_blocks=6,
                  em_dim=1024,
                  ffwd_dim=4096,
                  num_heads=16,
                  save_dir=tempfile.mkdtemp(),
                  train_steps_per_epoch=3000,
                  eval_steps_per_epoch=500):
    # first load the data
    train_data, eval_data, test_data = wikitext_103.load_data(data_dir)
    train_data, eval_data, test_data = TextDataset(train_data), TextDataset(eval_data), TextDataset(test_data)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=[
            Encode(inputs="x", outputs="x", tokenizer=tokenizer),
            LambdaOp(fn=lambda x: x[:context_len + 1], inputs="x", outputs="x")  # get 1 more token for target
        ])
    model = fe.build(
        model_fn=lambda: GPT(num_blocks=num_blocks,
                             vocab_size=tokenizer.vocab_size,
                             context_len=context_len,
                             em_dim=em_dim,
                             num_heads=num_heads,
                             ffwd_dim=ffwd_dim,
                             p_drop=0.3),
        optimizer_fn=lambda x: torch.optim.AdamW(x, lr=3e-4))
    network = fe.Network(ops=[
        TLambdaOp(fn=lambda x: (x[..., :-1], x[..., 1:]), inputs="x", outputs=("input", "target")),
        ModelOp(model=model, inputs="input", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "target"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=BestModelSaver(model=model, save_dir=save_dir),
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()

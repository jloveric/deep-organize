# partially modified from https://github.com/karpathy/nanoGPT/blob/master/model.py

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
import logging
from lion_pytorch import Lion
from torch import Tensor
import torch.optim as optim
from torchmetrics import Accuracy


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Attention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, bias: bool, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )

        # TODO: I wonder if this is why I run into memory issues. Maybe I need contiguous
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, bias: bool, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, bias: bool, dropout: float):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = Attention(n_embd, n_head, bias, dropout)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        layers: int,
        n_head: int = 4,
        bias: bool = True,
        dropout: int = 0.0,
    ):
        layer_list = []
        for i in range(layers):
            layer_list.append(
                Block(input_size=input_size, n_head=n_head, bias=bias, dropout=dropout)
            )

        output = nn.Linear(input_size, output_size, bias=bias)
        layer_list.append(output)
        self.model = torch.nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)


class RegressionMixin:
    def eval_step(self, batch: Tensor, name: str):
        x, y, idx = batch
        y_hat = self(x)
        loss = self.loss(y_hat.flatten(), y.flatten())

        self.log(f"{name}_loss", loss, prog_bar=True)
        return loss


class PredictionNetMixin:
    def forward(self, x):
        ans = self.model(x)
        return ans

    def training_step(self, batch, batch_idx):
        return self.eval_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, "test")

    def configure_optimizers(self):
        
        if self.cfg.optimizer.name == "lion":
            optimizer = Lion(
                self.parameters(), lr=self.cfg.optimizer.lr, weight_decay=0.0
            )
        elif self.cfg.optimizer.name == "adam":
            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.cfg.optimizer.lr,
            )
        else:
            raise ValueError(f"Optimizer {self.cfg.optimizer.name} not recognized")

        if self.cfg.optimizer:
            reduce_on_plateau = False
            if self.cfg.optimizer.scheduler == "plateau":
                logger.info("Reducing lr on plateau")
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=self.cfg.optimizer.patience,
                    factor=self.cfg.optimizer.factor,
                    verbose=True,
                )
                reduce_on_plateau = True
            elif self.cfg.optimizer.scheduler == "exponential":
                logger.info("Reducing lr exponentially")
                lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self.cfg.optimizer.gamma
                )
            else:
                return optimizer

            scheduler = {
                "scheduler": lr_scheduler,
                "reduce_on_plateau": reduce_on_plateau,
                "monitor": "train_loss",
            }
            return [optimizer], [scheduler]



class Net(RegressionMixin, PredictionNetMixin, pl.LightningModule):
    def __init__(self, cfg: DictConfig) :
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = 

        self.loss = torch.nn.functional.mse_loss

    
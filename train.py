import math
import os
import pickle
import random
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pax
import wandb
from optax.losses import softmax_cross_entropy_with_integer_labels

devices = jax.devices()
print(devices)
num_devices = jax.device_count()
vocab_size = 50304
dim = 768
num_head = 12
num_layer = 12
warmup_steps = 2_000
training_steps = 500_000
b1 = 0.9
b2 = 0.95
learning_rate = 6e-4
seq_len = 1024
weight_decay = 1e-1
grad_clip = 1.0
wandb_project = "gpt-2-small"
num_tokens_per_batch = 1024 * 8 * 8 * 8
gradient_accumulation_steps = 8 * 4
updates_per_step = gradient_accumulation_steps
batch_size = (
    num_tokens_per_batch // num_devices // gradient_accumulation_steps // seq_len
)
print("Batch size per device:", batch_size)
wandb_run_name = f"{wandb_project}-seq-{seq_len}-{num_layer}-layer-{num_head}-head-{learning_rate:.1e}-{weight_decay:.1e}-batch-{batch_size}"
np.random.seed(42)

config = {
    "prefix": prefix,
    "vocab_size": vocab_size,
    "dim": dim,
    "num_head": num_head,
    "num_layer": num_layer,
    "b1": b1,
    "b2": b2,
    "learning_rate": learning_rate,
    "seq_len": seq_len,
    "batch_size": batch_size,
    "weight_decay": weight_decay,
    "num_tokens_per_batch": num_tokens_per_batch,
    "grad_clip": grad_clip,
    "warmup_steps": warmup_steps,
    "training_steps": training_steps,
}
wandb.init(project=wandb_project, name=wandb_run_name, config=config)


class RMSNorm(pax.ParameterModule):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.W = jnp.ones(dim).astype(jnp.float32)

    def __call__(self, x):
        x = x * jax.lax.rsqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        x = x * self.W[None, None, :]
        return x


class SelfAttention(pax.ParameterModule):
    def __init__(self, dim: int, num_head: int):
        super().__init__()
        self.W_qkv = jnp.array(
            np.random.randn(dim, 3 * dim) / math.sqrt(3 * dim)
        ).astype(jnp.float32)
        self.W_o = jnp.array(np.random.randn(dim, dim) * 1e-2 / math.sqrt(dim)).astype(
            jnp.float32
        )
        assert dim % num_head == 0
        self.H = num_head
        self.D = dim

    def __call__(self, x: jnp.ndarray):
        qkv = jnp.einsum("NLD,DK->NLK", x, self.W_qkv)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        N, L, D = q.shape
        H = self.H
        q = jnp.reshape(q, (N, L, H, D // self.H))
        k = jnp.reshape(k, (N, L, H, D // self.H))
        v = jnp.reshape(v, (N, L, H, D // self.H))
        scores = jnp.einsum("NSHD,NTHD->NSTH", q, k) / math.sqrt(q.shape[-1])
        mask = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))
        scores = jnp.where(mask[None, :, :, None], scores, float("-inf"))
        scores = jax.nn.softmax(scores, axis=2)
        v = jnp.einsum("NTHD,NSTH->NSHD", v, scores).reshape(N, L, D)
        o = jnp.einsum("NLD,DK->NLK", v, self.W_o)
        return o


class MLP(pax.ParameterModule):
    def __init__(self, dim: int):
        super().__init__()
        self.W_0 = jnp.array(np.random.randn(dim, 4 * dim) / math.sqrt(4 * dim)).astype(
            jnp.float32
        )
        self.W_1 = jnp.array(np.random.randn(dim, 4 * dim) / math.sqrt(4 * dim)).astype(
            jnp.float32
        )
        self.W_2 = jnp.array(
            np.random.randn(4 * dim, dim) * 1e-2 / math.sqrt(dim / 2)
        ).astype(jnp.float32)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_0 = jnp.einsum("NLD,DK->NLK", x, self.W_0)
        x_1 = jnp.einsum("NLD,DK->NLK", x, self.W_1)
        x = jax.nn.silu(x_0) * x_1
        x = jnp.einsum("NLK,KD->NLD", x, self.W_2)
        return x


class GPT(pax.ParameterModule):
    def __init__(
        self, dim: int, num_head: int, num_block: int, vocab_size: int, max_seq_len
    ):
        super().__init__()
        self.W_embed = jnp.array(np.random.randn(vocab_size, dim)).astype(
            jnp.float32
        ) / math.sqrt(dim)
        self.W_pos_embed = jnp.array(np.random.randn(max_seq_len, dim)).astype(
            jnp.float32
        ) / math.sqrt(dim)
        self.W_unembed = jnp.array(np.random.randn(dim, vocab_size)).astype(
            jnp.float32
        ) / math.sqrt(vocab_size)
        self.attn_norm_layers = [RMSNorm(dim) for _ in range(num_block)]
        self.attn_layers = [SelfAttention(dim, num_head) for _ in range(num_block)]
        self.mlp_norm_layers = [RMSNorm(dim) for _ in range(num_block)]
        self.mlp_layers = [MLP(dim) for _ in range(num_block)]
        self.max_seq_len = max_seq_len
        self.norm = RMSNorm(dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        N, L = x.shape
        pos = jnp.arange(0, L, dtype=x.dtype)[None, :]
        pos = self.W_pos_embed[(pos,)]
        x = self.W_embed[(x,)]
        x = x + pos
        for attn_norm, attn_fn, mlp_norm, mlp_fn in zip(
            self.attn_norm_layers,
            self.attn_layers,
            self.mlp_norm_layers,
            self.mlp_layers,
        ):
            x = x + attn_fn(attn_norm(x))
            x = x + mlp_fn(mlp_norm(x))
        x = self.norm(x)
        x = jnp.einsum("NLD,DK->NLK", x, self.W_unembed)
        return x


def get_data_iter(split: str, batch_size: int, block_size: int, data_dir="data"):
    data = np.copy(
        np.memmap(os.path.join(data_dir, split + ".bin"), dtype=np.uint16, mode="r")
    )
    L = data.shape[0]

    while True:
        idx = []
        for _ in range(batch_size):
            i = random.randint(0, L - block_size - 1 - 1)
            idx.append(range(i, i + block_size + 1))
        yield data[idx]


def loss_fn(model, x, y):
    logits = model(x)
    loss = softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss


def load_ckpt(step: int, model, path: Path):
    ckpt = pickle.load(open(f"{wandb_run_name}_{step:05d}.ckpt", "rb"))
    model = model.load_state_dict(ckpt["state_dict"])
    optimizer_state = ckpt["optimizer_state"]
    return step, model, optimizer_state


def save_ckpt(step: int, pmodel, poptimizer, path: Path):
    model = jax.tree_util.tree_map(lambda x: x[0], pmodel)
    optimizer = jax.tree_util.tree_map(lambda x: x[0], poptimizer)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "step": step,
                "state_dict": jax.device_get(model.state_dict()),
                "optimizer_state": jax.device_get(optimizer),
            },
            f,
        )


def update_step(net_optimizer_state, xy):
    net, optimizer_state = net_optimizer_state
    loss_and_grad_fn = pax.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn(net, xy[:, :-1], xy[:, 1:])
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, new_optimizer_state = optimizer.update(
        grads, optimizer_state, params=net.parameters()
    )
    new_parameters = optax.apply_updates(net.parameters(), updates)
    new_net = net.update_parameters(new_parameters)
    return (new_net, new_optimizer_state), loss


@partial(jax.pmap, axis_name="i")
def update_fn(model, optimizer_state, multi_batch: jnp.ndarray):
    (model, optimizer_state), losses = pax.scan(
        update_step, (model, optimizer_state), multi_batch
    )
    return model, optimizer_state, jnp.mean(losses)


net = GPT(dim, num_head, num_layer, vocab_size, seq_len)

lr_scheduler = optax.warmup_cosine_decay_schedule(
    0.0, 1.0, warmup_steps, training_steps, 1e-1
)
optimizer = optax.chain(
    optax.clip_by_global_norm(grad_clip),
    optax.scale_by_adam(b1=b1, b2=b2),
    optax.add_decayed_weights(weight_decay),
    optax.scale_by_schedule(lr_scheduler),
    optax.scale(-learning_rate),
)
optimizer = optax.MultiSteps(optimizer, every_k_schedule=gradient_accumulation_steps)
optimizer_state = optimizer.init(net.parameters())

# replicate on multiple devices
net = jax.device_put_replicated(net, jax.devices())
optimizer_state = jax.device_put_replicated(optimizer_state, jax.devices())

num_devices = jax.device_count()
data_iter = get_data_iter("train", num_devices * updates_per_step * batch_size, seq_len)

start = time.perf_counter()

for xy in data_iter:
    step = step + 1
    xy = jnp.reshape(xy, (num_devices, updates_per_step, batch_size, xy.shape[-1]))
    net, optimizer_state, loss = update_fn(net, optimizer_state, xy)
    loss = jnp.mean(loss)
    end = time.perf_counter()
    duration = end - start
    start = end
    lr = lr_scheduler(optimizer_state.gradient_step[0]).item()
    wandb.log(
        {
            "iter": step,
            "train/loss": loss.item(),
            "train/lr": lr,
            "train/duration": duration,
        },
        step=step,
    )

    print(f"{step:06d} {duration:.1f} {loss.item():.9f}")
    if step % 1000 == 0:
        save_ckpt(step, net, optimizer_state, f"{wandb_run_name}_{step:05d}.ckpt")

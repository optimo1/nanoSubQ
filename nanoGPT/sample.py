"""Talk to a trained nanoSubQ checkpoint. CPU-friendly (no GPU needed).

Usage:
    python sample.py out/ckpt_step_16000.pt --prompt "What is the capital of France?"
    python sample.py out/ckpt_step_16000.pt --interactive     # chat loop
"""
import os, sys, argparse
import torch
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import nanoSubQ

USER, ASST, END = "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>"


def build_tokenizer():
    # must match data/prepare.py exactly: gpt2 base + ChatML special tokens
    base_enc = tiktoken.get_encoding("gpt2")
    return tiktoken.Encoding(
        name="gpt2_chatml",
        pat_str=base_enc._pat_str,
        mergeable_ranks=base_enc._mergeable_ranks,
        special_tokens={**base_enc._special_tokens, "<|im_start|>": 50257, "<|im_end|>": 50258},
    )


@torch.no_grad()
def generate(model, enc, ctx, max_new_tokens, temperature, top_k):
    model.eval()
    block = model.config.block
    idx = torch.tensor(enc.encode(ctx, allowed_special="all")).unsqueeze(0)
    for _ in range(max_new_tokens):
        idx = idx[:, -model.config.max_seq_len:]   # keep within the RoPE table / context
        # subq_route asserts n % block == 0 (routing is block-aligned; training always n=1024).
        # Pad the front with token-0 so the length is a multiple of block; those padding tokens
        # never appear in the reply (they decode before the assistant marker and get stripped).
        pad = (-idx.shape[1]) % block
        padded = idx if pad == 0 else torch.cat([torch.zeros(1, pad, dtype=torch.long), idx], dim=1)
        logits, *_ = model(padded)
        logits = logits[0, -1] / max(temperature, 1e-3)   # last row = last real token
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.numel()))
            logits[logits < v[-1]] = -float("inf")
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).unsqueeze(0)
        idx = torch.cat([idx, next_id], dim=1)
        if next_id.item() == 50258:                # <|im_end|> -> assistant turn finished
            break
    return enc.decode(idx[0].tolist())


def reply(model, enc, ctx, args):
    out = generate(model, enc, ctx, args.max_new_tokens, args.temperature, args.top_k)
    tail = out.rsplit(ASST, 1)[-1]                 # keep only the assistant part
    return tail.split("<|im_end|>")[0].strip()


def main():
    ap = argparse.ArgumentParser(description="Talk to a trained nanoSubQ.")
    ap.add_argument("ckpt", help="path to the .pt checkpoint (e.g. out/ckpt_step_16000.pt)")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--interactive", action="store_true", help="chat loop")
    args = ap.parse_args()

    print(f"loading {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    cfg.attn_impl = "masked"       # exact reference path; no CUDA / flex compile needed on CPU
    model = nanoSubQ(cfg).eval()
    model.load_state_dict(ckpt["model"])
    enc = build_tokenizer()

    if args.interactive:
        print("Chat with nanoSubQ (Ctrl-C to quit).\n")
        history = ""
        while True:
            user = input("you: ").strip()
            if not user:
                continue
            history += USER + user + END + "\n" + ASST + "\n"
            ans = reply(model, enc, history, args)
            print(f"model: {ans}\n")
            history += ans + END + "\n"
    else:
        ctx = USER + args.prompt + END + "\n" + ASST + "\n"
        print("model: " + reply(model, enc, ctx, args))


if __name__ == "__main__":
    main()

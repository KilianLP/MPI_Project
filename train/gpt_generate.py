import argparse
import os

import torch

from data.text import SimpleCharTokenizer, load_text
from models import GPT, GPTConfig


def top_k_filter(logits: torch.Tensor, k: int):
    if k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


@torch.no_grad()
def sample(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_k: int, device):
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        logits = top_k_filter(logits, top_k)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return tokenizer.decode(idx[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint.")
    parser.add_argument("--file", type=str, default=os.path.join("data", "tinyshakespeare.txt"), help="Training text file to rebuild tokenizer.")
    parser.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints", "gpt_checkpoint.pt"))
    parser.add_argument("--prompt", type=str, default="To be, or not to be", help="Prompt to start generation.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling; 0 disables.")
    args = parser.parse_args()

    device = torch.device("cpu")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isfile(args.file):
        raise FileNotFoundError(f"Text file not found: {args.file}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = GPTConfig(**ckpt["config"])

    text = load_text(args.file)
    tokenizer = SimpleCharTokenizer(text)

    missing_chars = [ch for ch in args.prompt if ch not in tokenizer.stoi]
    if missing_chars:
        raise ValueError(f"Prompt contains unseen characters: {set(missing_chars)}")

    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])

    out = sample(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(out)


if __name__ == "__main__":
    main()

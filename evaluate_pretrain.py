#!/usr/bin/env python
"""
事前学習モデルの評価スクリプト
- Perplexity
- Top-k Accuracy
- 次トークン予測の分布分析
"""

import torch
from transformers import AutoTokenizer
from model import Transformer, ModelConfig
import math
import argparse

def load_model(checkpoint_path, device='cuda'):
    """モデル読み込み"""
    config = ModelConfig(
        vocab_size=32000, num_dims=512, num_heads=16, num_kv_heads=4,
        num_layers=32, ffn_hidden_dims=2048, rmsnorm_eps=1e-6, rope_theta=1e5,
        context_len=1024, use_cache=False, use_flash=True, use_moe=False,
        moe_num_experts=2, moe_active_experts=2, moe_eps=1e-6,
        moe_aux_loss_coef=0.01, moe_shared_experts=1, use_lossfreebalance=False
    )

    model = Transformer(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v
                  for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    return model, config

@torch.no_grad()
def evaluate_accuracy(model, tokenizer, test_texts, device='cuda'):
    """Top-k accuracyを評価"""
    model.eval()

    total_tokens = 0
    top1_correct = 0
    top5_correct = 0
    total_loss = 0.0

    for text in test_texts:
        # トークナイズ
        tokens = tokenizer.encode(text, return_tensors='pt').to(device)

        if tokens.shape[1] < 2:
            continue

        # 各位置で次トークン予測
        for i in range(tokens.shape[1] - 1):
            input_ids = tokens[:, :i+1]
            target = tokens[:, i+1]

            logits, _, _ = model(input_ids)
            next_logits = logits[0, -1, :]

            # Loss計算
            loss = torch.nn.functional.cross_entropy(next_logits.unsqueeze(0), target)
            total_loss += loss.item()

            # Top-1
            pred = next_logits.argmax()
            if pred == target.item():
                top1_correct += 1

            # Top-5
            top5_preds = next_logits.topk(5).indices
            if target.item() in top5_preds:
                top5_correct += 1

            total_tokens += 1

    return {
        'top1_accuracy': top1_correct / total_tokens if total_tokens > 0 else 0,
        'top5_accuracy': top5_correct / total_tokens if total_tokens > 0 else 0,
        'avg_loss': total_loss / total_tokens if total_tokens > 0 else 0,
        'perplexity': math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf'),
        'total_tokens': total_tokens
    }

@torch.no_grad()
def analyze_prediction_distribution(model, tokenizer, prompt, device='cuda'):
    """予測分布を分析"""
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    logits, _, _ = model(input_ids)
    next_logits = logits[0, -1, :]

    probs = torch.softmax(next_logits, dim=-1)
    top_k = 10
    top_probs, top_indices = probs.topk(top_k)

    print(f"\nPrompt: '{prompt}'")
    print(f"Top {top_k} predictions:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token = tokenizer.decode([idx.item()])
        print(f"  {i+1}. Token {idx.item():5d} '{token:20s}' prob: {prob.item():.4f}")

    # 確率分布の統計
    entropy = -(probs * torch.log(probs + 1e-10)).sum()
    print(f"\nEntropy: {entropy.item():.4f}")
    print(f"Max probability: {probs.max().item():.4f}")
    print(f"Effective vocab size (prob > 0.001): {(probs > 0.001).sum().item()}")

def main():
    parser = argparse.ArgumentParser(description='事前学習モデルの評価スクリプト')
    parser.add_argument('--checkpoint_path', type=str,
                        default='checkpoints_japanese/model.checkpoint.epoch0_step419393_global419393.pt',
                        help='評価するチェックポイントファイルのパス')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # モデル読み込み
    checkpoint_path = args.checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model, config = load_model(checkpoint_path, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters\n")

    # テストテキスト
    test_texts = [
        "人工知能は機械学習と深層学習の技術を使用しています。",
        "日本の首都は東京です。",
        "プログラミングを学ぶには練習が重要です。",
        "猫は可愛い動物です。",
        "太陽は東から昇ります。",
    ]

    print("="*60)
    print("1. Top-k Accuracy評価")
    print("="*60)

    results = evaluate_accuracy(model, tokenizer, test_texts, device)
    print(f"Total tokens evaluated: {results['total_tokens']}")
    print(f"Top-1 Accuracy: {results['top1_accuracy']*100:.2f}%")
    print(f"Top-5 Accuracy: {results['top5_accuracy']*100:.2f}%")
    print(f"Average Loss: {results['avg_loss']:.4f}")
    print(f"Perplexity: {results['perplexity']:.2f}")

    print("\n" + "="*60)
    print("2. 予測分布分析")
    print("="*60)

    test_prompts = [
        "人工知能",
        "日本の",
        "太陽は",
        "プログラミング",
    ]

    for prompt in test_prompts:
        analyze_prediction_distribution(model, tokenizer, prompt, device)

if __name__ == '__main__':
    main()

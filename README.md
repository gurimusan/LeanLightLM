# LightLM

様々なGPUアーキテクチャに対応した軽量言語モデルの実装

## 概要

このリポジトリは、**言語モデルの事前学習（Pre-training）を学ぶための教育用プロジェクト**です。

[LightLM: Building Our LLM From Scratch](https://medium.com/@bogdan.su/in-this-article-we-will-build-our-llm-which-i-called-lightlm-from-scratch-choose-the-optimal-c1e1839668db) の記事をベースに、以下を実装しています：

- **Transformerアーキテクチャ**: Grouped Query Attention (GQA)、RoPE、RMSNorm
- **Mixture of Experts (MoE)**: 専門家混合による効率的なモデル拡張
- **最適化**: Flash Attention、混合精度学習（bf16/fp16）、勾配累積
- **VRAM適応**: GPU容量に応じた自動設定調整（RTX 4060 8GB〜RTX 5090 32GB対応）
- **分散学習**: DDP (Distributed Data Parallel) サポート

## インストール

このプロジェクトはRTX 4060とRTX 5090の両方に対応し、適切なPyTorchバージョンを自動選択します。

### 必要な環境

- Python 3.12以上
- NVIDIA GPU (RTX 4060、RTX 5090、または互換性のあるGPU)
- `uv` パッケージマネージャー ([インストールガイド](https://docs.astral.sh/uv/))

### クイックインストール

インストールスクリプトを実行：

```bash
./install.sh
```

以下の処理が自動実行されます：
1. 基本依存パッケージのインストール
2. GPUの自動検出
3. 適切なPyTorchバージョンのインストール (RTX 4060: CUDA 12.4、RTX 5090: CUDA 12.8)
4. プロジェクトのセットアップ完了

### 手動インストール

手動でインストールする場合：

```bash
# 基本依存パッケージのインストール
uv sync --no-install-project

# GPUに応じたPyTorchのインストール
uv run python setup_pytorch.py

# インストールの完了
uv sync
```

### GPU別のPyTorchインストール

**RTX 5090の場合 (CUDA 12.8+が必要):**
```bash
uv pip install torch>=2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**RTX 4060の場合 (CUDA 12.4):**
```bash
uv pip install torch>=2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### インストールの確認

インストール後、PyTorchが正しくインストールされているか確認：

```bash
uv run python -c 'import torch; print(f"PyTorch: {torch.__version__}"); print(f"CUDA: {torch.version.cuda}"); print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}")'
```

PyTorchバージョン、CUDAバージョン、GPU名が表示されます。

## 使い方

### 学習の実行

VRAM容量に応じて自動的に最適な設定で学習を開始：

```bash
# 自動設定（推奨）
uv run python train.py

# カスタム設定
uv run python train.py --batch_size 4 --accumulation_steps 8

# シーケンス長を変更
uv run python train.py --max_seq_len 2048 --batch_size 2

# チェックポイントから学習を再開（デフォルトで再開される）
uv run python train.py --checkpoint_path ./checkpoints_japanese/model.checkpoint.epoch0_step500_global500.pt

# チェックポイントから再開しない場合
uv run python train.py --checkpoint_path ./checkpoints_japanese/model.checkpoint.epoch0_step500_global500.pt --no_continue_train
```

### VRAM容量別の自動設定

| VRAM容量 | GPU例 | batch_size | accumulation | 実効batch | dtype | compile |
|---------|-------|------------|-------------|----------|-------|---------|
| **24GB+** | RTX 4090, 5090 | 8 | 4 | 32 | bf16 | ✓ |
| **12-24GB** | RTX 4060 Ti 16GB | 4 | 8 | 32 | bf16 | ✓ |
| **8-12GB** | RTX 4060 8GB | 1 | 32 | 32 | bf16 | ✗ |
| **<8GB** | 古いGPU | 1 | 32 | 32 | fp16 | ✗ |

### データセット

デフォルトでは [hotchpotch/fineweb-2-edu-japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese) の `sample_10BT`（100億トークン）を使用します。

初回実行時は自動的にダウンロードされ、`~/.cache/huggingface/datasets/` にキャッシュされます。

### 事前学習モデルの評価

学習済みモデルを定量的に評価できます。

```bash
uv run python evaluate_pretrain.py
```

**評価指標:**
- **Perplexity**: モデルの予測精度（低いほど良い）
- **Top-1/Top-5 Accuracy**: 次トークン予測の正解率
- **予測分布分析**: 各プロンプトに対する予測確率の分布

**出力例:**
```
Top-1 Accuracy: 32.50%
Top-5 Accuracy: 60.00%
Perplexity: 32.53

Prompt: '人工知能'
Top 10 predictions:
  1. Token 3 'EOS' prob: 0.4192
  2. Token 196 '...' prob: 0.1723
  ...
```

評価結果は `log/eval_japanese_10BT.txt` に記録され、`eval_loss_plot.png` でlossの推移を可視化できます。

**主要な評価指標の解釈:**
- **Perplexity**: 次トークン予測の不確実性。低いほど良い（優秀: <10, 良好: 10-20）
- **Top-1 Accuracy**: 最も確率の高いトークンが正解である割合
- **Top-5 Accuracy**: 上位5つの候補に正解が含まれる割合

### lm-evaluation-harnessを使った標準評価

HuggingFace形式に変換後、標準的なベンチマークで評価できます。

```bash
# モデルをHuggingFace形式に変換
uv run python convert_to_hf.py \
  --checkpoint ./checkpoints_japanese/model.checkpoint.epoch0_step419393_global419393.pt \
  --output ./hf_model

# lm-evaluation-harnessで評価（カスタムモデルの場合は追加設定が必要）
# 注: 現在のカスタムアーキテクチャは標準的なAutoModelに未登録のため、
#     evaluate_pretrain.pyでの評価を推奨
```

**利用可能な日本語ベンチマーク:**
- `ja_leaderboard_jcommonsenseqa`: 常識推論
- `ja_leaderboard_jnli`: 自然言語推論
- `ja_leaderboard_marc_ja`: 感情分析
- `ja_leaderboard_jsquad`: 読解

### チェックポイントのHuggingFace形式への変換

学習済みチェックポイントをHuggingFace形式に変換できます。

#### ローカル変換

```bash
python convert_to_hf.py \
  --checkpoint ./checkpoints_japanese/model.checkpoint.epoch0_step500_global500.pt \
  --output ./hf_model
```

#### HuggingFace Hubへアップロード

```bash
python convert_to_hf.py \
  --checkpoint ./checkpoints_japanese/model.checkpoint.epoch0_step500_global500.pt \
  --output ./hf_model \
  --push_to_hub username/lightlm-japanese-150m
```

#### 変換後のモデルの使用方法

**ローカルから読み込み:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルとトークナイザーの読み込み
model = AutoModelForCausalLM.from_pretrained("./hf_model")
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b")

# テキスト生成
inputs = tokenizer("こんにちは、", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.8, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**HuggingFace Hubから読み込み:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hubからモデルを直接読み込み
model = AutoModelForCausalLM.from_pretrained("username/lightlm-japanese-150m")
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b")

# テキスト生成
text = "人工知能とは"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    do_sample=True
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## トラブルシューティング

### ModuleNotFoundError: No module named 'torch'

PyTorchのインストールを実行してください：
```bash
uv run python setup_pytorch.py
```

### CUDAバージョンの不一致

CUDA関連のエラーが発生した場合、GPUを確認してPyTorchを再インストール：
```bash
nvidia-smi  # GPUモデルの確認
uv run python setup_pytorch.py  # 正しいバージョンで再インストール
```

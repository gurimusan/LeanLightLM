# python train_japanese.py --gpu rtx4060
# python train_japanese.py --gpu rtx5090
# 日本語特化LLMの学習スクリプト
# データセット: fineweb-2-edu-japanese (sample_10BT - 100億トークン)
# Tokenizer: rinna/japanese-gpt-neox-3.6b

from model import Transformer, ModelConfig
from trainer import Trainer, TrainerConfig, DataLoader

from transformers import AutoTokenizer
import torch
import argparse

torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()

# コマンドライン引数のパース
parser = argparse.ArgumentParser(description='日本語LLM学習スクリプト')
parser.add_argument('--gpu', type=str, default='rtx5090', choices=['rtx4060', 'rtx5090'],
                    help='GPU設定を選択 (rtx4060: 8GB VRAM, rtx5090: 32GB VRAM)')
args = parser.parse_args()

# GPU別の設定
if args.gpu == 'rtx4060':
    # RTX 4060 (8GB VRAM) 用設定
    GPU_CONFIG = {
        'max_seq_len': 1024,
        'batch_size': 1,
        'accumulation_steps': 32,
        'use_compile': False,
        'use_dtype': 'bfloat16',
    }
else:  # rtx5090
    # RTX 5090 (32GB VRAM) 用設定
    GPU_CONFIG = {
        'max_seq_len': 1024,
        'batch_size': 8,
        'accumulation_steps': 4,
        'use_compile': True,
        'use_dtype': 'bfloat16',
    }

# 日本語tokenizer (vocab_size=32000)
tokenizer_id = "rinna/japanese-gpt-neox-3.6b"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# チェックポイントから再開する場合の設定
checkpoint_path = ''  # 例: './checkpoints_japanese/model.checkpoint.epoch0_step100_global100.pt'
continue_train = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_config = TrainerConfig(
    vocab_size=tokenizer.vocab_size,  # 32000
    num_epochs=1,

    use_ddp=False,
    use_moe=False,
    use_lossfreebalance=False,
    clean_cuda_cache=True,
    use_compile=GPU_CONFIG['use_compile'],
    use_dtype=GPU_CONFIG['use_dtype'],

    # GradScaler設定 (float16使用時のみ有効)
    grad_scaler_init_scale=2.**12,
    grad_scaler_growth_factor=2.0,
    grad_scaler_backoff_factor=0.5,
    grad_scaler_growth_interval=2000,

    seed=1338,
    max_seq_len=GPU_CONFIG['max_seq_len'],
    batch_size=GPU_CONFIG['batch_size'],
    accumulation_steps=GPU_CONFIG['accumulation_steps'],  # 実効バッチサイズ = batch_size * accumulation_steps = 32

    weight_decay=0.1,
    warmup_ratio=0.1,
    learning_rate=4e-4,
    betas=(0.90, 0.97),
    update_rate=5e-6,

    val_ratio=0.005,
    steps_for_eval=20,
    eval_interval=100,

    checkpoints_frequency=500,
    path_to_checkpoints="./checkpoints_japanese",
    max_checkpoints_to_keep=3,  # 最新3つのチェックポイントを保持 (0=全て保持, -1=最新1つのみ)

    # データセット設定 (sample_10BT サブセット - 100億トークン)
    tokenized_dataset_path="hotchpotch/fineweb-2-edu-japanese",
    sub_target_files="",  # 設定名は別途指定
    eval_log_file="log/eval_japanese_10BT.txt",

    # チェックポイント再開設定
    continue_train=continue_train,
    checkpoint_path=checkpoint_path,
)

config = ModelConfig(
    vocab_size=tokenizer.vocab_size,  # 32000

    # モデルサイズ: 約150M parameters
    num_dims=512,
    num_heads=16,
    num_kv_heads=4,  # Grouped Query Attention
    num_layers=32,
    ffn_hidden_dims=512 * 4,

    rmsnorm_eps=1e-6,
    rope_theta=1e5,

    context_len=GPU_CONFIG['max_seq_len'],  # max_seq_lenと同じにする

    use_cache=False,  # 学習時はFalse、推論時はTrue
    use_flash=True,   # Flash Attention使用
    use_moe=False,

    # MoE設定 (use_moe=Trueの場合のみ有効)
    moe_num_experts=2,
    moe_active_experts=2,
    moe_eps=1e-6,
    moe_aux_loss_coef=0.01,
    moe_shared_experts=1,
    use_lossfreebalance=False,
)

print("=" * 80)
print("日本語LLM学習設定 (sample_10BT - 100億トークン)")
print("=" * 80)
print(f"GPU設定: {args.gpu.upper()}")
print(f"Tokenizer: {tokenizer_id}")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Dataset: {train_config.tokenized_dataset_path}")
print("Subset: sample_10BT (10 Billion Tokens)")
print("Model parameters: ~150M")
print(f"Batch size: {train_config.batch_size}")
print(f"Accumulation steps: {train_config.accumulation_steps}")
print(f"Effective batch size: {train_config.batch_size * train_config.accumulation_steps}")
print(f"Max sequence length: {train_config.max_seq_len}")
print(f"Use compile: {train_config.use_compile}")
print(f"Dtype: {train_config.use_dtype}")
print(f"Device: {device}")
print(f"Continue training: {continue_train}")
print("=" * 80)

# モデル初期化
model = Transformer(config)

# チェックポイントから再開する場合（train_config.continue_train=Trueの場合はTrainer内で自動的に読み込まれる）
if continue_train and not train_config.continue_train:
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    print("Checkpoint loaded successfully.")

model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# DataLoader初期化 (HuggingFace datasetsを使用)
# small_data_size: 動作確認用に小さいデータセットを使う場合に指定 (例: 1000)
# cache: トークナイズ済みデータをキャッシュするパス
# use_cache: キャッシュを使用するかどうか
print("Initializing DataLoader...")
print("Note: 初回実行時はデータセットのダウンロードとトークナイズに時間がかかります")
print("      キャッシュを有効にすると2回目以降は高速に読み込めます")

data_loader = DataLoader(
    train_config,
    tokenizer=tokenizer,
    rank=0,
    world_size=1,
    hf_split="train",
    hf_config="sample_10BT",  # データセット設定を指定
    streaming=False,  # sample_10BTは比較的小さいのでstreaming=False推奨
    small_data_size=None,  # 全データ使用。テスト時は1000など指定
    cache="./cache",  # トークナイズ済みデータをキャッシュ
    use_cache=True  # 2回目以降はキャッシュから高速読み込み
)

# Trainer初期化と学習開始
trainer = Trainer(train_config, model, tokenizer)

print("\n" + "=" * 80)
print("Starting training...")
print("=" * 80 + "\n")

trainer.train(data_loader)

print("\n" + "=" * 80)
print("Training completed!")
print(f"Checkpoints saved in: {train_config.path_to_checkpoints}")
print(f"Evaluation log: {train_config.eval_log_file}")
print("=" * 80)

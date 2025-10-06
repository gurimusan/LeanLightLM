#!/usr/bin/env python3
"""
チェックポイントをHuggingFace形式に変換するスクリプト

使い方:
    python convert_to_hf.py --checkpoint ./checkpoints/model.checkpoint.pt --output ./hf_model
    python convert_to_hf.py --checkpoint ./checkpoints/model.checkpoint.pt --output ./hf_model --push_to_hub username/model-name
"""

import argparse
import os
import torch
from model import Transformer, ModelConfig
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional


class LightLMConfig(PretrainedConfig):
    """HuggingFace用のモデル設定クラス"""
    model_type = "lightlm"

    def __init__(
        self,
        vocab_size: int = 32000,
        num_dims: int = 512,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        num_layers: int = 32,
        ffn_hidden_dims: int = 2048,
        context_len: int = 1024,
        rmsnorm_eps: float = 1e-6,
        rope_theta: float = 1e5,
        use_cache: bool = True,
        use_flash: bool = True,
        use_moe: bool = False,
        moe_num_experts: int = 2,
        moe_active_experts: int = 2,
        moe_eps: float = 1e-6,
        moe_aux_loss_coef: float = 0.01,
        moe_shared_experts: int = 0,
        use_lossfreebalance: bool = False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.ffn_hidden_dims = ffn_hidden_dims
        self.context_len = context_len
        self.rmsnorm_eps = rmsnorm_eps
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.use_flash = use_flash
        self.use_moe = use_moe
        self.moe_num_experts = moe_num_experts
        self.moe_active_experts = moe_active_experts
        self.moe_eps = moe_eps
        self.moe_aux_loss_coef = moe_aux_loss_coef
        self.moe_shared_experts = moe_shared_experts
        self.use_lossfreebalance = use_lossfreebalance
        super().__init__(**kwargs)


class LightLMForCausalLM(PreTrainedModel):
    """HuggingFace互換のラッパークラス"""
    config_class = LightLMConfig

    def __init__(self, config: LightLMConfig):
        super().__init__(config)

        # ModelConfigに変換
        model_config = ModelConfig(
            vocab_size=config.vocab_size,
            num_dims=config.num_dims,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            num_layers=config.num_layers,
            ffn_hidden_dims=config.ffn_hidden_dims,
            context_len=config.context_len,
            rmsnorm_eps=config.rmsnorm_eps,
            rope_theta=config.rope_theta,
            use_cache=config.use_cache,
            use_flash=config.use_flash,
            use_moe=config.use_moe,
            moe_num_experts=config.moe_num_experts,
            moe_active_experts=config.moe_active_experts,
            moe_eps=config.moe_eps,
            moe_aux_loss_coef=config.moe_aux_loss_coef,
            moe_shared_experts=config.moe_shared_experts,
            use_lossfreebalance=config.use_lossfreebalance,
        )

        self.model = Transformer(model_config)

    def forward(self, input_ids, labels=None, **kwargs):
        return self.model(input_ids, labels)

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=50, top_p=1.0, **kwargs):
        return self.model.generate(
            input_ids,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=True
        )


def load_checkpoint(checkpoint_path: str):
    """チェックポイントを読み込む"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def create_model_config_from_checkpoint(checkpoint: dict) -> ModelConfig:
    """チェックポイントからモデル設定を推定"""
    # state_dictからモデル構造を推定
    state_dict = checkpoint['model']

    # _orig_mod.プレフィックスを除去
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[len("_orig_mod."):]] = v
        else:
            clean_state_dict[k] = v

    # vocab_sizeの取得
    vocab_size = clean_state_dict['tokens_embedding.weight'].shape[0]

    # num_dimsの取得
    num_dims = clean_state_dict['tokens_embedding.weight'].shape[1]

    # num_layersの取得
    num_layers = len([k for k in clean_state_dict.keys() if k.startswith('blocks.') and k.endswith('.attention.wq.weight')])

    # num_headsの取得
    # wq.weightのshapeから推定 (num_dims, num_dims)
    num_heads = 16  # デフォルト値、必要に応じて調整

    # num_kv_headsの取得
    wk_shape = clean_state_dict['blocks.0.attention.wk.weight'].shape
    head_dim = num_dims // num_heads
    num_kv_heads = wk_shape[0] // head_dim

    # ffn_hidden_dimsの取得
    ffn_hidden_dims = clean_state_dict['blocks.0.ffn.w1.weight'].shape[0] if 'blocks.0.ffn.w1.weight' in clean_state_dict else num_dims * 4

    # MoEの検出
    use_moe = 'blocks.0.ffn.router.weight' in clean_state_dict

    print(f"Detected model config:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  num_dims: {num_dims}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  num_layers: {num_layers}")
    print(f"  ffn_hidden_dims: {ffn_hidden_dims}")
    print(f"  use_moe: {use_moe}")

    return ModelConfig(
        vocab_size=vocab_size,
        num_dims=num_dims,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        ffn_hidden_dims=ffn_hidden_dims,
        context_len=1024,  # デフォルト値
        use_cache=True,
        use_flash=True,
        use_moe=use_moe,
        moe_num_experts=2 if use_moe else 0,
        moe_active_experts=2 if use_moe else 0,
    )


def convert_checkpoint_to_hf(
    checkpoint_path: str,
    output_dir: str,
    model_config: Optional[ModelConfig] = None,
    push_to_hub: Optional[str] = None
):
    """チェックポイントをHuggingFace形式に変換"""

    # チェックポイント読み込み
    checkpoint = load_checkpoint(checkpoint_path)

    # モデル設定の作成
    if model_config is None:
        model_config = create_model_config_from_checkpoint(checkpoint)

    # HuggingFace Configに変換
    hf_config = LightLMConfig(
        vocab_size=model_config.vocab_size,
        num_dims=model_config.num_dims,
        num_heads=model_config.num_heads,
        num_kv_heads=model_config.num_kv_heads,
        num_layers=model_config.num_layers,
        ffn_hidden_dims=model_config.ffn_hidden_dims,
        context_len=model_config.context_len,
        rmsnorm_eps=model_config.rmsnorm_eps,
        rope_theta=model_config.rope_theta,
        use_cache=model_config.use_cache,
        use_flash=model_config.use_flash,
        use_moe=model_config.use_moe,
        moe_num_experts=model_config.moe_num_experts,
        moe_active_experts=model_config.moe_active_experts,
    )

    # モデルの作成
    print("Creating HuggingFace model...")
    hf_model = LightLMForCausalLM(hf_config)

    # 重みの読み込み
    state_dict = checkpoint['model']

    # _orig_mod.プレフィックスの除去（torch.compile使用時）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v
        else:
            new_state_dict[k] = v

    # モデルに重みをロード
    hf_model.model.load_state_dict(new_state_dict, strict=False)

    # 保存
    print(f"Saving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir, safe_serialization=False)
    hf_config.save_pretrained(output_dir)

    # README作成
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"""# LightLM Model

This model was converted from a LightLM checkpoint.

## Model Details

- Vocabulary Size: {model_config.vocab_size}
- Hidden Size: {model_config.num_dims}
- Number of Layers: {model_config.num_layers}
- Number of Attention Heads: {model_config.num_heads}
- Number of KV Heads: {model_config.num_kv_heads}
- FFN Hidden Dims: {model_config.ffn_hidden_dims}
- Context Length: {model_config.context_len}
- Uses MoE: {model_config.use_moe}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{output_dir}")
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b")

# Generate text
inputs = tokenizer("こんにちは", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Training Info

- Checkpoint: {checkpoint_path}
- Epoch: {checkpoint.get('epoch', 'N/A')}
- Step: {checkpoint.get('step', 'N/A')}
- Global Step: {checkpoint.get('global_step', 'N/A')}
""")

    print(f"✓ Model saved to {output_dir}")
    print(f"✓ README.md created")

    # HuggingFace Hubにプッシュ
    if push_to_hub:
        print(f"Pushing model to HuggingFace Hub: {push_to_hub}...")
        hf_model.push_to_hub(push_to_hub)
        hf_config.push_to_hub(push_to_hub)
        print(f"✓ Model pushed to https://huggingface.co/{push_to_hub}")


def main():
    parser = argparse.ArgumentParser(description='Convert LightLM checkpoint to HuggingFace format')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='チェックポイントファイルのパス')
    parser.add_argument('--output', type=str, required=True,
                        help='出力ディレクトリ')
    parser.add_argument('--push_to_hub', type=str, default=None,
                        help='HuggingFace Hubにプッシュする場合のリポジトリ名 (例: username/model-name)')

    args = parser.parse_args()

    # チェックポイントの存在確認
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return

    # 変換実行
    convert_checkpoint_to_hf(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        push_to_hub=args.push_to_hub
    )

    print("\n" + "=" * 80)
    print("Conversion completed successfully!")
    print("=" * 80)
    print(f"\nTo load the model:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output}')")


if __name__ == '__main__':
    main()

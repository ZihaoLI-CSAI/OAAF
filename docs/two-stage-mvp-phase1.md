# H-SAM 两阶段解码 — Phase 1 MVP

## 行为

- **Stage-1**：现有 HF-SAM `MaskDecoder_512`（AWFFM），输出 128×128 粗 logits，并暴露 `msk_feat` / `up_embed` 供后续阶段使用。
- **Stage-2**：`MaskDecoder2_512`（`mvp_mode=True`）
  - 先验 `P` = Stage-1 的 **128×128** logits（不再额外 `/4` 到 32×32）
  - 启用 **Mask Cross-Attention**（`TwoWayTransformer2` + `mask_feat`）
  - **关闭** CMAttn（`msk_feat` 自注意力增强）与 **Pixel skip**（`up_embed` 拼接），改用 `mvp_pixel_head` 上采样到 512

## 训练 / 测试

| 参数 | 默认 | 含义 |
|------|------|------|
| `--two_stage` | `1` | 开启两阶段 |
| `--lambda_stage1` | `0.25` | Stage-1 deep supervision 权重 |
| `--stage`（test.py） | `2` | `2`=Stage-2 输出；`1`=仅看 Stage-1 |

从 Patch ABC 的 `epoch_299.pth` 加载：LoRA + Stage-1 有权重，**Stage-2 随机初始化**（checkpoint 不含 `mask_decoder2` 键时正常）。

## 对照实验

1. 训练：与 Patch ABC 相同超参，或先 50–100 epoch 冒烟。
2. 测试：12 例 Synapse volume，`test.py --stage 2`。
3. 对照：`汇报/PatchABC-测试结果-2026-10-28.md`；可选 `--stage 1` 看先验质量。

## 关闭两阶段（回退 Patch ABC 行为）

`train.py` / 构建模型后设置 `--two_stage 0`，或 `sam.two_stage = False`。

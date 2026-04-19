# Sa2VA / SAMTok Repo Audit

## 审查结论

本项目第 1 版应当以 `Sa2VA` 仓库中的 `projects/samtok` 作为离散 mask token 主干，以 `projects/sa2va` 的 Qwen3-VL 适配器和评测逻辑作为辅助参考，并从 `projects/vrt_sa2va` 复用现成的 group-normalized RL 实现思路。

核心理由：

1. `projects/samtok` 已经把 `Qwen3-VL + SAMTok`、mask tokenizer、Qwen3-VL 训练配置和 Qwen3-VL 评测脚本打通。
2. `projects/sa2va` 已经提供了 Qwen3-VL 适配器、通用训练入口、RefCOCO 评测脚本和 RefCOCO 数据读取工具。
3. `projects/vrt_sa2va/models/sa2va_grpo.py` 已经实现了 group-normalized RL / GRPO 风格的最小可复用逻辑，适合迁移到本项目的 Stage 3。

## 关键真实入口

### 1. Qwen3-VL 模型加载

- `Sa2VA/projects/sa2va/models/mllm/qwen3vl.py`
  - `Qwen3VL` 封装 `Qwen3VLForConditionalGeneration.from_pretrained(...)`
  - 提供 `add_special_tokens(...)`
  - 提供 `generate(...)`
- `Sa2VA/projects/samtok/models/qwen3vl.py`
  - `QWEN3VL_VQSAM2Model`
  - 这是 SAMTok 训练配置里直接使用的 Qwen3-VL 训练封装

### 2. SAMTok mask tokenizer 编解码

- `Sa2VA/projects/samtok/models/sam2.py`
  - `VQ_SAM2Config`
  - `VQ_SAM2.forward(...)` 用 GT mask + bbox 编码出 `quant_codes`
  - `VQ_SAM2.forward_with_codes(...)` 用离散 code 解码回 mask
- `Sa2VA/projects/samtok/models/__init__.py`
  - 暴露 `VQ_SAM2`, `VQ_SAM2Config`, `SAM2Config`, `DirectResize`
- `Sa2VA/projects/samtok/utils/add_special_tokens.py`
  - 真实 special tokens 形态是 `<|mt_start|>`, `<|mt_XXXX|>`, `<|mt_end|>`
  - 这些 token 已经是 Qwen3-VL SAMTok checkpoint 的词表一部分

### 3. SAMTok Qwen3-VL 训练配置与数据输入

- `Sa2VA/projects/samtok/configs/qwen3vl_4b_mt256x2.py`
  - 现成 Qwen3-VL + SAMTok xtuner 配置
- `Sa2VA/projects/samtok/datasets/qwen3vl_dataset.py`
  - Qwen3-VL 训练样本格式
  - 使用 `processor.apply_chat_template(...)`
  - 证明项目当前使用 Qwen chat template 而不是手写 token 拼接
- `Sa2VA/projects/samtok/datasets/collect_fns.py`
  - `qwen25vl_vqsam2_collate_fn(...)`
  - 当前 repo 对 Qwen-VL 样本的 padding / image_grid_thw 处理方式

### 4. RefCOCO 数据读取与评测

- `Sa2VA/third_parts/mmdet/datasets/refcoco.py`
  - 通用 RefCOCO/RefCOCO+/RefCOCOg 读取器
- `Sa2VA/projects/sa2va/evaluation/utils/refcoco_refer.py`
  - 另一套 `REFER` API
- `Sa2VA/projects/sa2va/evaluation/sa2va_eval_refcoco.py`
  - 现成 RefCOCO 评测入口

### 5. SAMTok Qwen3-VL 评测与推理脚本

- `Sa2VA/projects/samtok/evaluation/qwen3vl/qwen3vl_mrrefcoco_eval.py`
- `Sa2VA/projects/samtok/evaluation/qwen3vl/qwen3vl_mrrefcoco+_eval.py`
- `Sa2VA/projects/samtok/evaluation/qwen3vl/qwen3vl_mrrefcocog_eval.py`
- `Sa2VA/projects/samtok/evaluation/qwen3vl/qwen3vl_dam_infer.py`

这些脚本确认了两件事：

1. `Qwen3-VL-4B-SAMTok-co` 是 refseg 强势起点。
2. `Qwen3-VL-4B-SAMTok-dam` 是 mask understanding / region captioning 起点。

### 6. RL / GRPO 可复用代码

- `Sa2VA/projects/vrt_sa2va/models/sa2va_grpo.py`
  - 已有 reward 归一化、advantage 归一化、group rollout、reference-KL 风格逻辑
  - 可直接借鉴 Stage 3 的 `GRPO-lite`

## 与规格书对齐后的实现决策

### 保留

- 继续使用 `Qwen3-VL-4B-SAMTok-co` 作为默认 student 起点
- 继续使用 `VQ_SAM2` 作为唯一 mask tokenizer / decoder
- 继续使用 Qwen chat template
- 继续使用 RefCOCO 官方结构和 Sa2VA 中已有读取逻辑

### 不直接沿用

- 不直接把 `projects/samtok/configs/qwen3vl_4b_mt256x2.py` 当成最终训练主入口
  - 原因：它只有标准 SFT，没有 overlay teacher / failed-only OPD / joint RL
- 不直接复用 `projects/sa2va/models/sa2va.py`
  - 原因：它面向 `[SEG]` token + SAM2 grounding decoder，而不是 SAMTok 两 token 编解码

## 结论

新增项目会放在：

- `Sa2VA/projects/pixvl_idea1/`

实现策略是：

1. 复用 `Sa2VA` 仓库作为唯一主代码库。
2. 复用其 `Qwen3-VL` 和 `SAMTok` 组件。
3. 单独实现 joint SFT / OPD / RL trainer，避免污染原有 `sa2va` / `samtok` 主线。

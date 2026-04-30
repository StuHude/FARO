# Idea 1 实施规格书（for Codex）

> 升级说明（2026-04-19）  
> - 保留原 Stage 0~3、MVP、baseline、ablation 与工程执行顺序，不删除原内容。  
> - 从后半部分新增的 V2/V3 小节开始，加入升级版主线，用于在第 1 版验证有效后继续补 novelty、hard-slice performance 与 paper differentiation。  
> - 升级版的重点不是简单堆更多 reward，而是把 overlay teacher 的 privileged 信息压缩成 student 可学习的短证据（evidence），并把 binary fail gate 升级成 confidence-aware triage。  


> 工作名：**Joint Overlay-Teacher OPD with Segmentation-First RL**
>
> 一句话：在 **Qwen3-VL + SAMTok** 上，只用**现成公开数据**做一个可快速验证的 joint pixel-level MLLM 版本：
> - 同时训练 **text -> mask** 和 **mask -> text**
> - student 看原图，teacher 只在训练时看 **same-resolution GT overlay/highlight 图**
> - 在 student 的 **on-policy** 输出上做 **failed-only distillation**
> - RL 先以 **referring segmentation** 为主，caption 侧只做弱辅助，不让工程复杂度失控

---

## 0. 本文档的目标

这份文档不是论文写作稿，而是给 Codex 的**工程执行规格书**。

目标只有一个：

**尽快做出第 1 版可运行、可验证、可复现实验的 joint pixel-level MLLM pipeline。**

这个第 1 版只验证下面这条核心命题：

> 在不造新数据、不改 SAMTok tokenizer、不引入 crop teacher / evidence bottleneck / cycle / confuser reward 的前提下，
> 仅用 **same-resolution overlay privileged teacher + failed-only on-policy distillation + segmentation-first RL**，
> 能否在 **joint mask captioning + referring segmentation** 设置里稳定提升 RefCOCO 系列，同时不显著损害、最好还能带动 DLC-Bench。

---

## 1. 这个版本的边界

## 1.1 必须做

1. **joint 两个任务**
   - text -> mask（Referring Segmentation）
   - mask -> text（Mask Captioning / Region Captioning）

2. **只用公开资源**
   - base repo: Sa2VA / SAMTok codebase
   - base model: Qwen3-VL + SAMTok released checkpoint
   - train data: RefCOCO / RefCOCO+ / RefCOCOg + DAM training data + GAR Seed/Fine-Grained data

3. **同分辨率 overlay teacher**
   - student 输入原图
   - teacher 输入 full-image 坐标系下的 GT-highlight 图
   - teacher 只在训练时存在

4. **failed-only on-policy distillation**
   - 先让 student 自己 sample
   - 样本失败时才蒸馏
   - 不做 all-sample distill 作为主方法

5. **RL 一定要进 pipeline**
   - 但先做 **segmentation-first RL**
   - caption RL 只做低权重辅助，避免因为 caption reward 太脆弱而拖慢第一版验证

---

## 1.2 明确不做

下面这些全部**排除在第 1 版之外**：

- evidence bottleneck
- confuser consistency / retrieval 扩展
- crop-only teacher / multi-view crop teacher
- semantic / relation / geometry error taxonomy routing
- cycle reward / task-level闭环
- 额外 pseudo data 生成
- 新 benchmark 构建
- tokenizer 改造
- segmentation decoder 改造

这些内容属于后续版本，不要提前混入第 1 版。

---

## 2. 为什么第 1 版这样设计

### 2.1 公开资源足够支撑这版实验

Sa2VA 官方 repo 已公开训练代码、评测代码、数据下载入口和 fine-tuning 示例；DAM repo 已公开训练数据与 DLC-Bench；GAR repo 已公开 Seed / Fine-Grained / Relation 数据集下载方式。Qwen3-VL 官方 repo 也已公开 4B / 8B 模型。这样第 1 版可以只依赖现成资源开跑。  
来源：
- Sa2VA repo README / training / data / fine-tuning guide 入口
- DAM repo README / training data / DLC-Bench
- GAR repo README / dataset download / training entry
- Qwen3-VL official repo release note

### 2.2 为什么 teacher 先只用 overlay，不用 crop

SAMTok 的 mask token 需要和“当前图像”一起解码成 2D mask。也就是说，如果 teacher 看 pure crop、student 看 full image，很容易让 token 的坐标语义不一致。第 1 版只用 **same-resolution full-image overlay**，保持 teacher/student 的图像坐标系一致，这是工程上最稳的做法。

### 2.3 为什么 RL 先偏向 segmentation

SAMTok 已经证明：把 mask 当离散 token 后，mask generation 可以用标准 next-token learning 和简单 RL 来优化，并且文本型 reward 能有效提升 mask generation 指标。caption 这边虽然也可做 RL，但第一版最稳妥的策略仍然是：

- segmentation 分支做主 RL
- caption 分支以 SFT + OPD 为主，RL 只做低权重辅助

这样可以满足“joint + RL”，但不让 caption reward 成为主风险源。

---

## 3. 第 1 版的方法名与论文主张

建议内部命名：

# **JOINT-OVERLAY-OPD-RL**

建议论文化主张：

> A same-resolution privileged overlay teacher can improve joint pixel-language alignment in SAMTok-style pixel-level MLLMs when combined with failed-only on-policy distillation and segmentation-first reinforcement learning.

这个主张必须通过下面三件事证明：

1. **joint 训练比只做单任务更稳或更强**
2. **overlay teacher 比原图 teacher 更有用**
3. **failed-only OPD 比 all-sample distill 更好**

---

## 4. 基础实现决策

## 4.1 基础代码库

以 **Sa2VA 官方 repo** 为主代码库，不另起炉灶。

理由：

- repo 已公开训练 / 评测 / 推理代码和数据入口
- 同仓包含 Sa2VA 与 SAMTok 相关代码资产
- 能减少 Codex 的基建成本

Codex 第一件事：

1. clone `bytedance/Sa2VA`
2. 检查 repo 中和 SAMTok / Qwen3-VL 相关的 project/config/model loader
3. 找到：
   - Qwen3-VL + SAMTok 的 model loading 入口
   - mask token encode/decode 入口
   - trainer 入口
   - evaluation 入口

> 注意：**不要先写代码**，先定位 repo 里的真实路径和命名。不要臆造 exact token string / config file path。

---

## 4.2 起始 checkpoint

首选起点：**Qwen3-VL-4B SAMTok checkpoint，优先选择 segmentation 更强的 released checkpoint**。

建议顺序：

1. **主起点**：`Qwen3-VL-4B-SAMTok-co` 或同等定位的 referring segmentation 强势 checkpoint
2. **可选对照**：`Qwen3-VL-4B-SAMTok-dam` 或 caption 更强的 checkpoint

第 1 版默认先从 segmentation 更强的 checkpoint 起步，因为：

- RL 主目标在 segmentation 端
- caption 数据会在 joint SFT 中补回来

若 repo/HF 实际命名与这里不同，以 repo 内实际可用模型为准。

---

## 4.3 teacher 形式

第 1 版 teacher 不引入额外大模型。

采用：

# **frozen reference teacher**

具体做法：

- 先完成 Stage 1 joint SFT，得到一个 warm checkpoint
- 固定这个 checkpoint 作为 teacher reference
- 后续 Stage 2 / Stage 3 中：
  - student：继续更新，输入原图
  - teacher：冻结，输入 overlay 图

这样比“同权重自蒸馏”更稳，也比引入更大 teacher 省资源。

可选增强（不是默认）：

- EMA teacher
- 定期刷新 reference teacher

但第 1 版不要求。

---

## 5. 数据范围

## 5.1 使用的数据

### A. text -> mask 主数据

- RefCOCO train
- RefCOCO+ train
- RefCOCOg train

### B. mask -> text 主数据

- DAM training data
- GAR `Seed-Dataset`
- GAR `Fine-Grained-Dataset`

### C. 暂不混入第一轮训练的数据

- GAR `Relation-Dataset`

它先作为 **hard validation / zero-shot generalization** 使用，不进入第 1 轮训练混合。

理由：第 1 版要先验证 joint + overlay teacher + OPD + RL 的主效应，relation-heavy 数据会增加 prompt 设计、reward 设计和数据格式复杂度。

---

## 5.2 数据混合策略

### Stage 1 joint SFT

建议初始采样比：

- 60% refseg bucket
- 40% maskcap bucket

其中：

- refseg bucket 内部：RefCOCO / RefCOCO+ / RefCOCOg 尽量均匀采样
- maskcap bucket 内部：DAM : GAR-Seed : GAR-Fine-Grained = 5 : 3 : 2

### Stage 2 OPD warm stage

沿用同样的数据混合比。

### Stage 3 RL stage

建议改成：

- 75% refseg bucket
- 25% maskcap bucket

原因：第 1 版的 RL 主信号来自 segmentation 端。

---

## 5.3 统一内部数据 schema

Codex 需要先把不同数据源统一成一个内部 schema。

推荐 JSONL / parquet 统一字段：

```json
{
  "id": "refcoco_000001",
  "task": "refseg",
  "source": "refcoco",
  "image_path": "...",
  "mask": {
    "format": "rle",
    "counts": "...",
    "size": [480, 640]
  },
  "query": "the man in the red shirt",
  "caption": null,
  "split": "train"
}
```

```json
{
  "id": "dam_000001",
  "task": "maskcap",
  "source": "dam",
  "image_path": "...",
  "mask": {
    "format": "rle",
    "counts": "...",
    "size": [720, 1280]
  },
  "query": null,
  "caption": "a silver spoon lying diagonally on the right side of the plate",
  "split": "train"
}
```

可选字段：

```json
{
  "meta": {
    "dataset_name": "...",
    "instance_id": "...",
    "category": "optional",
    "width": 640,
    "height": 480
  }
}
```

### 强制要求

- 内部统一把 mask 转成 **COCO RLE** 或 repo 已有的统一 mask object
- 不要在训练前离线生成 overlay 图；**overlay 在 dataloader / collator 里动态生成**
- 不要在统一 schema 中硬编码 SAMTok token string；mask token 由 tokenizer encode 阶段产生

---

## 6. 任务模板与 prompt 设计

> 下面是逻辑模板，不是最终硬编码字符串。
> Codex 必须先检查 repo 当前 conversation template / chat template / special token names。

## 6.1 text -> mask

逻辑输入：

```text
<image>
Please segment the region referred to by: "{query}".
Return only the region mask.
```

逻辑输出：

```text
<mask_tok_1><mask_tok_2>
```

要求：

- answer span 尽量只包含 mask token
- 如果 repo 默认会输出其他文本，训练时要明确截取 **最终 mask answer span** 用于 loss / RL / OPD

---

## 6.2 mask -> text

逻辑输入：

```text
<image>
The highlighted region / provided region mask indicates a target region.
Describe this region precisely in one sentence.
```

如果 repo 的 mask input 需要通过两枚离散 token 表示，则采用：

```text
<image>
Region: <mask_tok_1><mask_tok_2>
Describe this region precisely in one sentence.
```

逻辑输出：

```text
a small metal spoon on the right side of the white plate
```

caption 风格约束：

- 一句话
- 优先写类别 + 关键属性 + 必要关系
- 不写无关背景
- 避免 “maybe / probably / looks like” 这类不确定表达
- 尽量避免纯泛化描述，如 “an object”, “a person”, “something”

---

## 6.3 teacher prompt

teacher 与 student 的 prompt 完全一致。

唯一差别：

- student 输入原图 `I`
- teacher 输入 overlay 图 `I_overlay`

这样可以把 teacher 的额外优势限制在视觉输入，而不是文本模板上。

---

## 7. overlay teacher 设计

## 7.1 原则

teacher 图像必须：

1. 与原图**同分辨率**
2. 保持 full-image 坐标系
3. 只通过视觉高亮提供 GT 区域特权信息

不能做：

- crop-only teacher
- resize 到目标区域后再单独输入
- 改变图像坐标系导致 mask token 语义漂移

---

## 7.2 推荐 overlay 构造

输入：原图 `I`，GT mask `M`

输出：`I_overlay`

建议规则：

```python
# pseudo
I_overlay = I.copy()
I_overlay[~M] = I_overlay[~M] * darken_alpha      # 例如 0.35 ~ 0.5
I_overlay[boundary(M)] = blend(boundary_color)   # 可选红/绿边界线
I_overlay[M] = I[M]                              # mask 内保持原图
```

推荐默认值：

- `darken_alpha = 0.4`
- boundary 1~3 px
- boundary color 只用于 teacher，可固定单色

不要做：

- 对目标区域强烈着色导致失真
- 对图像做模糊裁剪后替换整图
- 存盘 overlay 图

---

## 7.3 两个任务里的 overlay 使用

### text -> mask

- student：原图 + query
- teacher：overlay 图 + 同一个 query
- overlay 用 **GT mask** 构造

### mask -> text

- student：原图 + mask input
- teacher：overlay 图 + 同一个 mask input
- overlay 也用 **同一 GT mask** 构造

---

## 8. 训练阶段总览

建议拆成 3 个正式阶段 + 1 个 smoke test。

---

## Stage 0. Smoke test（必须先做）

目标：只验证环境、数据、模型、mask encode/decode、评测脚本能跑。

### 要求

- 跑通 32~128 个样本的小 batch overfit
- 确认：
  - RefCOCO 数据能正常转成训练样本
  - DAM / GAR 数据能正常转成 maskcap 样本
  - model 能 forward / generate
  - mask token 能 encode / decode
  - RefCOCO eval / DLC-Bench eval 能跑通最小链路

### 通过标准

- loss 明显下降
- 对 32~128 样本基本可记忆
- 评测脚本无格式错误

> 若 Stage 0 不通过，禁止进入后续阶段。

---

## Stage 1. Joint SFT baseline

目标：先得到一个**可靠 joint baseline**，再引入 OPD 和 RL。

### Loss

```text
L_stage1 = L_ce_refseg + λ_cap_ce * L_ce_maskcap
```

建议初值：

- `λ_cap_ce = 1.0`

### 输出要求

训练后保存：

- `ckpt_stage1_joint_sft`

它有两个用途：

1. 作为 joint baseline
2. 作为后续 Stage 2 / 3 的 frozen teacher reference

### 记录指标

至少记录：

- RefCOCO val
- RefCOCO+ val
- RefCOCOg val
- DLC-Bench dev/val（若有）
- train loss 分 task 曲线

---

## Stage 2. Joint failed-only OPD warmup

目标：在不引入 RL 的情况下，先验证 **overlay teacher + failed-only OPD** 是否有稳定收益。

### 训练流程

对每个 batch：

1. 正常 teacher-forcing，算 `L_ce`
2. 让 student 在当前输入上 **on-policy sample** 一个输出
3. 根据任务计算 sample 的质量 / reward
4. 如果失败，则调用 frozen teacher 在 overlay 输入上对**同一条 student 轨迹**打 logits
5. 对 answer span 做 distillation

### 总损失

```text
L_stage2 = L_ce + λ_opd * L_opd
```

其中：

```text
L_ce = L_ce_refseg + λ_cap_ce * L_ce_maskcap
```

### Distillation span

- refseg：只蒸馏最终 2 个 mask token span
- maskcap：蒸馏 caption answer span

第 1 版不要蒸馏整段 conversation 前缀。

---

## Stage 3. Joint OPD + Segmentation-First RL

目标：把 RL 加进来，但保持工程可控。

### 总原则

- refseg：**主 RL 分支**
- maskcap：**弱 RL 辅助**（默认可开，权重小）
- failed-only OPD 继续保留
- teacher 只负责 token-level correction / weighting，不决定 reward 方向

### 总损失

```text
L_stage3 = λ_ce * L_ce + λ_rl_seg * L_rl_seg + λ_rl_cap * L_rl_cap + λ_opd * L_opd + β_kl * L_kl_ref
```

建议初值：

- `λ_ce = 0.3`
- `λ_rl_seg = 1.0`
- `λ_rl_cap = 0.1`
- `λ_opd = 0.3`
- `β_kl = 0.02`

如果 caption RL 不稳定：

- 直接把 `λ_rl_cap = 0`
- 保留 caption 的 CE + OPD

这仍然算“joint captioning + RL 的第 1 版”，因为 RL 已经进入整个 joint pipeline，只是主优化目标放在 segmentation 端。

---

## 9. failed-only OPD 细节

## 9.1 核心定义

给定 student sample `y_hat`，如果该样本在对应任务上的 reward 低于阈值，则认为是 fail sample。

只有 fail sample 才触发 OPD。

---

## 9.2 fail 判定

### text -> mask

计算：

- decoded mask `m_hat`
- `cIoU(m_hat, m_gt)`
- 可选 `exact_pair = 1[token_pair == gt_pair]`

建议 fail 规则：

```text
fail_seg = (cIoU < 0.5)
```

可选更严格：

```text
fail_seg = (0.8 * cIoU + 0.2 * exact_pair) < 0.55
```

### mask -> text

第一版不做复杂 judge。

建议 reward：

```text
R_cap = 0.6 * semantic_sim + 0.4 * rougeL_f1
```

其中：

- `semantic_sim`：sentence-transformer cosine，相似度归一化到 `[0,1]`
- `rougeL_f1`：对参考 caption 的 ROUGE-L F1

建议 fail 规则：

```text
fail_cap = (R_cap < 0.65)
```

如果不想额外下载 sentence embedding 模型，可退化成：

```text
R_cap = rougeL_f1
fail_cap = (rougeL_f1 < 0.45)
```

---

## 9.3 teacher 计算方式

对 fail sample：

1. 用 student sample 的相同 prefix / sampled tokens 对齐时间步
2. teacher 在 overlay 图上前向
3. 只取 answer span 的 logits

即：

```text
p_teacher(. | overlay_image, prefix_from_student)
p_student(. | original_image, prefix_from_student)
```

不要让 teacher 自己重新生成另一条不同序列再去蒸馏。

---

## 9.4 distillation objective

建议默认使用 **JSD** 或对称 KL，而不是单向 forward KL。

```text
L_opd = Σ_t w_t * JSD(p_teacher^t || p_student^t)
```

其中 `t` 只遍历 answer span。

### token weight

```text
w_t = conf_teacher_t * fail_gate
```

推荐：

```text
conf_teacher_t = clamp(1 - H(p_teacher^t) / log(V), 0, 1)
```

即：

- teacher entropy 高 -> 权重低
- teacher entropy 低 -> 权重高

这和 SRPO / RLSD 的直觉一致：

- 失败样本更适合 dense correction
- 不可靠 teacher target 不应强蒸馏

---

## 10. RL 设计

## 10.1 不实现 value model，不实现 PPO critic

第 1 版只实现一个**足够简单但是真正 on-policy 的 group-normalized policy gradient**。

也可以称作：

# **GRPO-lite**

理由：

- 比 PPO 更好接入现有 repo
- 不需要额外 value network
- 适合短 answer span（尤其是 2-token mask output）

---

## 10.2 refseg 的 RL

### rollout

对每个 prompt 采样 `G` 个候选，建议：

- `G = 4`

### reward

建议：

```text
R_seg = 0.8 * cIoU + 0.2 * exact_pair
```

其中：

- `cIoU`：decoded mask 与 GT 的 cIoU
- `exact_pair`：生成 token pair 是否与 GT token pair 完全相同

如果 exact pair 不稳定或多码本存在近义冲突，可直接用：

```text
R_seg = cIoU
```

### advantage

组内标准化：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
```

### policy loss

```text
L_rl_seg = - mean_i [ A_i * Σ_t log p_theta(y_i_t | x, y_i_<t) ]
```

其中 `t` 只取 mask answer span。

---

## 10.3 maskcap 的 RL

caption RL 只做**低权重辅助**。

### rollout

建议：

- `G = 2` 或 `G = 4`
- `max_new_tokens` 适中，比如 32~64
- 加 repetition penalty / no-repeat-ngram 以减少无效长输出

### reward

推荐轻量实现：

```text
R_cap = 0.6 * semantic_sim + 0.4 * rougeL_f1
```

其中：

- `semantic_sim`：MiniLM / bge-small 等轻量句向量 cosine，相似度映射到 `[0,1]`
- `rougeL_f1`：与参考 caption 的 ROUGE-L F1

可选 length regularization：

```text
R_cap = R_cap - 0.05 * overlong_penalty
```

### policy loss

```text
L_rl_cap = - mean_i [ A_i * Σ_t log p_theta(y_i_t | x, y_i_<t) ]
```

其中 `t` 只取 caption answer span。

### 默认建议

- 先把 caption RL 权重设小：`λ_rl_cap = 0.1`
- 若训练不稳定，直接设 `λ_rl_cap = 0`

---

## 10.4 RL 的 KL regularization

对 student 加 reference KL，reference 使用 `ckpt_stage1_joint_sft` 或 Stage 2 开始时保存的 checkpoint。

```text
L_kl_ref = Σ_t KL(p_ref^t || p_student^t)
```

span 范围：

- 默认只加在 answer span 上
- 若实现简单，也可对全生成 span 加，但第 1 版更推荐 answer span

---

## 11. joint 训练的关键实现点

## 11.1 batch 组织

建议 dataloader 返回统一 batch，但 task-aware collator 负责分支：

- `task=refseg`
- `task=maskcap`

为了减少实现复杂度，**一个 mini-batch 内可以只放同一种 task**，通过 mixed sampler 交替出 batch。

不要在同一个 batch 里混不同 task 的 answer decoding 逻辑，第一版太容易引入 bug。

---

## 11.2 训练时序

推荐时序：

1. Stage 0 smoke test
2. Stage 1 joint SFT
3. 保存 `ckpt_stage1_joint_sft`
4. Stage 2 joint failed-only OPD warmup
5. 保存 `ckpt_stage2_joint_opd`
6. Stage 3 joint OPD + seg-first RL
7. 保存 `ckpt_stage3_joint_opd_rl`

---

## 11.3 为什么先 Stage 2 再 Stage 3

因为如果直接上 RL + OPD，很难区分：

- 提升来自 overlay teacher
- 提升来自 failed-only routing
- 提升来自 RL

Stage 2 是非常关键的中间台阶，它能单独验证 overlay teacher + OPD 的价值。

---

## 12. 伪代码

## 12.1 Stage 2：joint failed-only OPD

```python
for batch in loader:
    # batch is homogeneous: either refseg or maskcap
    task = batch.task

    # 1) CE branch
    logits_gt = student.forward(batch.original_inputs, teacher_forcing=batch.gt_answer_tokens)
    L_ce = cross_entropy_on_answer_span(logits_gt, batch.gt_answer_tokens)

    # 2) on-policy sample
    sampled_tokens = student.generate(
        batch.original_inputs,
        do_sample=True,
        temperature=temp_map[task],
        top_p=top_p_map[task],
        max_new_tokens=max_len_map[task],
    )

    # 3) compute reward / fail gate
    if task == "refseg":
        pred_mask = decode_mask_from_tokens(sampled_tokens)
        reward = compute_seg_reward(pred_mask, batch.gt_mask)
        fail = reward < tau_seg
    else:
        pred_text = decode_text(sampled_tokens)
        reward = compute_cap_reward(pred_text, batch.gt_caption)
        fail = reward < tau_cap

    # 4) failed-only teacher distillation
    if fail:
        teacher_logits = teacher.forward(
            batch.overlay_inputs,
            forced_prefix=sampled_tokens
        )
        student_logits = student.forward(
            batch.original_inputs,
            forced_prefix=sampled_tokens
        )
        L_opd = jsd_on_answer_span(
            student_logits,
            teacher_logits,
            confidence_weight=True
        )
    else:
        L_opd = 0.0

    loss = L_ce + lambda_opd * L_opd
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 12.2 Stage 3：joint OPD + seg-first RL

```python
for batch in loader:
    task = batch.task

    # 1) teacher-forcing CE
    logits_gt = student.forward(batch.original_inputs, teacher_forcing=batch.gt_answer_tokens)
    L_ce = cross_entropy_on_answer_span(logits_gt, batch.gt_answer_tokens)

    # 2) sample group rollouts
    rollouts = sample_group(student, batch.original_inputs, group_size=group_size_map[task])

    rewards = []
    logprobs = []
    fail_flags = []
    opd_losses = []

    for y_hat in rollouts:
        if task == "refseg":
            pred_mask = decode_mask_from_tokens(y_hat)
            r = compute_seg_reward(pred_mask, batch.gt_mask)
            fail = r < tau_seg
        else:
            pred_text = decode_text(y_hat)
            r = compute_cap_reward(pred_text, batch.gt_caption)
            fail = r < tau_cap

        rewards.append(r)
        logprobs.append(answer_span_logprob(student, batch.original_inputs, y_hat))
        fail_flags.append(fail)

        if fail:
            t_logits = teacher.forward(batch.overlay_inputs, forced_prefix=y_hat)
            s_logits = student.forward(batch.original_inputs, forced_prefix=y_hat)
            opd_losses.append(jsd_on_answer_span(s_logits, t_logits, confidence_weight=True))
        else:
            opd_losses.append(0.0)

    advantages = normalize_group_rewards(rewards)
    L_pg = policy_gradient_loss(logprobs, advantages)
    L_opd = mean(opd_losses)
    L_kl_ref = reference_kl(student, ref_model, batch.original_inputs, rollouts)

    if task == "refseg":
        loss = lambda_ce * L_ce + lambda_rl_seg * L_pg + lambda_opd * L_opd + beta_kl * L_kl_ref
    else:
        loss = lambda_ce * L_ce + lambda_rl_cap * L_pg + lambda_opd * L_opd + beta_kl * L_kl_ref

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 13. 实现目录建议

所有新增代码放到一个清晰的新 project 目录里，不要污染原 repo 太多。

建议：

```text
projects/pixvl_idea1/
├── README.md
├── configs/
│   ├── idea1_joint_sft.py
│   ├── idea1_joint_opd.py
│   └── idea1_joint_opd_rl.py
├── datasets/
│   ├── unified_region_dataset.py
│   ├── adapters_refcoco.py
│   ├── adapters_dam.py
│   ├── adapters_gar.py
│   └── overlay_utils.py
├── trainers/
│   ├── joint_sft_trainer.py
│   ├── joint_opd_trainer.py
│   └── joint_opd_rl_trainer.py
├── rewards/
│   ├── seg_reward.py
│   ├── cap_reward.py
│   └── text_similarity.py
├── eval/
│   ├── eval_refseg.py
│   ├── eval_dlc.py
│   └── eval_joint_summary.py
└── scripts/
    ├── prepare_refseg_data.py
    ├── prepare_dam_data.py
    ├── prepare_gar_data.py
    ├── train_stage1_joint_sft.sh
    ├── train_stage2_joint_opd.sh
    ├── train_stage3_joint_opd_rl.sh
    └── run_all_eval.sh
```

---

## 14. Codex 的具体执行顺序

## Step A. Repo 审查

先做：

- 定位 Qwen3-VL + SAMTok model loader
- 定位 mask tokenizer encode/decode
- 定位 trainer / dataloader / eval 入口
- 定位 RefCOCO fine-tuning example

输出：

- 一份 `repo_audit.md`
- 写清楚真实文件路径、类名、配置入口

---

## Step B. 环境与数据下载脚本

Codex 需要先写：

- `scripts/setup_env.sh`
- `scripts/download_public_data.sh`
- `scripts/check_data_integrity.py`

要求：

- 支持下载 / 校验 RefCOCO/+/g
- 支持下载 / 校验 DAM training data
- 支持下载 / 校验 GAR Seed/Fine-Grained
- 输出统一目录结构

---

## Step C. 数据统一

Codex 需要先完成：

- 各数据集 adapter
- 统一 schema 导出
- small smoke split 导出（32 / 128 / 512 samples）

要求：

- 能打印前 3 条样本
- 能可视化 3 条 overlay teacher 图
- 能检查 mask decode 后是否和原 mask 对齐

---

## Step D. Stage 0 smoke test

要求：

- 32~128 样本过拟合
- 两个任务都单独能下降
- eval 脚本返回非空结果

没有这个结果，禁止进入 Step E。

---

## Step E. Stage 1 joint SFT

先产出 baseline。

输出：

- `ckpt_stage1_joint_sft`
- baseline metrics 表

---

## Step F. Stage 2 joint failed-only OPD

实现：

- overlay transform
- on-policy sampling
- fail gate
- teacher frozen branch
- answer-span JSD

至少做 3 个实验：

1. `joint_sft_baseline`
2. `joint_sft + all_sample_opd`
3. `joint_sft + failed_only_opd`

目的：先证明 failed-only routing 的必要性。

---

## Step G. Stage 3 RL

实现：

- group-normalized policy gradient
- refseg reward
- weak cap reward
- reference KL

至少做 2 个实验：

1. `failed_only_opd + seg_rl`
2. `failed_only_opd + seg_rl + weak_cap_rl`

---

## Step H. 汇总评测

输出：

- RefCOCO / RefCOCO+ / RefCOCOg
- DLC-Bench
- 训练耗时 / 显存 / tokens/sec
- 每个阶段的最优 checkpoint
- ablation summary

---

## 15. 评测方案

## 15.1 主指标

### Referring segmentation

- RefCOCO val / testA / testB
- RefCOCO+ val / testA / testB
- RefCOCOg val / test

指标使用 repo 官方 / 论文标准指标（通常为 cIoU 或相关 mask 指标），优先复用 Sa2VA / SAMTok 的现成 evaluation code。

### Mask captioning

- DLC-Bench 平均分 / official score
- 如 DLC-Bench evaluation repo 支持子类统计，一并保留

---

## 15.2 快速验证子集

为了缩短第一次出结果的时间，先跑：

- RefCOCO val
- RefCOCO+ val
- DLC-Bench val/dev

只要这三项趋势正确，再跑完整大表。

---

## 15.3 hard validation

第 1 版可以额外做一个轻量 hard validation：

- GAR Relation dataset 零样本评估

目的不是冲榜，而是看第 1 版的 overlay teacher + joint training 是否对关系理解有外溢收益。

---

## 16. 必做消融

必须按下面顺序做，不要跳。

### A0. Stage 1 joint SFT baseline

- joint 两任务
- 无 OPD
- 无 RL

### A1. + all-sample OPD

- teacher 用 overlay 图
- 所有样本都蒸馏

### A2. + failed-only OPD

- teacher 用 overlay 图
- 只有 fail sample 蒸馏

### A3. + failed-only OPD + seg RL

- 第 1 个完整主方法版本

### A4. + failed-only OPD + seg RL + weak cap RL

- joint + RL 完整版本

### A5. teacher ablation

- teacher 也看原图，不看 overlay

目的：验证 gain 是否真的来自 privileged overlay，而不是多一个 frozen branch 本身。

---

## 17. 成功标准

第 1 版成功，不要求立刻刷榜。

满足下面任一即可判定为正结果：

1. 相对 Stage 1 joint SFT baseline，RefCOCO val / RefCOCO+ val 稳定提升
2. 在 segmentation 提升的同时，DLC-Bench 不明显下降
3. `failed-only OPD > all-sample OPD`
4. `overlay teacher > raw-image teacher`

如果只出现：

- segmentation 涨、caption 大幅掉

则说明 joint pipeline 还不够平衡，但也仍然是有价值的负结果，需要保留日志与分析。

---

## 18. 风险点与规避

## 18.1 最大风险：mask token / decode 路径搞错

规避：

- Stage 0 先做 32/128 样本 overfit
- 单独测试 mask encode -> decode 的 round-trip
- 不要先碰 RL

---

## 18.2 overlay teacher 可能过强，导致 student 追不上

规避：

- 只在 fail sample 上蒸馏
- teacher entropy weighting
- Stage 2 先做 warmup
- `λ_opd` 不要一开始太大

---

## 18.3 caption reward 太脆弱

规避：

- caption RL 默认低权重
- 不稳定就先关掉 `λ_rl_cap`
- 仍保留 caption CE + OPD

---

## 18.4 RL 容易把输出压短或模式坍缩

规避：

- 只在 answer span 上做 policy gradient
- 维持 CE loss
- 保持 reference KL
- caption 端限制 max length 和 repetition

---

## 19. 推荐初始超参数

> 下面是起跑默认值，不是最终最优值。

## 通用

- precision: bf16
- grad checkpointing: on
- flash attention: on if repo supports
- optimizer: AdamW
- lr: 依照 repo 默认 finetune 配置起步
- save every: 500~1000 steps
- eval every: 500~1000 steps

## Stage 1

- task mix: 60 / 40
- `λ_cap_ce = 1.0`

## Stage 2

- `λ_opd = 0.3`
- `tau_seg = 0.5`
- `tau_cap = 0.65`
- sampling: `temperature=0.7`, `top_p=0.95`

## Stage 3

- group size: `G_seg=4`, `G_cap=2`
- `λ_ce = 0.3`
- `λ_rl_seg = 1.0`
- `λ_rl_cap = 0.1`
- `λ_opd = 0.3`
- `β_kl = 0.02`

如果资源吃紧：

- 先只开 `G_seg=2`
- 先关 `λ_rl_cap`

---

## 20. 对 Codex 的硬性要求

1. **先审 repo，再改代码**
2. **先做 Stage 0 smoke test，再做大训练**
3. **不要硬编码 SAMTok special token string**
4. **不要在第 1 版加入 evidence / crop teacher / cycle**
5. **不要离线保存 overlay 图，必须 online 生成**
6. **不要把不同 task 混在一个 batch 里做复杂分支**
7. **必须先产出 baseline，再做 OPD，再做 RL**
8. **每个阶段都要保存可复现 config、日志和 metrics JSON**

---

## 21. 期望产出文件

Codex 最终至少要产出：

```text
repo_audit.md
training_plan.md
metrics_stage1.json
metrics_stage2.json
metrics_stage3.json
ablation_summary.md
```

以及：

- 可运行训练脚本
- 可运行评测脚本
- 最终 config 文件
- 至少一个小规模成功 checkpoint

---

## 22. 如果第 1 版有效，下一步怎么扩

只有在第 1 版出现正收益后，才进入后续版本：

1. **+ evidence bottleneck**
2. **+ semantic / relation / geometry error routing**
3. **+ crop / multi-view privileged teacher**
4. **+ confuser consistency / retrieval**
5. **+ cycle reward**

顺序不要乱。

---

## 23. 这版 paper 的最小可写结论

如果第 1 版做出正结果，最小 paper story 可以写成：

> We show that a same-resolution privileged overlay teacher is already sufficient to improve joint pixel-language training in SAMTok-style MLLMs. Combined with failed-only on-policy distillation and segmentation-first RL, the method strengthens referring segmentation while retaining competitive mask captioning performance, without introducing extra test-time branches or new data annotation.

---

## 24. 公开资源参考（给 Codex 查用）

> 下面只是查阅入口，不要求在代码里硬写这些 URL。

- Sa2VA official repo
- SAMTok paper / code entry（在 Sa2VA codebase 下）
- DAM official repo
- DAM training data + DLC-Bench HF entry
- GAR official repo
- GAR dataset HF entry
- Qwen3-VL official repo

---

## 25. 本文档最后再强调一次

第 1 版只做这一件事：

# **joint two-task training + same-resolution overlay teacher + failed-only OPD + segmentation-first RL**

任何会显著增加数据构造、模型结构、坐标系不一致风险、或训练不稳定性的东西，全部放到后续版本。


---

## 26. 升级版总定位（V2 after MVP）

### 26.1 V2 方法名

# **CARE-OPD: Confidence-Aware Routed Evidence Distillation with Overlay Teacher**

### 26.2 一句话

在原有 **same-resolution overlay teacher + failed-only OPD + segmentation-first RL** 的基础上，进一步把样本分成 **clean / suspicious / corrupted** 三类，只对 **suspicious** 或 **被纠正后的 corrupted** 样本做 dense correction，并新增一个 **training-only short evidence auxiliary task**，把 teacher 的 privileged visual cue 压缩成 student 可学的 discriminative evidence。

### 26.3 这版升级要解决的三个问题

1. **binary fail gate 太粗**  
   现在的 fail / non-fail 二分类足够做 MVP，但对 pixel-level MLLM 来说，不同 fail 的可纠正性差别很大。  
   - 有些是 reward 边界附近的 **near-miss**，适合做 OPD。  
   - 有些是明显错对象、错关系、错边界的 **corrupted sample**，直接蒸馏很可能把 teacher 特权信息硬压给 student，反而不稳。  

2. **teacher 现在给的是“结果级” privileged signal，不是“证据级” privileged signal**  
   当前 teacher 直接看 overlay 图给 answer-span logits，这很强，但 novelty 还不够。  
   升级版的关键是：  
   > 让 teacher 先把 overlay 中真正有用的局部视觉信息压缩成一个极短的 evidence，再让 student 学 evidence 和 answer 的对应关系。  

3. **对最脏样本直接蒸馏，容易放大错误传播**  
   对明显错误的轨迹，正确做法不应该是“teacher 直接矫正答案”，而应该先做 **association correction / trajectory repair**，修到可学，再蒸馏。

---

## 27. 升级版的核心借鉴与边界

### 27.1 借鉴来源

这版升级主要吸收四类思路：

1. **SRPO / RLSD 风格的路由与 teacher 使用边界**  
   - 正确或接近正确的样本更适合 reward-aligned RL。  
   - fail sample 更适合 dense correction。  
   - teacher 最好只控制 **token-level correction magnitude / weighting**，不要直接决定 update direction。  

2. **CPL++ 风格的 suspicious association detection + self-correction**  
   - 不是所有低分样本都该硬蒸馏。  
   - 要先识别 suspicious / corrupted association。  
   - 对 corrupted sample 先做 correction，再训练。  

3. **PET-DINO 风格的 prompt diversity / memory cues bank**  
   - 不是把 detector 头搬过来。  
   - 只借鉴 **IBP / DMD 的 prompt-source diversity 与 memory bank 思想**。  
   - 训练时维护 high-confidence evidence bank，用来稳定 teacher 证据生成或做 prototype regularization。  

4. **MPO 风格的 multimodal prompt optimization**  
   - 不把 MPO 做成主方法。  
   - 只把它作为 **training-time prompt/overlay setting search**：  
     搜索 teacher prompt wording、overlay darken alpha、边界宽度、evidence 长度上限等。  

### 27.2 明确不借什么

升级版仍然**不建议**直接上下面这些东西作为主方法：

- 不把 crop teacher 变成主故事  
- 不把 detector-style retrieval head 大改到 student inference path
- 不要求 test-time 输出 evidence 才能完成主任务
- 不把 prompt search 写成主要 novelty

---

## 28. 升级版核心 1：Confidence-Aware Triage（从 binary fail 到三段式样本分流）

## 28.1 三类样本定义

给定 student 的 on-policy sample `y_hat`，不再只判断 fail / non-fail，而是划分为：

1. **clean**  
   - task reward 高  
   - answer-span NLL 低  
   - teacher confidence 高  
   - 双向一致性高  
   这类样本主要走 RL / CE，不触发 dense correction。

2. **suspicious**  
   - reward 中等或临界  
   - answer 不是完全错，但 teacher/student disagreement 明显  
   - 很可能是属性遗漏、关系词弱化、边界轻微偏差  
   这类样本是 **OPD + evidence distillation** 的主战场。

3. **corrupted**  
   - reward 明显低  
   - answer-span 不可信  
   - teacher/student 差异极大  
   - 很可能已经错对象、错关系、或掺杂严重边界错误  
   这类样本不应该立即蒸馏，先做 correction。

---

## 28.2 suspiciousness score

定义一个统一的可实现分数：

```text
q(x, y_hat) =
  λ_r * (1 - R_task)
+ λ_n * NLL_norm(answer_span)
+ λ_h * H_norm(teacher_answer)
+ λ_b * (1 - C_bidir)
```

其中：

- `R_task`
  - refseg: `R_seg`
  - maskcap: `R_cap`

- `NLL_norm(answer_span)`  
  对 answer span 的平均 token NLL 做 `[0,1]` 归一化。

- `H_norm(teacher_answer)`  
  teacher 在 answer span 上的平均 entropy，归一化到 `[0,1]`。  
  高 entropy = teacher 本身也不确定。

- `C_bidir`  
  当前 joint pixel-language 模型内部的**轻量双向一致性分数**。  
  这不是新主任务，只是 routing feature。

建议初值：

```text
λ_r = 0.45
λ_n = 0.20
λ_h = 0.15
λ_b = 0.20
```

### 阈值建议

不要一开始手写死阈值。  
升级版推荐：

- 用 Stage 2 / Stage 3 前 2k~5k step 的 rollout 统计 `q`
- 设：
  - `τ_clean = P60(q)`
  - `τ_corr  = P85(q)`

即：

```text
q <= τ_clean           -> clean
τ_clean < q <= τ_corr  -> suspicious
q > τ_corr             -> corrupted
```

这样更稳，也更方便跨数据源迁移。

---

## 28.3 轻量双向一致性 `C_bidir`

这项是升级版很重要、但不能做重的部分。  
只做 **cheap local back-check**，不把它升级成另一个大训练系统。

### 对 refseg 样本

已知 student rollout 得到 `m_hat`。  
做一个辅助 prompt：

```text
<image>
Region: <predicted_mask>
Give the minimal evidence that identifies this region.
Return a short phrase only.
```

得到 `e_back`。  
把 `e_back` 和原 query 的 atom 做对比：

```text
C_bidir_refseg = atom_F1(query_atoms, evidence_atoms(e_back))
```

### 对 maskcap 样本

已知 student rollout 得到 caption / evidence `t_hat`。  
在同一图上做一个 cheap reground：

```text
<image>
Please segment the region referred to by: "{short_evidence(t_hat)}".
Return only the region mask.
```

得到 `m_back`。  

```text
C_bidir_maskcap = IoU(m_back, m_gt)
```

### 工程建议

- 默认只在 **fail / near-fail** 样本上计算 `C_bidir`
- 或者只在 25% 的 batch 上计算，用 EMA 填平统计

这样不会把训练复杂度拉爆。

---

## 28.4 三类样本各自怎么训练

### clean

```text
loss_clean = CE + RL + KL_ref
```

- 不做 OPD
- 不做 self-correction
- teacher 最多只参与日志分析，不参与 update

### suspicious

```text
loss_suspicious = CE + RL + OPD_answer + EvidenceDistill + KL_ref
```

- 这是 upgraded OPD 的主更新区间
- 只蒸馏 answer span 与 evidence aux span
- teacher entropy 高时自动降权

### corrupted

先做 correction，再决定是否蒸馏。

```text
if corrected and reward_improved:
    loss_corrupted = CE + small_RL + OPD_answer + EvidenceDistill + CorrectionLoss
else:
    loss_corrupted = CE + small_RL + KL_ref
```

重点是：

> corrupted sample 默认不直接进 OPD；  
> 只有在 correction 通过后，才允许它进入蒸馏通道。

这和 SRPO / RLSD 的边界是一致的：  
teacher 不是直接替 student 决定策略方向，而是把“哪些 token 该重点修、修多大”这件事做细。

---

## 29. 升级版核心 2：Evidence Bottleneck（training-only auxiliary task）

## 29.1 为什么 evidence bottleneck 值得加

当前 overlay teacher 的特权来自：
- 它知道 GT region 在哪
- 它看的是 high-contrast visual prompt 图

但如果 teacher 直接只给最终 answer logits，novelty 和可解释性都还不够强。

升级版更强的做法是：

> 让 teacher 先把“为什么是这个区域”的视觉证据压成一个短 evidence，  
> 再让 student 在原图上学习这个 evidence。

这比直接把 full answer 蒸馏给 student 更合理，因为：

1. evidence 是 teacher privilege 的“最小充分统计量”
2. evidence 比 full caption / full answer 更短、更稳
3. evidence 可以在 refseg 和 maskcap 两边共享，增强 joint consistency

---

## 29.2 evidence 的形式

evidence 不是长 caption。  
默认只用**极短短语**，强调 discriminative cue。

### refseg 侧 evidence

输入：

```text
<image>
Expression: "{query}"
Give the minimal evidence that uniquely identifies the referred region.
Return a short phrase only, <= 8 tokens.
```

目标形式示例：

```text
red shirt, left man
black phone in hand
small spoon on plate
```

### maskcap 侧 evidence

输入：

```text
<image>
Region: <mask>
Summarize the target region with a short discriminative phrase.
Return <= 10 tokens.
```

目标形式示例：

```text
silver spoon, right side
woman with blue hat
rear white bus
```

### 强约束

- 不要求成句
- 不要求 fluent prose
- 只要求 **category + 1~2 个真正有区分度的 cue**
- 不在 test-time 强制输出

---

## 29.3 evidence 是 training-only auxiliary task，不改主推理接口

升级版默认**不改**主任务接口：

- text -> mask 仍然输出 mask token
- mask -> text 仍然输出 caption

evidence 只通过下面两种方式进入训练：

1. **额外 auxiliary prompt**
2. **suspicious / corrected-corrupted 样本上的 evidence distillation**

即：

```text
L_total = L_main + λ_ev * L_evidence
```

其中 `L_evidence` 只在需要时触发。

这样 inference path 不变，评测脚本也不用重写。

---

## 29.4 evidence teacher 怎么拿到目标

### 默认做法：online teacher evidence

- student 看原图
- frozen teacher 看 overlay 图
- teacher 在 evidence prompt 上生成 `z_teacher`

### 更稳的做法：cache evidence

当 Stage 4 开始后，可以先用当前 best teacher：

- 对训练集跑一遍 overlay-evidence generation
- 把高 confidence 的 `z_teacher` 缓存起来
- 后续训练优先读 cache，减少 online 开销

cache 中建议保存：

```json
{
  "sample_id": "...",
  "task": "refseg or maskcap",
  "teacher_evidence": "...",
  "teacher_entropy": 0.18,
  "teacher_reward_est": 0.91
}
```

---

## 29.5 evidence loss

建议使用两部分：

```text
L_evidence = L_ev_ce + λ_ev_jsd * L_ev_jsd
```

- `L_ev_ce`
  - student 在 evidence prompt 上对 teacher evidence 或 cached evidence 做 CE

- `L_ev_jsd`
  - 对 suspicious sample，在同一 prefix 下比较 teacher/student 的 evidence token 分布

建议：
- clean 样本不启用
- suspicious 样本正常启用
- corrupted 样本只有 correction 后才启用

---

## 30. 升级版核心 3：Self-Correction for Corrupted Samples

## 30.1 总原则

对 corrupted sample，第一目标不是蒸馏，而是：

> 先把轨迹修到“可学”，再蒸馏。

这里借鉴的是 CPL++ 的思想，但要改成适配 joint pixel-level MLLM 的版本。

---

## 30.2 refseg 的 correction

对于 `text -> mask` 的 corrupted rollout：

1. student 先生成错误 mask `m_hat`
2. teacher 在 overlay 图上生成 short evidence `z_teacher`
3. student 在原图上，条件化这个 evidence 重新做一次 mask decoding：
   - 方式 A：新 auxiliary prompt  
     `"Based on the expression and this short evidence, return only the mask."`
   - 方式 B：把 evidence 拼到 query 后做 constrained re-decode

4. 若新 mask `m_corr` 满足：

```text
R_seg(m_corr) >= R_seg(m_hat) + δ_corr
```

则接受 corrected trajectory

建议：
- `δ_corr = 0.05 ~ 0.10`
- 每个 corrupted sample 最多 correction 1 次，不要反复重采样

---

## 30.3 maskcap 的 correction

对于 `mask -> text` 的 corrupted rollout：

1. 生成 `t_hat`
2. teacher 给出 `z_teacher`
3. 把 `t_hat` 与 `z_teacher` 做 atom 对齐，识别：
   - 缺失 atom
   - 冲突 atom
   - 泛化 atom（如 object / person / thing）

4. 只重写这部分 atom，不整句推倒重来

推荐 prompt：

```text
Keep the main object category unchanged.
Rewrite only the missing or wrong discriminative cues.
Return one short sentence.
```

若：

```text
R_cap(t_corr) >= R_cap(t_hat) + δ_cap
```

则接受 correction。

---

## 30.4 correction loss

推荐最简单做法：

- correction 本身当作 **data improvement**
- 不额外设计复杂 RL

也就是：

```text
if corrected:
    use y_corr as the trajectory for OPD / CE / RL
else:
    keep original sample but skip OPD
```

如果后续要更强，可以加一个 pairwise preference：

```text
y_corr ≻ y_hat
```

用 DPO / pairwise CE 做轻量 preference fine-tuning。

---

## 31. 可选升级：Evidence Cues Bank（只在 CARE-OPD 稳定后再开）

## 31.1 为什么值得做

很多 suspicious sample 不是“完全没学会”，而是缺少一个稳定的 reference cue。  
PET-DINO 的启发是：

- prompt 不应该只来自当前样本
- 训练中应该维护一个 **memory-driven prompt source**

对本项目最合理的映射不是 detector prompt，而是：

> **teacher-side evidence bank**

---

## 31.2 bank 存什么

对高 confidence 样本，缓存：

```json
{
  "sample_id": "...",
  "task": "...",
  "region_embedding": "...",
  "query_atoms": ["red", "shirt", "left"],
  "teacher_evidence": "red shirt, left man",
  "reward": 0.93
}
```

建议 key：

- region embedding
- atom bag
- dataset source

建议 value：

- short evidence text
- teacher confidence
- reward
- optional prototype id

---

## 31.3 bank 怎么用

### 方式 A：retrieve-as-prototype

对 suspicious sample，检索 top-k 相似 bank entry。  
把这些 entry 的 evidence embedding 平均成 prototype，作为：

- evidence loss 的额外 prototype target
- correction 时的候选 evidence prior

### 方式 B：retrieve-as-example（可选）

把 1~2 个最相似 evidence 作为 in-context cue 给 teacher evidence prompt。  
但这会让 prompt 变长，默认不作为第一实现。

### 建议

先做 **prototype regularization**，不要先做 in-context retrieval。

---

## 32. 可选升级：MPO-lite（仅用于 training-time prompt / overlay setting search）

## 32.1 为什么这里适合借 MPO

对 idea1 来说，teacher 输入本质上就是一个 **multimodal prompt**：

- 文本 instruction
- visual overlay 形式
- evidence length 约束

这非常适合一个小规模的 training-time prompt search。

---

## 32.2 搜索空间

MPO-lite 只搜索下面这些 **离散低维设置**：

1. overlay darken alpha  
   - `0.30 / 0.40 / 0.50`

2. boundary width  
   - `0 / 1 / 2 / 3 px`

3. teacher evidence prompt wording  
   - `minimal evidence`
   - `discriminative cue`
   - `short identifying phrase`

4. evidence max length  
   - `6 / 8 / 10 tokens`

5. correction prompt wording  
   - rewrite missing cue
   - rewrite wrong cue
   - keep category, fix details

### 目的

不是把 MPO 写成 main method，  
而是用一个小预算 search，找到更稳的 overlay-teacher 设置。

---

## 32.3 推荐实施方式

- 用 RefCOCO val + DLC small val 做 30~50 个 setting 的小 search
- 评价目标：
  - Stage 2 gain
  - evidence quality
  - OPD stability
- 固定最优 setting 后，再做正式训练

---

## 33. 升级版训练阶段建议（在原 Stage 0~3 成功后）

### Stage 4. Calibration + evidence cache

目标：
- 统计 `q(x, y_hat)` 分布
- 自动确定 `τ_clean / τ_corr`
- 构建第一版 teacher evidence cache

输出：
- `triage_stats.json`
- `cached_teacher_evidence.jsonl`

---

### Stage 5. CARE-OPD without correction bank

目标：
- 只加 triage + evidence bottleneck
- 暂不开 bank
- 暂不做 correction DPO

总损失建议：

```text
L_stage5 =
  λ_ce * L_ce
+ λ_rl_seg * L_rl_seg
+ λ_rl_cap * L_rl_cap
+ λ_ans_opd * L_opd_answer
+ λ_ev * L_evidence
+ β_kl * L_kl_ref
```

建议初值：

```text
λ_ans_opd = 0.25
λ_ev      = 0.20
```

---

### Stage 6. + self-correction

目标：
- 把 corrupted sample 修后再蒸馏

总损失：

```text
L_stage6 = L_stage5 + λ_corr * L_corr
```

其中 `L_corr` 最简单可以只是：

- corrected trajectory 替代原 trajectory 后的主损失
- 不额外写独立 loss

---

### Stage 7. + optional evidence bank / MPO-lite

目标：
- 做最后的稳定性和 hard-slice 增益

注意：
- bank 与 MPO-lite 是 enhancement，不是必须项
- 如果 Stage 6 已经有明显收益，不要为了复杂度强加 Stage 7

---

## 34. 升级版伪代码（核心逻辑）

```python
for batch in loader:
    task = batch.task

    # 1) main CE branch
    logits_gt = student.forward(batch.original_inputs, teacher_forcing=batch.gt_answer_tokens)
    L_ce = ce_on_answer_span(logits_gt, batch.gt_answer_tokens)

    # 2) on-policy rollout
    y_hat = student.generate(batch.original_inputs, do_sample=True, ...)

    # 3) compute task reward
    if task == "refseg":
        reward = compute_seg_reward(decode_mask(y_hat), batch.gt_mask)
    else:
        reward = compute_cap_reward(decode_text(y_hat), batch.gt_caption)

    # 4) compute triage features
    nll_ans = answer_span_nll(student, batch.original_inputs, y_hat)
    teacher_ans_logits = teacher.forward(batch.overlay_inputs, forced_prefix=y_hat)
    teacher_entropy = entropy_on_answer_span(teacher_ans_logits)
    c_bidir = cheap_bidirectional_check(student, batch, y_hat, task)

    q = (
        lambda_r * (1 - reward)
        + lambda_n * normalize(nll_ans)
        + lambda_h * normalize(teacher_entropy)
        + lambda_b * (1 - c_bidir)
    )

    route = triage(q, tau_clean, tau_corr)

    if route == "clean":
        L_opd = 0.0
        L_ev = 0.0
        y_train = y_hat

    elif route == "suspicious":
        y_train = y_hat
        L_opd = jsd_answer_span(student, teacher, batch, y_train, entropy_weight=True)
        L_ev = evidence_distill(student, teacher, batch, y_train)

    else:  # corrupted
        y_corr, improved = self_correct(student, teacher, batch, y_hat, task)
        if improved:
            y_train = y_corr
            L_opd = jsd_answer_span(student, teacher, batch, y_train, entropy_weight=True)
            L_ev = evidence_distill(student, teacher, batch, y_train)
        else:
            y_train = y_hat
            L_opd = 0.0
            L_ev = 0.0

    L_pg = maybe_policy_gradient(student, batch.original_inputs, y_train, task)
    L_kl = reference_kl(student, ref_model, batch.original_inputs, y_train)

    loss = (
        lambda_ce * L_ce
        + lambda_rl(task) * L_pg
        + lambda_ans_opd * L_opd
        + lambda_ev * L_ev
        + beta_kl * L_kl
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 35. 升级版必须做的新增消融

在保留原 A0~A5 的基础上，新增：

### B1. binary fail vs clean/suspicious/corrupted triage

目的：
- 证明三段式分流比原始 failed-only 更稳

### B2. no `C_bidir` vs with `C_bidir`

目的：
- 验证 cheap joint consistency 是否真的能提升 routing 质量

### B3. no evidence distill vs evidence distill

目的：
- 验证 privileged overlay signal 压成 evidence 是否比只蒸馏 answer 更有效

### B4. direct OPD on corrupted vs correction-then-OPD

目的：
- 验证 corrupted sample 必须先修再蒸馏

### B5. no bank vs prototype bank

目的：
- 验证 bank 是否只是锦上添花，而不是核心依赖

### B6. fixed overlay/prompt vs MPO-lite searched setting

目的：
- 验证 prompt/overlay search 是否值得作为最终 enhancement

---

## 36. 升级版 success criteria

如果升级版成立，希望至少看到下面两个层面的结果：

### 主表层面

1. 相比原 Stage 3：
   - RefCOCO / RefCOCO+ / RefCOCOg 再涨
   - DLC-Bench 持平或小涨

2. `triage + evidence` 明显优于：
   - failed-only binary routing
   - 只蒸馏 answer span

### hard slice 层面

建议额外汇报：

- attribute-heavy RefCOCO+ 子集
- relation-heavy GAR subset / Ref-Adv subset
- boundary-hard small-object 子集

理想结果：

- attribute / relation slice 有更明显收益
- 而不是只在 easy overall val 上涨

---

## 37. 对 Codex 的新增硬性要求（升级版）

1. **先复现原 Stage 0~3，不要直接跳 V2**
2. **triage 阈值默认从 rollout 统计自动估计，不要手写死**
3. **evidence 默认做 training-only auxiliary，不改主推理接口**
4. **corrupted sample 未 correction 成功时，默认跳过 OPD**
5. **MPO-lite 只做小预算 setting search，不得演化成主项目**
6. **prototype bank 先离线实现，不要一开始做复杂 online retrieval**
7. **升级版所有新增指标都要按 task 和 route type 分开记日志**

---

## 38. 本升级版最小 paper story

如果升级版有效，这篇 paper 的最终故事可以从：

> same-resolution overlay teacher + failed-only OPD + segmentation-first RL

升级为：

> A privileged overlay teacher is most useful when its signal is **compressed into short evidence** and **routed only to suspicious trajectories**.  
> With confidence-aware triage, training-only evidence bottleneck distillation, and correction-before-distillation for corrupted samples, a SAMTok-style joint pixel-level MLLM becomes more stable and more discriminative on both mask generation and region description, without any extra test-time branch.

---

## 39. 最后再强调一次升级优先级

### 必须先做
1. triage
2. evidence auxiliary task
3. correction-before-OPD

### 再考虑做
4. evidence bank
5. MPO-lite

### 仍然不建议抢跑做
6. crop teacher 主线
7. confuser-heavy retrieval head 主线
8. test-time evidence chain
9. 大规模 prompt search

升级版的真正中心仍然是：

# **confidence-aware routed OPD + training-only evidence bottleneck + correction-before-distillation**

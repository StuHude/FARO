# Idea 1 实施规格书（for Codex）

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


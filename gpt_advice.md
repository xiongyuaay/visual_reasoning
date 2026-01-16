修改指南：Latent Warm-up（A）→ 动态停机（C）

0. 目标与约束（必须满足）
	1.	训练 warm-up（方案 A）：每个样本有 M 张“思维链图片”，模型必须为每张图片分配固定的 latent 数 L_i，总 latent 数 K=\sum_i L_i。
	2.	训练后期（方案 C）：每张图片的 latent 仍有预算上限 B_i=L_i，但模型可在 0 \le L_i' \le B_i 内自回归决定实际使用数量（动态停机）。
	3.	latent token 只替换 think 部分：不引入额外“思考文本 token”；think 部分只由 latent_token_id 占位。
	4.	对齐：latent 段与对应图片（以及其大小/patch 数）发生约束；warm-up 强对齐，切换后保留对齐 + 增加计算成本/停机机制。
	5.	实现必须可训练：不要引入“变长序列拼接重打包”这种大改；允许保留固定占位 token（预算上限），通过 mask/gating 实现“未使用 latent 对后续无影响”。

⸻

1. 需要新增/修改的配置项（不可缺）

在 config（或训练参数）新增以下字段，名称必须一致：
	•	latent_token_id: int（你已有）
	•	latent_budget_alpha: float（例如 0.02）
	•	latent_budget_gamma: float（例如 0.5，次线性）
	•	latent_budget_min: int（例如 1）
	•	latent_budget_max: int（例如 32）
	•	latent_mode_stage: str ∈ { "A", "C" }
	•	latent_stop_loss_weight: float（例如 1.0）
	•	latent_cost_weight: float（例如 1e-3）
	•	latent_stop_threshold: float（推理阈值，例如 0.5）
	•	latent_null_scale: float（未使用 latent 的 KV 抑制强度，建议 0.0=完全置零）

规则：
	•	warm-up 使用 latent_mode_stage="A"
	•	切换后使用 latent_mode_stage="C"

⸻

2. 数据与 Collator 必须输出的额外字段（关键）

你必须在 batch 里提供 “每张思维链图片对应的预算上限” 以及 “latent 段与图片的对应关系”。不要在 model forward 内猜测段边界；要在 collator 里显式给出来，避免歧义。

2.1 Collator 输出新增字段（必须）

每个 batch 输出（除原有 input_ids/pixel_values/image_grid_thw 等）增加：
	•	latent_image_counts: List[int]，长度为 batch_size
	•	每个样本思维链图片数 M（不含 question image，按你的定义统一）
	•	latent_budgets: List[List[int]]，形状 [B][M]
	•	每张图片预算上限 B_i（warm-up 时也等于定长 L_i）
	•	latent_segments: List[List[Tuple[int,int,int]]]，形状 [B][num_segments]
	•	每个 segment 三元组 (start_pos, end_pos, img_idx)
	•	表示 input_ids 中 latent_token_id 连续区间 [start_pos, end_pos) 对应第 img_idx 张图片
	•	规则：每张图片对应 exactly 一个 segment
	•	latent_budget_total: List[int]，形状 [B]
	•	该样本总预算 K=\sum_i B_i

约束：
	•	warm-up（A）必须满足：每个 segment 长度 end_pos-start_pos == latent_budgets[b][img_idx]
	•	动态（C）仍保持 segment 为预算长度（占位），但实际使用长度由模型 stop 决定。

⸻

3. 预算计算（方案 A）必须采用的唯一公式

在 collator 中，根据每张图片“视觉 token 数”计算预算，不要用原图分辨率，用模型侧的 grid_thw 最稳定。

3.1 从 image_grid_thw 拆分到每个样本的每张图

你代码中已有：

split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()

但这给的是“每张图 embed 序列长度”，你需要按样本的 image_nums 拆分到每个样本。

做法（collator 侧）：在样本级别你通常就知道每个样本有几张图；把对应的 grid_thw 列表与之对应即可。

3.2 预算函数（唯一版本）

对第 i 张图：
	•	N_i = (t*h*w) // (spatial_merge_size**2)（与 split_sizes 一致口径）
	•	B_i = clip(ceil(alpha * (N_i ** gamma)), min, max)

⸻

4. Model 代码修改点总览（按你贴的文件结构）

你贴的核心文件相当于一个 patched modeling_qwen2_5_vl.py。修改点如下：
	1.	在 Qwen2_5_VLModelOutputWithPast 增加字段（必须）
	2.	在 Qwen2_5_VLModel.__init__ 增加 latent_stop_head（用于方案 C）
	3.	在 Qwen2_5_VLAttention.forward 增加对 latent KV 的 gating（保证“未使用 latent 不影响后续”）
	4.	在 Qwen2_5_VLModel.forward(latent_mode=True) 内实现：
	•	warm-up（A）：严格使用 full budget（全部 latent 都 active）
	•	动态（C）：逐步生成 latent，预测 stop，生成 active_mask，并输出给第二次 forward 使用
	5.	在 Qwen2_5_VLModel.forward(latent_mode=False) 的 CE forward 中：
	•	只 patch active latents
	•	对未使用 latents 施加 KV gating（通过 kwargs 传入 latent_kv_mask）

⸻

5. 具体修改步骤（逐条执行，禁止跳过）

5.1 修改输出结构：新增 active mask 与 stop 相关统计

在 Qwen2_5_VLModelOutputWithPast(ModelOutput) 里新增：
	•	latent_kv_mask: Optional[torch.BoolTensor] = None  # [B, S] True=保留KV，False=抑制
	•	latent_stop_probs: Optional[List[List[float]]] = None  # 仅用于调试/日志
	•	latent_used_lens: Optional[List[List[int]]] = None  # [B][M] 每段实际使用长度（C）

同时在 Qwen2_5_VLCausalLMOutputWithPast 里也加同名字段（可选但建议）。

⸻

5.2 增加 stop head（方案 C 必须）

在 Qwen2_5_VLModel.__init__ 中加：
	•	self.latent_stop_head = nn.Linear(config.text_config.hidden_size, 1, bias=True)

规则：stop head 输入使用 latent token 的 last_hidden_state（或你更偏好用其 embedding），下面统一按 step_out.last_hidden_state[0,0]。

⸻

5.3 Attention 层：加入 latent KV gating（核心保证）

在 Qwen2_5_VLAttention.forward(...) 中，在 key_states/value_states 得到后、进入 attention_interface 之前加入：
	•	从 kwargs 读入 latent_kv_mask（形状 [B,S]，bool）
	•	将被屏蔽位置的 key_states/value_states 置零
	•	同时在 attention_mask 上对这些 key 位置施加 -inf（推荐做法：直接在 logits 里加 mask，更稳定）

最低侵入版本（必须实现其一，推荐版本 1）：

版本 1（推荐）：在 logits 前加 key mask（无需构造4D大mask）
在 attention_interface 是 eager 时你能控制 logits；但你现在用 ALL_ATTENTION_FUNCTIONS，不同实现不好插。为了统一，采取 缩放 K/V 为 0 + 额外给 attention_mask 加列屏蔽。

实现方式：
	1.	先做 K/V 置零（所有实现都生效）：

latent_kv_mask = kwargs.get("latent_kv_mask", None)
if latent_kv_mask is not None:
    # latent_kv_mask: [B,S] bool, True keep, False drop
    m = latent_kv_mask.to(key_states.device).to(key_states.dtype)  # [B,S]
    m = m[:, None, :, None]  # [B,1,S,1]
    key_states = key_states * m
    value_states = value_states * m

	2.	再对 attention_mask 做列屏蔽（仅当 attention_mask 是 additive 4D 或 3D 时）：

	•	若 attention_mask 是 [B,1,S,S]：对 attention_mask[:,:,:,~mask] = -inf
	•	若 attention_mask 是 [B,S,S]：对 attention_mask[:,:,~mask] = -inf
	•	其他情况（None/2D padding）直接不处理（但 KV 已置零，仍能显著降低影响）

规则：-inf 用 torch.finfo(dtype).min 或固定 -1e9（bf16/FP16 推荐 -1e9）。

⸻

5.4 latent_mode=True：warm-up A 与动态 C 的统一输出

你当前 latent_mode 里是“按 latent token 位置跑 segment forward + step forward”。你要加入来自 collator 的 latent_segments 与 latent_budgets，并生成 latent_kv_mask。

5.4.1 forward 签名新增参数（必须）
在 Qwen2_5_VLModel.forward(... latent_mode=True ...) 增加参数：
	•	latent_segments: Optional[List[List[Tuple[int,int,int]]]] = None
	•	latent_budgets: Optional[List[List[int]]] = None

并在 Qwen2_5_VLForConditionalGeneration.forward 中透传。

5.4.2 入口检查（必须，避免歧义）
在 if latent_mode: 分支开头加入：
	•	断言 latent_segments、latent_budgets 非空
	•	对每个 segment 检查 [start,end) 都是 latent_token_id
	•	检查 segment 与 budgets 对应一致（img_idx 不越界）

warm-up A 还要额外断言：end-start == latent_budgets[b][img_idx]

5.4.3 构造 latent_kv_mask（两阶段都需要）
初始化：
	•	latent_kv_mask = torch.ones((B, S), dtype=torch.bool, device=device)

然后对每个 segment 做：
	•	warm-up A：该 segment 全部 True（保持）
	•	动态 C：根据 stop 决定实际使用长度 used_len，将 [start+used_len, end) 置为 False

并将其写入输出 Qwen2_5_VLModelOutputWithPast.latent_kv_mask。

⸻

5.5 动态 C：stop 逻辑（确定版本，禁止自由发挥）

5.5.1 生成方式（唯一规定）
对每个 segment [s,e)：
	•	从 pos = s 开始逐个 latent step forward（你已在做）
	•	每生成一个 latent hidden h，计算 stop prob：

p_stop = torch.sigmoid(self.latent_stop_head(h)).item()

	•	采用阈值停机（训练和推理都一致）：
	•	若 p_stop >= latent_stop_threshold，则 used_len = k+1 并停止该 segment
	•	若到达 e-1 仍未停，则 used_len = B_i

训练稳定性要求：动态阶段的前几千 step 可以把阈值设得更高（例如 0.9 → 0.5），但这是训练脚本调度，不改模型逻辑。

5.5.2 stop loss（必须）
为了让 stop 可学习，你必须引入监督信号。你既然有对齐数据集，最简单且无歧义的监督是：
	•	warm-up A：不训练 stop（latent_stop_loss_weight=0 或跳过）
	•	切换到 C：用 预算函数期望长度 作为 pseudo label：
	•	target_used_len = latent_budgets[b][img_idx] 的某个比例（例如 70%）或直接等于 budgets（会导致不愿停）
	•	更推荐：用固定比例 target = max(1, round(rho * budget))，rho=0.6~0.8（写死一个默认值，比如 0.7）

并训练 stop head 在第 target 个 latent 处输出 stop=1，之前 stop=0：
	•	对 segment 内每步 k：
	•	label = 0 if k < target-1 else 1 at k == target-1
	•	loss 用 BCEWithLogitsLoss（逐步平均）

这一步如果你已有“真实 CoT 图片数量/难度”对应的真实 used_len 标签，就用真实标签替换 pseudo；但此指南默认你没有真实 used_len 标签。

5.5.3 cost loss（必须）
动态 C 增加计算成本项：

\mathcal{L}_{cost} = \lambda \cdot \sum_i used\_len_i

以 latent_cost_weight 为 \lambda。实现时直接用整数 used_len 之和（不做期望）。

⸻

6. latent_mode=False（第二次 forward / CE forward）的必要改动

你现在逻辑是：若传入 ce_patch_pos/ce_patch_vec，则把对应位置 embeddings 替换，然后走 self.language_model(...)。

你要做两件事：
	1.	只 patch active latent（动态 C 停机后未使用的 latent 不 patch）
	2.	把 latent_kv_mask 透传到 attention，从而抑制未使用 latent 的 KV

6.1 只 patch active latent（必须）

在 if ce_patch_pos is not None and ce_patch_vec is not None: 里：
	•	保持原逻辑，但要求 collator/model 在动态 C 时只把 active latent 的 pos/vec 放进去
	•	未 active 的 latent 位置不要出现在 ce_patch_pos

6.2 透传 latent_kv_mask（必须）

在调用 self.language_model(...) 时，传入：
	•	latent_kv_mask=outputs.latent_kv_mask（来自 latent_mode 输出）
	•	或者如果 latent_mode=False 时直接给 latent_kv_mask 入参并透传（看你训练管线如何组织）

然后在 Qwen2_5_VLAttention.forward 按 5.3 实现 gating。

⸻

7. 训练调度（A→C）的唯一执行流程

7.1 Warm-up A（阶段 A）
	•	latent_mode_stage="A"
	•	latent_stop_loss_weight=0
	•	latent_cost_weight=0
	•	collator 生成 严格等于 budgets 的 latent segments（segment 长度=budget）
	•	模型：
	•	latent_mode=True：生成所有 latent（不早停），输出 latent_kv_mask=全 True
	•	latent_mode=False：patch 全部 latent，正常算 CE + alignment（你已有 alignment）

训练若干 epoch/steps 后切换到 C（由训练脚本控制）。

7.2 Dynamic C（阶段 C）
	•	latent_mode_stage="C"
	•	latent_stop_loss_weight>0
	•	latent_cost_weight>0
	•	collator 仍生成 segment 长度=budget（占位）
	•	模型：
	•	latent_mode=True：逐步生成并按 stop 阈值决定 used_len，输出 latent_kv_mask
	•	latent_mode=False：只 patch active latents + 通过 latent_kv_mask 抑制 inactive latents

最终总 loss：

\mathcal{L} = \mathcal{L}_{ce} + w_{align}\mathcal{L}_{align} + w_{stop}\mathcal{L}_{stop} + w_{cost}\mathcal{L}_{cost}

其中 w_align 你已有；w_stop=latent_stop_loss_weight；w_cost=latent_cost_weight。

⸻

8. 必须新增的单元测试/断言（防止 silent bug）

在训练开始前跑一次 batch 做这些检查（必须）：
	1.	latent_segments 覆盖的区间内 input_ids==latent_token_id 全为 True
	2.	warm-up A：每个 segment 长度 == budget
	3.	dynamic C：used_len ≤ budget，且 latent_kv_mask 在 segment 内呈现 True...True False...False
	4.	将 latent_kv_mask 全 False（把所有 latent KV 抑制），模型输出应近似退化为“不带 think”的效果（用于 sanity）

⸻

9. 推理（Inference）执行方式（明确规定）

推理必须分两步：
	1.	latent_mode=True 预填充：给定输入与预算 segments（由预算函数从图片 token 数算出来），模型生成 latents 并决定每段 used_len，输出：
	•	ce_patch_pos/ce_patch_vec
	•	latent_kv_mask
	2.	latent_mode=False 主生成：用 patch + kv_mask 跑 generate(...) 或你现有 decode

禁止：推理时只跑一次 forward 想同时生成 latent + answer（你现在的结构需要两阶段）。
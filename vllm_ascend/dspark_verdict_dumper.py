"""Per-slot accept/reject dumper for DSpark speculative decoding (failure-mode analysis).

WHAT IT RECORDS AND WHY. The serve already exposes aggregate per-position accept rates via
``/metrics``, which answers "how often does slot k survive" but not "on WHICH tokens, and was
the target even sure". This dumper writes one row per DRAFTED SLOT so both questions become
joinable against the generated text.

★ THE WHOLE BLOCK, NOT JUST THE FIRST MISMATCH. Verification runs the target over every draft
slot in ONE forward, so ``target_argmax`` for the slots AFTER the rejection point is already
computed and sitting in memory. Logging only the rejection index throws away 14 of 15 columns
that cost nothing. Keeping them answers the counterfactual the aggregate metrics cannot:
*would the draft have recovered on its own two slots later?*

★ ``target_top1_p`` IS THE LOAD-BEARING COLUMN. Without it, "rejected" conflates two opposite
findings:
  * target is confident (p>0.9) and the draft still missed  -> the DRAFT is weak. Fixable, and
    it is exactly what more/better training buys.
  * target itself is unsure (p<0.5)                         -> nobody predicts that position.
    Not a draft failure; spending training on it is spending on noise.
Those two demand opposite next moves, so a dump without this column cannot drive a decision.

★ ALIGNMENT IS BY ``out_idx``, NOT BY STEP. Two drafts with different block sizes (released is
block-5 at ns=5, ours is block-15 at ns=15) never share step boundaries, and their accept
lengths differ anyway. But at temperature 0 both produce the SAME output token sequence, so the
running index of emitted tokens per request is a valid join key across runs. That is what
``out_idx`` is, and it is the only reason the two dumps can be compared at all.

SCOPE. Greedy only (``sampling_metadata.all_greedy``). Under sampling there is no single
"target token" to compare against and the accept rule is stochastic; the dumper no-ops with one
warning rather than writing numbers that look comparable and are not. Our evals are temp=0.

ENV
    DSPARK_VERDICT_DUMP=1          enable
    DSPARK_VERDICT_TOPK=64         另外记草稿的全局 top-k token id(0=关,默认关)。
                                   回答:目标要的那个词排在草稿的第几位?rank 2-5 = 草稿知道
                                   但排错序(蒸馏/损失问题);top-k 之外 = 草稿不知道(容量
                                   问题)。⚠ 单路径 dump 只能算【首个断点处】的覆盖率,推不出
                                   accept_len —— oracle 改了 slot s 的词,后面几位是基于草稿
                                   自己那个错词草出来的,得重草。
    DSPARK_VERDICT_DIR=<dir>       output directory (required when enabled)
    DSPARK_VERDICT_PROBS=0         drop the two probability columns (skips 3 small all-reduces)
    DSPARK_VERDICT_TAG=<str>       goes into the filenames; use it to tell the two drafts apart

OUTPUT   ``<dir>/verdict_<tag>_<pid>.bin``(定长记录,流式追加,每个投机步一次 write)
         ``<dir>/reqs_<tag>_<pid>.txt``(一行一个请求 id,行号 = ``req`` 列的取值)
         ``<dir>/topk<K>_<tag>_<pid>.bin``(K×int32/行,行序与主流严格对应;仅 TOPK>0 时)

★★ TOPK 为什么按 req_id 对齐(2026-09-18,实测五轮才定位,别改回按位置)
    五轮实测依次撞到:钩子挂错路(DSpark 不走 compute_draft_token_ids)→ dummy 跑的残留
    (begin_draft_pass)→ 尾部 padding 多一个 block(截断)→ 最后停在
    `行数 20 vs 20,首列不符 5 行`:行数对得上,但正好一整个 block 错位。
    真因:proposer 草稿的那批请求,与采样器验证的那批,集合/顺序不一致 —— 中间隔着调度器,
    它在草稿与验证之间可能丢掉或重排请求,而采样器的 draft_token_ids 是按 input_batch 顺序
    从 scheduler_output.scheduled_spec_decode_tokens【重建】的。
    ⟹ 按位置对齐无论怎么转置都会在某些步上错开。现在两边都用 req_id 说话:proposer 交出
       本批的 req_ids(set_draft_topk_req_ids),capture() 按采样器的验证顺序查表重排。
    ⚠ 「首列必须逐行等于 draft_token_ids」那条校验保留 —— 它同时兜住「proposer 第 b 行 =
       input_batch 第 b 个请求」这个映射假设。
    ★ 不缓冲、不分片。缓冲省的是压缩开销,换来的是「阈值没到整轮不落盘」「SIGKILL
      丢尾巴」两类静默数据丢失 —— 都实际发生过。采集跑不在乎这点速度。
    step            int32   global forward counter (this process)
    req             int32   行号,指向 reqs_*.txt 里的请求 id
    out_idx         int32   ★ index of this slot's token in the request's OUTPUT stream
    slot            int8    position inside the draft block (0 = first drafted token)
    draft_tok       int32   what the draft proposed
    target_tok      int32   what the target would emit (global argmax, already all-gathered)
    accepted        bool    slot <= accepted prefix length
    bonus_tok       int32   the step's bonus token for this request (see below)
    target_top1_p   float16 target's probability on its own argmax   (PROBS=1)
    target_p_draft  float16 target's probability on the draft's token (PROBS=1)

★ WHY ``bonus_tok`` IS HERE. Two runs of the same eval get DIFFERENT request-id strings, so
the only thing that can join them is the OUTPUT TOKEN SEQUENCE (identical at temperature 0).
Rebuilding that sequence from the rows needs the token actually emitted at every position:

    rejected at slot k -> the k accepted drafts, then ``target_tok`` at slot k
    all slots accepted -> every draft, then the BONUS token

The bonus case has no row of its own -- it is emitted past the last drafted slot -- so without
this column the reconstruction silently drops one token exactly on the steps that went best,
and the two runs then fail to align. Repeated per row (same value within a request-step);
830k rows of int32 is ~3 MB, which is not worth a second array to avoid.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from vllm.logger import logger

_COLS_I32 = ("step", "req", "out_idx", "draft_tok", "target_tok", "bonus_tok")

# 流式追加的定长记录。★ 为什么不缓冲:缓冲的唯一好处是摊薄压缩开销,而这是采集跑,慢一点
# 无所谓;代价却是「阈值没到就整轮不落盘」「SIGKILL 丢尾巴」「要 atexit / 要 SIGTERM 处理器」
# —— 这些坑我们一天之内全踩了一遍,而且每一个都是静默的。每步直接 append 之后,没有阈值、
# 没有信号处理、没有分片编号,进程怎么死都只可能丢最后一步。30 字节/行 × 22 万行 = 6.6 MB,
# 不值得为这点体积换回上面那串复杂度。
_REC = np.dtype([
    ("step", "<i4"), ("req", "<i4"), ("out_idx", "<i4"),
    ("draft_tok", "<i4"), ("target_tok", "<i4"), ("bonus_tok", "<i4"),
    ("slot", "i1"), ("accepted", "?"),
    ("target_top1_p", "<f2"), ("target_p_draft", "<f2"),
])


class DsparkVerdictDumper:
    def __init__(self) -> None:
        self.enabled = os.environ.get("DSPARK_VERDICT_DUMP", "0") == "1"
        self.out_dir = os.environ.get("DSPARK_VERDICT_DIR", "")
        self.want_probs = os.environ.get("DSPARK_VERDICT_PROBS", "1") == "1"
        self.tag = os.environ.get("DSPARK_VERDICT_TAG", "run")
        # ⚠ 默认值从 200000 降到 20000。实测教训:gsm8k 全集 released@ns=5 只产生
        # num_draft_tokens=219,640 行,而 DP2 下两个 writer 各写一半 ≈110k —— 【都没到
        # 200000】,于是整轮跑完一个字节都没落盘,行全堆在 worker 内存里,没有任何报错。
        # 阈值必须远小于「一次正常评测的每-writer 行数」,否则它就是个静默丢数据的陷阱。

        self._step = 0
        # ★ top-k 侧流。放【单独的文件】而不是加宽主记录:主格式保持不变(旧分析照跑),
        # top-k 可选、K 可变,而且文件名里带 K 所以自描述。行序与主流严格一一对应。
        # ⚠ 没开 dump 时必须【强制归零】。proposer 里 top-k 那段只看 `_vd.topk > 0`,不看
        # enabled —— 上一轮采集留在 shell 里的 DSPARK_VERDICT_TOPK=64,会让下一轮只想跑
        # 干净评测的 serve 照样每个草稿位做一次全词表 topk(129280 类 × 每 slot × 每步),
        # 算完谁也不写。accept_len 不受影响,但 tok/s 白掉一截,而且不留任何痕迹 ——
        # 正好污染那个用来和基线比吞吐的数。
        self.topk = int(os.environ.get("DSPARK_VERDICT_TOPK", "0")) if self.enabled else 0
        self._topk_iters: list = []      # 本步各次草稿迭代的 [B, K],capture 时拼装
        self._topk_req_ids: list[str] | None = None   # proposer 那批的请求 id(按行)
        self._topk_debug = os.environ.get("DSPARK_VERDICT_TOPK_DEBUG", "0") == "1"
        self._topk_dbg_left = 3
        self._topk_skipped = 0        # 跳过的步数;静默降级必须看得见
        self._topk_dead = False          # 对齐校验失败后置位,只停 top-k,不影响主流
        self._fh_topk = None
        self._buf: list[tuple] = []      # 仅在一次 capture 内累积,出函数即落盘
        self._fh = None                  # 行文件(append,二进制)
        self._fh_req = None              # 请求 id 文件(append,一行一个)
        self._req_ids: list[str] = []          # index -> request id string
        self._req_index: dict[str, int] = {}
        self._out_pos: dict[str, int] = {}     # request id -> next output index
        self._batch_req_ids: list[str] | None = None
        self._warned_nongreedy = False
        self._warned_no_req_ids = False
        self._is_writer = False

        if not self.enabled:
            return
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            self._is_writer = get_tensor_model_parallel_rank() == 0
        except Exception:  # pragma: no cover - single rank / not initialised yet
            self._is_writer = True
        if not self.out_dir:
            raise ValueError("DSPARK_VERDICT_DUMP=1 requires DSPARK_VERDICT_DIR")
        if self._is_writer:
            os.makedirs(self.out_dir, exist_ok=True)
            base = os.path.join(self.out_dir, f"{{}}_{self.tag}_{os.getpid()}")
            self._fh = open(base.format("verdict") + ".bin", "ab", buffering=0)
            self._fh_req = open(base.format("reqs") + ".txt", "a", buffering=1)
            if self.topk > 0:
                self._fh_topk = open(base.format(f"topk{self.topk}") + ".bin", "ab", buffering=0)
            logger.info(
                "DsparkVerdictDumper active: tag=%s probs=%s topk=%d streaming -> %s",
                self.tag, self.want_probs, self.topk, self._fh.name,
            )

    @property
    def active(self) -> bool:
        return self.enabled and self._is_writer

    def begin_draft_pass(self) -> None:
        """一趟草稿开始。★ 必须有这个显式起点。

        proposer 在 profiling / dummy 跑里【也会】草稿,而那些跑不走 capture(),累积的
        top-k 没人消费。没有这个清零,第一次真 capture 看到的是好几趟混在一起的残留 ——
        实测报错 `stack expects each tensor to be equal size, got [64,64] at entry 0 and
        [62,64] at entry 10`(num_spec=5 却攒到 entry 10,且 64/62 是 dummy 的批大小)。
        """
        if self.topk > 0:
            self._topk_iters = []

    def set_draft_topk_req_ids(self, req_ids) -> None:
        """proposer 交出【它这批】的请求 id,按行对应。

        ★★ 这是 top-k 能对上的关键,也是五轮试错的结论。proposer 出完草稿之后,调度器可能
        丢掉或重排请求,而采样器的 draft_token_ids 是按 input_batch 顺序从 scheduler_output
        【重建】的 —— 两边的行顺序没有任何保证。按位置对齐(无论怎么转置)都会在某些步上错开
        整整一个 block(实测 20 行里错 5 行)。两边都用 req_id 说话,顺序就无关了。
        proposer 的第 b 行 = input_batch 的第 b 个请求(token_indices_to_sample 取的是
        query_start_loc[1:]-1,即每个请求的最后一个 token,按 input_batch 顺序)。
        """
        if self.topk > 0:
            self._topk_req_ids = list(req_ids) if req_ids is not None else None

    def push_draft_topk(self, topk: "torch.Tensor") -> None:
        """proposer 每草一位调一次,传 [B, K] 的全局 top-k id。

        ⚠ 这里【只累积不落盘】:proposer 的输出是 slot-major(先给所有请求的 slot0,再 slot1
        …),而拒绝采样器要的是 req-major(req0 的 slot0..K-1,再 req1…)。顺序不同,必须等
        一整块草完、在 capture() 里转置。转错了就是一份行数对得上、内容全错的数据 —— 所以
        capture() 里有逐行自检,见那里。
        """
        if self.topk > 0 and not self._topk_dead:
            self._topk_iters.append(topk.detach().to("cpu", torch.int32))

    def set_batch_req_ids(self, req_ids) -> None:
        """Called by the model runner once per forward, before sampling.

        The rejection sampler does not receive request ids, and batch SLOTS are recycled
        across requests, so without this the per-request output stream cannot be rebuilt --
        rows would be unjoinable to anything. Cheap enough to call unconditionally.
        """
        if self.enabled:
            self._batch_req_ids = list(req_ids) if req_ids is not None else None

    # ------------------------------------------------------------------ capture

    @torch.inference_mode()
    def capture(
        self,
        draft_token_ids: torch.Tensor,     # [num_tokens]
        cu_num_draft_tokens: torch.Tensor, # [batch]
        target_argmax: torch.Tensor,       # [num_tokens]  already GLOBAL (greedy_sample)
        output_token_ids: torch.Tensor,    # [batch, max_spec_len + 1]
        bonus_token_ids: torch.Tensor,     # [batch, 1]
        raw_target_logits: torch.Tensor | None,  # UNPROCESSED logits
        all_greedy: bool,
        logits_sharded: bool,
    ) -> None:
        if not self.active:
            return
        if not all_greedy:
            if not self._warned_nongreedy:
                self._warned_nongreedy = True
                logger.warning("DsparkVerdictDumper: sampling is not all-greedy; not dumping.")
            return

        step = self._step
        self._step += 1

        cu = cu_num_draft_tokens.tolist()
        draft = draft_token_ids.tolist()
        targ = target_argmax.tolist()
        # Accepted prefix length per request. output_token_ids pads rejected slots with -1,
        # so counting non-negative entries and subtracting the always-present bonus token
        # gives the number of ACCEPTED DRAFT tokens. Deriving it here rather than taking it
        # from /metrics keeps the row-level and aggregate views from ever disagreeing.
        out = output_token_ids.tolist()
        n_emit = [sum(1 for t in row if t >= 0) for row in out]
        bonus = [int(x[0]) for x in bonus_token_ids.tolist()]

        probs_top1 = probs_draft = None
        if self.want_probs and raw_target_logits is not None:
            probs_top1, probs_draft = self._global_probs(
                raw_target_logits, draft_token_ids, logits_sharded
            )
            probs_top1 = probs_top1.tolist()
            probs_draft = probs_draft.tolist()

        prev = 0
        rids_in_order: list[str] = []   # 本步【实际产出草稿行】的请求,按采样器切片顺序
        for b, end in enumerate(cu):
            if self._batch_req_ids and b < len(self._batch_req_ids):
                rid = self._batch_req_ids[b]
            else:
                # ⚠ 没有请求 id = 这份 dump 【无法 join】。槽位在请求之间复用,所以 `?slotN`
                # 会把不相干的请求拼成一条流,out_idx 随之失去意义。以前这里静默回退,结果是
                # 采了 20 万行、列齐全、看着完好、实际报废(MRV1 上没挂 set_batch_req_ids)。
                # 宁可吵,也不要再产出一份「看着能用」的废数据。
                if not self._warned_no_req_ids:
                    self._warned_no_req_ids = True
                    logger.error(
                        "DsparkVerdictDumper: 模型运行器没有交接 req_ids —— 本次 dump 的 "
                        "out_idx 【不可用于跨 run join】。该运行器缺少 "
                        "get_verdict_dumper().set_batch_req_ids(...) 调用。"
                    )
                rid = f"?slot{b}"
            ridx = self._req_index.get(rid)
            if ridx is None:
                ridx = len(self._req_ids)
                self._req_index[rid] = ridx
                self._req_ids.append(rid)
                self._note_req(rid)
            base = self._out_pos.get(rid, 0)
            if end > prev:
                rids_in_order.append(rid)   # 0 草稿的请求不进采样器的 draft_token_ids
            n_acc = max(0, n_emit[b] - 1)          # emitted = accepted drafts + 1 bonus
            for slot, i in enumerate(range(prev, end)):
                self._push(
                    step=step, req=ridx, out_idx=base + slot, slot=slot,
                    draft_tok=draft[i], target_tok=targ[i], bonus_tok=bonus[b],
                    accepted=slot < n_acc,
                    top1=probs_top1[i] if probs_top1 else 0.0,
                    pdraft=probs_draft[i] if probs_draft else 0.0,
                )
            self._out_pos[rid] = base + n_emit[b]
            prev = end

        # ★ 每个投机步直接落盘,不攒。buffering=0 意味着 write() 直达内核,进程被 SIGKILL
        # 也只会丢正在写的这一步。
        if self._buf:
            np.asarray(self._buf, dtype=_REC).tofile(self._fh)
            self._buf = []
        self._write_topk(draft_token_ids, rids_in_order)

    def _write_topk(self, draft_token_ids, rids_in_order) -> None:
        """按 req_id 把本步的 top-k 摆成采样器的顺序并落盘,附逐行自检。

        rids_in_order = 采样器这一步【实际验证】的请求 id,按 cu_num_draft_tokens 的切片顺序。
        proposer 那批用 self._topk_req_ids 标识。两者取交集并按采样器的顺序重排。

        ★ 自检不变:摆好之后 top-k 的第 0 列必须逐行等于 draft_token_ids。它同时兜住了
        「行→请求映射假设是否成立」—— 假设错了,重排完必然对不上。
        """
        iters, self._topk_iters = self._topk_iters, []
        p_rids, self._topk_req_ids = self._topk_req_ids, None
        if self.topk <= 0 or self._topk_dead or self._fh_topk is None or not iters:
            return
        try:
            stacked = torch.stack(iters, dim=1)          # [B, n_iter, k]  slot-major -> req-major
            want = draft_token_ids.detach().to("cpu", torch.int32).reshape(-1)
            n_slot = stacked.shape[1]

            if p_rids is None:
                raise RuntimeError("proposer 没有交出 req_ids(set_draft_topk_req_ids 未调用)")
            pos = {r: i for i, r in enumerate(p_rids[: stacked.shape[0]])}
            missing = [r for r in rids_in_order if r not in pos]
            if missing:
                # ⚠ 永久停,不是跳过:一个请求不可能没被草稿过却拿到草稿 token,所以这只能是
                # 「proposer 第 b 行 = input_batch 第 b 个请求」这个映射不成立 —— 结构性问题,
                # 后面每一步都会重演。跳过的话就是一条警告 + 整轮静默无数据,今天已经吃过。
                self._topk_dead = True
                logger.error(
                    "DsparkVerdictDumper: 采样器这步验证的 %d 个请求里有 %d 个不在 proposer "
                    "那批(%d 个)里,例 %s —— 行->请求的映射不成立,已永久停止写 top-k,"
                    "主流不受影响。开 DSPARK_VERDICT_TOPK_DEBUG=1 打印两边的 rids。",
                    len(rids_in_order), len(missing), len(pos), missing[:3],
                )
                if self._topk_debug:
                    logger.error("[TOPK-DEBUG] proposer rids=%s\n             采样器 rids=%s",
                                 p_rids[:12], list(rids_in_order)[:12])
                return
            sel = torch.tensor([pos[r] for r in rids_in_order], dtype=torch.long)
            flat = stacked[sel].reshape(-1, stacked.shape[-1])

            if flat.shape[0] != want.numel() or not torch.equal(flat[:, 0], want):
                bad = int((flat[:, 0] != want).sum()) if flat.shape[0] == want.numel() else -1
                if self._topk_debug and self._topk_dbg_left > 0:
                    self._topk_dbg_left -= 1
                    logger.error(
                        "[TOPK-DEBUG] proposer B=%d slots=%d rids=%s\n"
                        "             采样器 rids=%s\n"
                        "             重排后首列=%s\n"
                        "             draft_token=%s",
                        stacked.shape[0], n_slot, p_rids[:6], list(rids_in_order)[:6],
                        flat[:12, 0].tolist(), want[:12].tolist(),
                    )
                self._topk_dead = True
                logger.error(
                    "DsparkVerdictDumper: 按 req_id 重排后 top-k 仍与 draft_token_ids 不符"
                    "(行数 %d vs %d,首列不符 %s 行)—— 已停止写 top-k,主流不受影响。"
                    "说明「proposer 第 b 行 = input_batch 第 b 个请求」这个映射不成立。"
                    "开 DSPARK_VERDICT_TOPK_DEBUG=1 可打印两边的 rids 与首列。",
                    flat.shape[0], want.numel(), bad,
                )
                return
            flat.numpy().astype(np.int32).tofile(self._fh_topk)
        except Exception as e:  # noqa: BLE001
            self._topk_skipped += 1
            # ⚠ 不只报一次:只报一次 = 整轮静默少数据而没人知道。按指数间隔持续报,
            # 既不刷屏,又让「跳了很多步」这件事一定会浮出来。
            if self._topk_skipped in (1, 10, 100, 1000, 10000):
                logger.warning("DsparkVerdictDumper: top-k 已跳过 %d 步(最近一次:%r)",
                               self._topk_skipped, e)

    def _push(self, *, step, req, out_idx, slot, draft_tok, target_tok, bonus_tok,
              accepted, top1, pdraft):
        # 顺序必须与 _REC 的字段顺序一致 —— 元组是按位置填进结构化数组的。
        self._buf.append((step, req, out_idx, draft_tok, target_tok, bonus_tok,
                          slot, accepted, top1, pdraft))

    def _note_req(self, rid: str) -> None:
        """新请求 id 追加一行。行号 = req 索引,所以【只能追加、不能重排】。"""
        if self._fh_req is not None:
            self._fh_req.write(rid.replace("\n", " ") + "\n")

    # ------------------------------------------------------------------ probabilities

    @staticmethod
    def _global_probs(logits: torch.Tensor, draft_token_ids: torch.Tensor, sharded: bool):
        """Target probability on its own argmax, and on the draft's token — GLOBAL under TP.

        ⚠️ ``sharded`` is PASSED IN, never guessed. ``rejection_sample`` has two greedy
        branches -- ``enable_reduce_sample`` gathers (so its input IS vocab-sharded), the other
        takes a plain ``argmax`` (so its input is already full-vocab). Inferring which one ran
        from the tensor shape would need the vocab size, which this module has no business
        knowing; getting it wrong silently produces per-rank "probabilities" that are not
        probabilities. The caller knows, so the caller says.

        ⚠️ When sharded, a local softmax has the wrong denominator, so a
        per-rank probability is not a probability at all. The global log-sum-exp needs two
        all-reduces (max, then the shifted sum) and the draft token's logit needs a third
        because it lives on whichever rank owns its slice. Three [num_tokens] all-reduces per
        forward — negligible bytes, but three extra sync points, which is why PROBS=0 exists.

        The shard layout (``global_id = rank * V_local + local_id``) is not assumed: it is the
        same arithmetic ``greedy_sample`` uses to rebuild global ids, so the two agree by
        construction.
        """
        tp, world = None, 1
        if sharded:
            try:
                from vllm.distributed import get_tp_group

                tp = get_tp_group()
                world = tp.world_size
            except Exception:  # pragma: no cover
                world = 1

        lg = logits.float()
        if world == 1:
            lse = lg.logsumexp(dim=-1)
            top1 = (lg.max(dim=-1).values - lse).exp()
            pdr = (lg.gather(1, draft_token_ids.view(-1, 1).long()).squeeze(1) - lse).exp()
            return top1.half().cpu(), pdr.half().cpu()

        import torch.distributed as dist

        grp = tp.device_group
        v_local, rank = lg.shape[1], tp.rank_in_group

        gmax = lg.max(dim=-1).values.contiguous()
        dist.all_reduce(gmax, op=dist.ReduceOp.MAX, group=grp)
        sumexp = (lg - gmax.unsqueeze(1)).exp().sum(dim=-1).contiguous()
        dist.all_reduce(sumexp, op=dist.ReduceOp.SUM, group=grp)
        lse = gmax + sumexp.log()

        d = draft_token_ids.long()
        owner = (d // v_local) == rank
        local_idx = (d % v_local).clamp_(0, v_local - 1).view(-1, 1)
        dlogit = lg.gather(1, local_idx).squeeze(1)
        dlogit = torch.where(owner, dlogit, torch.full_like(dlogit, float("-inf"))).contiguous()
        dist.all_reduce(dlogit, op=dist.ReduceOp.MAX, group=grp)

        return (gmax - lse).exp().half().cpu(), (dlogit - lse).exp().half().cpu()

    # ------------------------------------------------------------------ output

_DUMPER: DsparkVerdictDumper | None = None


def get_verdict_dumper() -> DsparkVerdictDumper:
    global _DUMPER
    if _DUMPER is None:
        _DUMPER = DsparkVerdictDumper()
    return _DUMPER

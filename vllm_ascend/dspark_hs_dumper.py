#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Plan B — DSpark hidden-state dumper for DSV4 (memory-light).

Rides the validated dspark serve. The DSV4 model forward already captures the aux
target layers (``config.dspark_target_layer_ids``, e.g. [40, 41, 42], mean over the
mHC ``hc_mult`` streams, POST-layer — matching the DeepSpec canonical convention)
into the ~200 MB per-forward scratch buffer exposed by
``get_mtp_target_hidden_states()``. This dumper copies that buffer plus the model's
post-norm final hidden state (the verifier-last) out to CPU per request and writes
speculators-format ``hs_<id>.safetensors`` files.

It deliberately avoids the standard ``extract_hidden_states`` path: that path stores
the hidden states in a block-based, context-sized, TP-replicated ``HiddenStateCacheSpec``
KV cache which on DSV4 (hyper-compressed MLA+DSA KV) dwarfs the real KV budget and OOMs.
Plan B holds only the current forward's tokens (the fixed scratch buffer, overwritten
each step) and moves the "hold until request-end" accumulation to abundant CPU RAM.

See ``docs/deployment/ascend-npu-dsv4-hs-dumper-planB.md`` in the speculators repo.

Enable via serve env:
    DSPARK_HS_DUMP=1                 # opt-in
    DSPARK_HS_DIR=/path/hidden_states   # output dir (required when enabled)
The DSV4 target config must carry ``dspark_target_layer_ids`` (else the buffer is never
allocated and the getter returns None — the dumper then no-ops). Writes come only from
TP rank 0 (the residual-stream hidden is TP-replicated); each DP replica's rank 0 writes
its own disjoint set of requests.
"""

import os

import torch
from safetensors.torch import save_file
from vllm.logger import init_logger

logger = init_logger(__name__)


class DsparkHSDumper:
    """Per-request CPU accumulator + speculators-format writer for DSpark HS.

    On-disk layout of each ``hs_<id>.safetensors`` (the standardized v1 format that
    ``speculators.train.data.ArrowDataset`` loads directly, no ``standardize_data_v1``
    needed):

        hidden_states               : [seq, num_target_layers * hidden_size]   (aux buffer)
        verifier_last_hidden_states : [seq, hidden_size]                        (post-norm final)
        input_ids                   : [seq]                                     (long)
        loss_mask                   : [seq]                                     (bool)
    """

    def __init__(self) -> None:
        self.enabled = os.environ.get("DSPARK_HS_DUMP", "0") == "1"
        self.out_dir = os.environ.get("DSPARK_HS_DIR", "")
        # req_id -> {"aux": [chunk, ...], "last": [...], "ids": [...]}
        self._acc: dict[str, dict[str, list[torch.Tensor]]] = {}
        self._is_writer = False
        self._written = 0

        if not self.enabled:
            return
        # Only TP rank 0 writes (the hidden state is TP-replicated; rank 0 holds the
        # full tensor). Import lazily so a disabled dumper never touches distributed.
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            self._is_writer = get_tensor_model_parallel_rank() == 0
        except Exception:  # pragma: no cover - single-rank / not yet initialized
            self._is_writer = True
        if self._is_writer:
            if not self.out_dir:
                raise ValueError(
                    "DSPARK_HS_DUMP=1 requires DSPARK_HS_DIR to point at the "
                    "hidden_states output directory."
                )
            os.makedirs(self.out_dir, exist_ok=True)
            logger.info("DsparkHSDumper active: writing hs_<id>.safetensors to %s", self.out_dir)

    @property
    def active(self) -> bool:
        return self.enabled and self._is_writer

    @torch.inference_mode()
    def capture(
        self,
        *,
        scheduler_output,
        input_batch,
        hidden_states: torch.Tensor,
        aux_buffer: torch.Tensor | None,
        input_ids: torch.Tensor,
        num_tokens: int,
    ) -> None:
        """Copy this forward's per-token aux + final hidden to CPU, routed per request.

        ``hidden_states``  : [>=num_tokens, H]         post-norm final hidden (verifier-last)
        ``aux_buffer``     : [>=num_tokens, L_aux*H]    the dspark scratch buffer (may be None)
        ``input_ids``      : [>=num_tokens]             token ids for this forward
        Must be called BEFORE the next forward overwrites ``aux_buffer``; the NPU->CPU
        copies below are synchronous, so the data is safe once ``capture`` returns.
        """
        if not self.active or aux_buffer is None:
            return

        aux = aux_buffer[:num_tokens]
        last = hidden_states[:num_tokens]
        ids = input_ids[:num_tokens]
        req_ids = input_batch.req_ids

        offset = 0
        for i, req_id in enumerate(req_ids):
            n = int(scheduler_output.num_scheduled_tokens[req_id])
            if n <= 0:
                continue
            sl = slice(offset, offset + n)
            offset += n

            ent = self._acc.setdefault(req_id, {"aux": [], "last": [], "ids": []})
            # Synchronous device->host copies (aux_buffer is overwritten next forward).
            ent["aux"].append(aux[sl].to("cpu"))
            ent["last"].append(last[sl].to("cpu"))
            ent["ids"].append(ids[sl].to("cpu"))

            # Flush when we have accumulated the request's whole prompt. Teacher-forced
            # extraction is prefill-only (max_tokens=1), so the full (prompt+response)
            # sequence is prefilled in one or more chunks and completes here — no decode
            # tokens follow. Timing-independent: keyed on accumulated length, not on
            # num_computed_tokens.
            total = int(input_batch.num_prompt_tokens[i])
            acc_len = sum(t.shape[0] for t in ent["ids"])
            if total > 0 and acc_len >= total:
                self._flush(req_id)

    def _flush(self, req_id) -> None:
        ent = self._acc.pop(req_id, None)
        if ent is None:
            return
        aux = torch.cat(ent["aux"], dim=0).contiguous()   # [seq, L_aux*H]
        last = torch.cat(ent["last"], dim=0).contiguous()  # [seq, H]
        ids = torch.cat(ent["ids"], dim=0).to(torch.long).contiguous()  # [seq]
        seq = int(ids.shape[0])

        data = {
            "hidden_states": aux,
            "verifier_last_hidden_states": last,
            "input_ids": ids,
            # P1: train on all positions. The prompt/response split (loss_mask=0 on the
            # prompt) is an open item (planB doc §8.4) — the driving client knows each
            # rollout row's split and can refine this. All-ones is a safe placeholder.
            "loss_mask": torch.ones(seq, dtype=torch.bool),
        }

        stem = self._stem(req_id)
        final = os.path.join(self.out_dir, f"{stem}.safetensors")
        tmp = final + ".tmp"
        # Write to a temp file then rename so a rolling-buffer reader never sees a
        # partial file (the online trainer streams these with on_missing="generate").
        save_file(data, tmp)
        os.replace(tmp, final)
        self._written += 1
        if self._written <= 3 or self._written % 200 == 0:
            logger.info(
                "DsparkHSDumper wrote %s (seq=%d, hidden_states=%s, verifier_last=%s) [#%d]",
                os.path.basename(final), seq, tuple(aux.shape), tuple(last.shape), self._written,
            )

    @staticmethod
    def _stem(req_id) -> str:
        """File stem ``hs_<rollout_row_index>``.

        The extraction client sets the request id to the rollout row index so the
        trainer (which reads ``hs_{index}.safetensors``) picks files up automatically.
        If the id already looks like ``hs_...`` use it verbatim; otherwise prefix.
        """
        s = str(req_id)
        return s if s.startswith("hs_") else f"hs_{s}"

#!/usr/bin/env python3
"""
生成一个简化版的 MoE（Mixture-of-Experts）执行轨迹，格式符合 Chakra Execution Trace。

使用方式（在 WSL 或 Linux 环境）：
  source /mnt/c/astra-sim/.venv/bin/activate
  python utils/generate_moe_trace.py \
      --num-experts 8 \
      --tokens-per-batch 4096 \
      --topk 2 \
      --hidden-dim 4096 \
      --expert-ffn-dim 16384 \
      --output-dir examples/workload/moe/minimal_moe_8e_top2

这会在目标目录下生成 8 个 .et 文件，可直接作为 ASTRA-sim 的 workload 输入。
"""

import argparse
import os
from pathlib import Path
from typing import List

from extern.graph_frontend.chakra.schema.protobuf import et_def_pb2 as et
from extern.graph_frontend.chakra.src.third_party.utils.protolib import (
    encodeMessage as encode_message,
)

# Chakra 节点 ID 需要保持全局自增
NODE_ID = 0


def _next_node(name: str, node_type: et.NodeType) -> et.Node:
    """创建带唯一 ID 的节点。"""
    global NODE_ID  # pylint: disable=global-statement
    node = et.Node()
    node.id = NODE_ID
    node.name = name
    node.type = node_type
    NODE_ID += 1
    return node


def _comm_attr(comm_type: int, comm_size: int) -> List[et.AttributeProto]:
    """构造通信节点的属性列表。"""
    return [
        et.AttributeProto(name="is_cpu_op", bool_val=False),
        et.AttributeProto(name="comm_type", int64_val=comm_type),
        et.AttributeProto(name="comm_size", int64_val=comm_size),
    ]


def _write_trace(
    out_path: Path,
    num_experts: int,
    tokens_per_batch: int,
    topk: int,
    hidden_dim: int,
    expert_ffn_dim: int,
) -> None:
    """为单个 NPU（视作一个专家）生成 MoE 流水的核心节点。"""
    # 估算通信与计算量（粗略比例，便于调参与仿真）
    tokens_per_expert = tokens_per_batch * topk // num_experts
    expert_param_bytes = 2 * hidden_dim * expert_ffn_dim  # 简化：两层 FFN
    activation_bytes = tokens_per_expert * hidden_dim * 2

    with out_path.open("wb") as et_file:
        # 写入元数据
        encode_message(et_file, et.GlobalMetadata(version="0.0.4"))

        # 1) 路由 (compute)
        routing = _next_node("router_forward", et.COMP_NODE)
        routing.attr.append(et.AttributeProto(name="is_cpu_op", bool_val=False))
        routing.duration_micros = max(tokens_per_batch // 128, 10)
        encode_message(et_file, routing)

        # 2) tokens -> experts 的 All-to-All
        dispatch = _next_node("token_dispatch", et.COMM_COLL_NODE)
        dispatch.attr.extend(_comm_attr(et.ALL_TO_ALL, activation_bytes))
        dispatch.data_deps.append(routing.id)
        encode_message(et_file, dispatch)

        # 3) 专家前向 (compute)
        expert = _next_node("expert_ffn_forward", et.COMP_NODE)
        expert.attr.append(et.AttributeProto(name="is_cpu_op", bool_val=False))
        expert.duration_micros = max(expert_param_bytes // (32 * 1024), 50)
        expert.data_deps.append(dispatch.id)
        encode_message(et_file, expert)

        # 4) experts -> tokens 的 All-to-All
        combine = _next_node("token_gather", et.COMM_COLL_NODE)
        combine.attr.extend(_comm_attr(et.ALL_TO_ALL, activation_bytes))
        combine.data_deps.append(expert.id)
        encode_message(et_file, combine)

        # 5) 残差+输出投影 (compute)
        output_proj = _next_node("output_projection", et.COMP_NODE)
        output_proj.attr.append(et.AttributeProto(name="is_cpu_op", bool_val=False))
        output_proj.duration_micros = max(hidden_dim * hidden_dim // (32 * 1024), 40)
        output_proj.data_deps.append(combine.id)
        encode_message(et_file, output_proj)

        # 6) 辅助损失 AllReduce
        aux_loss = _next_node("auxiliary_loss_allreduce", et.COMM_COLL_NODE)
        aux_loss.attr.extend(_comm_attr(et.ALL_REDUCE, 4 * num_experts))
        aux_loss.data_deps.append(output_proj.id)
        encode_message(et_file, aux_loss)


def generate_moe_trace(
    num_experts: int,
    tokens_per_batch: int,
    topk: int,
    hidden_dim: int,
    expert_ffn_dim: int,
    output_dir: Path,
) -> None:
    """为多个专家生成 .et 文件，分别命名为 expert_rankX.et。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for expert_rank in range(num_experts):
        filename = output_dir / f"expert_rank{expert_rank}.et"
        _write_trace(
            filename,
            num_experts=num_experts,
            tokens_per_batch=tokens_per_batch,
            topk=topk,
            hidden_dim=hidden_dim,
            expert_ffn_dim=expert_ffn_dim,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 MoE 执行轨迹 (Chakra ET)")
    parser.add_argument("--num-experts", type=int, default=8, help="专家数量 / 并行设备数")
    parser.add_argument("--tokens-per-batch", type=int, default=4096, help="每批 token 数")
    parser.add_argument("--topk", type=int, default=2, help="每个 token 选择的专家数 (Top-K)")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="模型隐藏维度")
    parser.add_argument("--expert-ffn-dim", type=int, default=16384, help="专家 FFN 的扩展维度")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/workload/moe/minimal_moe_8e_top2"),
        help="输出目录，相对于仓库根目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    absolute_output = (repo_root / args.output_dir).resolve()
    generate_moe_trace(
        num_experts=args.num_experts,
        tokens_per_batch=args.tokens_per_batch,
        topk=args.topk,
        hidden_dim=args.hidden_dim,
        expert_ffn_dim=args.expert_ffn_dim,
        output_dir=absolute_output,
    )
    print(f"[MoE Trace] 已生成 {args.num_experts} 个 .et 文件，保存路径：{absolute_output}")


if __name__ == "__main__":
    main()

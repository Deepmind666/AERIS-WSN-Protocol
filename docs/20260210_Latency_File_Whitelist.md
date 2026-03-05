# 延迟实验文件白名单/黑名单

**创建日期**: 2026-02-10
**创建者**: Claude 4.6 (任务 T3)
**目的**: 防止论文误引旧版无效延迟文件

---

## 一、可引用文件（白名单）

### 1.1 v2 修复版实验数据（论文可用）

| 文件名 | 大小 | avg_hops_to_bs 范围 | git_commit | 状态 |
|--------|------|---------------------|------------|------|
| `latency_indoor_office_fix_20260209_074608.json` | 51K | 1.197-1.243 | c51f8679 | 有效 |
| `latency_indoor_factory_fix_20260209_074608.json` | 51K | 1.173-1.225 | c51f8679 | 有效 |
| `latency_outdoor_urban_fix_20260209_074608.json` | 51K | 1.123-1.165 | c51f8679 | 有效 |
| `latency_outdoor_suburban_fix_20260209_074608.json` | 52K | 1.200-1.237 | c51f8679 | 有效 |

### 1.2 v2 统计汇总文件（论文可用）

| 文件名 | 说明 |
|--------|------|
| `latency_hop_v2_stats.csv` | 4环境×5协议 hop 统计 (n=30) |
| `latency_hop_v2_significance.csv` | Welch + Hedges g + Holm 显著性检验 |
| `latency_hop_v2_stats.md` | 可读格式统计报告 |

---

## 二、禁止引用文件（黑名单）

### 2.1 含全 0 值的无效文件

| 文件名 | 问题 | 处置 |
|--------|------|------|
| `latency_smoke_test.json` | avg_hops_to_bs 含 0.0 | 禁引用 |
| `latency_indoor_office_smoke_fix_20260209.json` | 10 条 avg_hops_to_bs=0.0 | 禁引用 |

### 2.2 旧版本文件（已被 _fix_ 版本替代）

| 文件名 | 时间戳 | 问题 | 处置 |
|--------|--------|------|------|
| `latency_indoor_office_20260208_234902.json` | 20260208 | 含异常值 51.0 | 禁引用 |
| `latency_indoor_factory_20260208_234929.json` | 20260208 | 旧版本 | 禁引用 |
| `latency_outdoor_urban_20260208_234952.json` | 20260208 | 旧版本 | 禁引用 |
| `latency_outdoor_suburban_20260209_071707.json` | 20260209 | 旧版本 | 禁引用 |

### 2.3 中间版本文件（已被 _fix_ 版本替代）

| 文件名 | 说明 | 处置 |
|--------|------|------|
| `latency_indoor_office_20260209_132945.json` | 被 _fix_ 替代 | 禁引用 |
| `latency_indoor_factory_20260209_133051.json` | 被 _fix_ 替代 | 禁引用 |
| `latency_outdoor_urban_20260209_133155.json` | 被 _fix_ 替代 | 禁引用 |
| `latency_outdoor_suburban_20260209_133257.json` | 被 _fix_ 替代 | 禁引用 |

### 2.4 旧版统计文件

| 文件名 | 说明 | 处置 |
|--------|------|------|
| `latency_hop_stats.csv` | 旧版统计 | 禁引用 |
| `latency_hop_significance.csv` | 旧版统计 | 禁引用 |
| `latency_hop_stats.md` | 旧版统计 | 禁引用 |

---

## 三、仅供诊断文件（不可用于论文）

| 文件名 | 说明 |
|--------|------|
| `latency_indoor_office_smoke_fix2_20260209.json` | smoke test，样本不足 |
| `latency_smoke_test2.json` | smoke test，样本不足 |
| `latency_smoke_resource_guard_20260209.json` | 资源门控测试 |
| `latency_hop_fix_20260209_074608_stats.csv` | _fix_ 版中间统计（与 v2 重复） |
| `latency_hop_fix_20260209_074608_significance.csv` | _fix_ 版中间统计（与 v2 重复） |
| `latency_hop_fix_20260209_074608_stats.md` | _fix_ 版中间统计（与 v2 重复） |

---

## 四、Section 6.5 引用路径修正建议

当前 Section 6.5 引用的数据源：
```
- latency_indoor_office_20260209_132945.json      ← 黑名单（中间版本）
- latency_indoor_factory_20260209_133051.json      ← 黑名单（中间版本）
- latency_outdoor_urban_20260209_133155.json       ← 黑名单（中间版本）
- latency_outdoor_suburban_20260209_133257.json    ← 黑名单（中间版本）
```

应改为：
```
- latency_indoor_office_fix_20260209_074608.json   ← 白名单
- latency_indoor_factory_fix_20260209_074608.json  ← 白名单
- latency_outdoor_urban_fix_20260209_074608.json   ← 白名单
- latency_outdoor_suburban_fix_20260209_074608.json ← 白名单
```

**注意**：数值内容一致（均为 hop-based, n=30），但引用路径必须指向 _fix_ 版本以确保证据链完整。

---

## 五、审计追溯

| 项目 | 值 |
|------|-----|
| 排查方法 | Glob + JSON 逐文件检查 avg_hops_to_bs 字段 |
| 总文件数 | 26 个 latency 相关文件 |
| 白名单 | 7 个（4 数据 + 3 统计） |
| 黑名单 | 13 个 |
| 诊断类 | 6 个 |

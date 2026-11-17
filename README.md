# 大模型（千问 3.5）做跨接收机射频指纹识别：实验代码

用 Qwen3.5 识别 WiSig ManyRx 数据集里 10 台 WiFi 发射机，重点测"训练时没见过的接收机"（跨接收机）。
数据处理、4 折划分、早停规则和评测集全部沿用 IB-RFF（Wan26b，IEEE TIFS 2026）作者的代码：

- 非均衡版数据，每个（接收机，天）每台发射机取前 100 个包，每包 256 点 IQ，逐包功率归一化；
- 每折 24 台接收机第 1、2 天训练（30 % 做验证），另外 8 台只用来测试；
- 测试集：seen_same（训练接收机、同天）、seen_d34（训练接收机、第 3、4 天）、CR（没见过的接收机、第 1、2 天）、CRD（没见过的接收机、第 3、4 天）。

## 文件

| 文件 | 内容 |
|---|---|
| `qwen_run.py` | 方法一：去掉 lm head。信号按 16 点一段、步长 8 投影成 31 个向量 → 千问解码器 → 平均 → 10 类线性分类头。可选预训练 / 随机初始化；4B 用 LoRA |
| `qwen_lm_v2.py` | 方法二：保留 lm head，模型生成一个数字 0–9 作为答案。先 SFT，再 GRPO。输入方式：`time`（时域切片）、`freq`（频域切片）、`text`（IQ 写成整数文字）、`ref`（提示词里带 10 台发射机各一个参考包） |
| `fan_freq_run.py` | 对照：作者的 FAN 代码不改，只把输入换成频域（与 `qwen_lm_v2.py --input freq` 同一变换） |
| `make_chains_fanfreq.py` | 生成 FAN 频域对照的 4 折任务 |
| `diag_fan_errors.py` | 评测工具（四个测试集、逐接收机统计），前两个脚本都会调用，要和它们放在同一目录 |
| `run_lm_chain_v2.sh` | 方法二第 0 折任务链：冒烟 → 时域 / 频域 SFT 和 GRPO → 文本、参考样例 SFT |
| `run_freq4_chain.sh` | FAN 频域对照 4 折 → 方法二频域第 1–3 折 |
| `setup_qwen.sh` | 建环境、从 ModelScope 下载 Qwen3.5-0.8B / 4B |
| `run_chain.py`、`pool_dyn.py` | 按顺序 / 并行跑任务的小工具 |

## 结果

已跑完的结果（json、每个测试样本的概率、日志）和汇总表在 [`results/`](results/README.md)。

## 不在仓库里的东西

- IB-RFF 作者代码（`main.py`、`utils/` 等），运行时用 `--code` 指定目录；
- WiSig ManyRx 数据；
- 千问模型权重（见 `setup_qwen.sh`）和训练好的权重（太大）。

## 用法

```bash
# 方法二：频域输入，第 0 折，先 SFT 再 GRPO
python qwen_lm_v2.py --code IB-RFF目录 --qwen Qwen3.5-0.8B目录 --workdir 输出目录 --input freq --round 0 --seed 2023
python qwen_lm_v2.py --code IB-RFF目录 --qwen Qwen3.5-0.8B目录 --workdir 输出目录 --input freq --round 0 --seed 2023 \
    --stage grpo --init_from 输出目录/weights/lm_freq_sft_r0_s2023.pt

# 方法一：分类头版本
python qwen_run.py --code IB-RFF目录 --qwen Qwen3.5-0.8B目录 --workdir 输出目录 --init pretrained --round 0 --seed 2023
```

`.sh` 脚本里的路径是实验室服务器上的绝对路径，换机器要改开头的 `R=`。

## 环境

RTX 4090（24 GB）；Python 3.10，torch 2.6.0，transformers 5.20.2，flash-linear-attention 0.4.2，scikit-learn。

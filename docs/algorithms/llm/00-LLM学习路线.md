# LLM主要内容



## 大模型完整流程

 - 0. 数据准备 
 - 1. 训练tokenizer
 - 2. 自监督预训练
 - 3. 监督微调（SFT）
 - 4. 偏好对齐（Alignment/RLHF） 强化学习DPO / GRPO
 - 5. 专项能力强化(可选，比如Reasoning, Coding, Tool Calling)
 - 6. 蒸馏&量化
 - 7. 部署推理



## training
分布式并行训练
- [picotron](https://github.com/huggingface/picotron)

picotron是一个由 Hugging Face 团队创建的轻量级、用于教育目的的分布式大模型训练框架，帮助理解大规模语言模型分布式训练的底层技术。项目代码追求极简和可读性，重点在于 “4D并行” 技术
完整的4D并行：它完整实现了现代大模型训练中的四种关键并行策略：数据并行，张量并行，流水线并行，上下文并行

- [nanotron](https://github.com/huggingface/nanotron)

nanotron是一个由 Hugging Face 主导开发的、面向真实大规模（百亿/千亿级）模型训练的分布式训练框架，目标不是教学 Demo，而是作为工业级大模型训练系统的参考实现。
相比 picotron 的“极简教学”，nanotron 更强调 工程完整性、可扩展性与可复现性。     

## inference
prefill + decoding
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)

nano-vLLM 是对 vLLM 推理引擎的极简化实现版本，目标是教学和理解大模型高性能推理的底层机制，而不是直接用于生产环境。它专注于解释 vLLM 为什么快、快在哪里，强调代码可读性与核心思想的清晰呈现。


- [mini-sglang](https://github.com/sgl-project/mini-sglang)

mini-sglang 是对 SGLang 推理系统的极简教学版实现，由 SGLang 项目团队维护，目标是帮助理解 “LLM 推理系统如何把模型执行、请求调度和语言级控制结合起来”。
相比 nano-vLLM 更偏底层算子与内存管理，mini-sglang 更关注 推理系统层的编排与控制逻辑。


## RL强化学习

协调训练和推理

- [unsloth](https://github.com/unslothai/unsloth)


Unsloth 是一个开源的高效 LLM 微调与强化学习（RL）库，其目标是显著降低显存占用、提高训练速度，并支持多种模型与训练模式（如全精度/量化/强化学习）。
主要用途：大模型微调（Fine-tuning）；强化学习后训练（如 GRPO、GSPO、DAPO、PPO、Reward-based training 等）；支持多种模型与任务：LLM、TTS（文本-语音）、Vision、Embedding 等。


- [trl](https://github.com/huggingface/trl)

这是由 Hugging Face 官方维护的一个用于 Transformer 语言模型 后训练（post-training） 的库，核心围绕强化学习与偏好优化 + 传统微调方法。它在 🤗 Transformers 生态之上构建，提供了一组面向训练策略的 Trainer 类和工具，让你可以用 RL-style 或带奖励信号的方式去精调/后训练大语言模型。

## 重点

- [nanochat](https://github.com/karpathy/nanochat)

nanochat 是一个极简的大模型聊天（Chat）示例项目，目标不是性能或功能完备，而是用尽可能少的代码，展示一个 LLM Chat 应用从推理到对话状态管理的最小闭环。
它更像是一个 “可读性极强的 Chat 系统参考实现”。


- [minimind](https://github.com/jingyaogong/minimind)
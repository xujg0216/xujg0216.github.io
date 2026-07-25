
# ollama 使用

## 1. ollama常见命令

```bash
ollama serve         #启动ollama
ollama create        #从模型文件创建模型
ollama show          #显示模型信息
ollama run           #运行模型
ollama pull          #从注册表中拉取模型
ollama push          #将模型推送到注册表
ollama list          #列出模型
ollama cp            #复制模型
ollama rm            #删除模型
ollama help          #获取有关任何命令的帮助信息

```

## 2. Modelfile

**Modelfile** 是用于 **定义、配置和构建模型（Model）的一种脚本文件**，类似于 Dockerfile 对镜像的作用。并不是模型文件本身。

它包含模型的：

* 基础模型来源（如 Llama3、Qwen、Mistral 等）
* 系统提示词（system prompt）
* 量化方式（如 Q4_0、Q8_0）
* 参数配置（上下文长度、温度等）
* 自定义模板（Prompt 模型风格）
* 可选插件或 API 设定

### 2.1 **Modelfile 示例**

```dockerfile
FROM qwen2.5:7b-instruct-q4_0

PARAMETER temperature 0.7
PARAMETER num_ctx 4096

SYSTEM """
你是一个专业的嵌入式视觉工程师助手，会用中文回答问题。
"""

TEMPLATE """
用户问：{{ .Prompt }}
你需要给出专业且简洁的回答。
"""
```

保存为 `Modelfile` 后执行：

```bash
ollama create my-vision-assistant -f Modelfile
```

以后就可以直接：

```bash
ollama run my-vision-assistant
```

---

### 2.2 Modelfile 的常见指令

| 指令                                | 作用                |
| --------------------------------- | ----------------- |
| `FROM <model>`                    | 指定基础模型            |
| `PARAMETER <key> <value>`         | 设置模型参数（温度、上下文长度等） |
| `SYSTEM """..."""`                | 固定的系统提示词          |
| `TEMPLATE """..."""`              | 自定义 Prompt 格式     |
| `LICENSE """..."""`               | 声明授权信息（可选）        |
| `MESSAGE role="user"/"assistant"` | 预设对话内容            |
| `ADAPTER`                         | 加载 LoRA 等微调模型     |
| `DEFINE`                          | 变量定义（较少使用）        |


当然可以！在 **Ollama 的 Modelfile** 中，`PARAMETER` 用来设置 **模型推理参数**，也就是你调用模型时会生效的 **推理行为控制**。这些参数类似于 HuggingFace / OpenAI API 中的 `temperature`、`top_p` 等。

我给你整理一下 Ollama 常用的 `PARAMETER` 选项、含义以及取值范围：

---

**temperature**

```dockerfile
PARAMETER temperature 0.7
```

* **作用**：控制模型生成文本的随机性
* **取值**：0 ~ 2（通常 0.0 ~ 1.0 最常用）
* **含义**：

  * 0 → 完全确定性（每次输出一样）
  * 越大 → 输出越随机、多样化
* **场景**：

  * 回答精确问题 → 0~0.3
  * 创意生成 / 文学文本 → 0.7~1.0

---

**num_ctx**

```dockerfile
PARAMETER num_ctx 4096
```

* **作用**：模型上下文窗口大小（token 数量）
* **单位**：token
* **含义**：模型一次可以看到的最大文本长度
* **场景**：

  * 一般默认即可
  * 对于长对话或文档生成，需要增大
* **注意**：过大可能占用更多显存

---

**top_p**

```dockerfile
PARAMETER top_p 0.9
```

* **作用**：核采样（nucleus sampling）
* **取值**：0 ~ 1
* **含义**：

  * 模型只在概率总和达到 `top_p` 的 token 中采样
  * 越小 → 输出越保守
  * 越大 → 输出更随机
* **对应 OpenAI**：`top_p` 参数

---

**top_k**

```dockerfile
PARAMETER top_k 50
```

* **作用**：限制采样候选 token 数量
* **含义**：只从概率前 K 个 token 中采样
* **默认**：通常 50~100

---

**repetition_penalty**

```dockerfile
PARAMETER repetition_penalty 1.2
```

* **作用**：抑制重复 token
* **取值**：>1 → 惩罚重复，=1 → 无惩罚
* **场景**：生成长文本时防止重复

---

**seed**

```dockerfile
PARAMETER seed 42
```

* **作用**：随机种子，用于生成固定结果
* **场景**：可复现输出

---

**temperature / top_p / top_k** 的组合使用

* **temperature** 控制随机性
* **top_p** 限制概率质量
* **top_k** 限制候选数量
* **常见组合**：

  ```dockerfile
  PARAMETER temperature 0.7
  PARAMETER top_p 0.9
  PARAMETER top_k 50
  ```



### 2.3 和 GGUF、Safetensors 的关系

| 名称                                    | 说明                              | 用途                                |
| ------------------------------------- | ------------------------------- | --------------------------------- |
| **GGUF**                              | 一种模型格式（适合 CPU/GPU 推理，quantized） | huggingface 上下载或 llama.cpp 使用     |
| **Safetensors (.safetensors)**        | PyTorch 安全模型文件                  | 在 HF/训练时使用                        |
| **Modelfile**                         | 构建 Ollama 模型的配置脚本               | 让 Ollama 使用 GGUF 或 safetensors 模型 |
| **Modelfile ⇢ 生成本地 Ollama 模型（.blob）** |                                 |                                   |

---

### 2.4 Modelfile 的典型用途

✔️ 改系统人设（system prompt）
✔️ 调温度 num_ctx 等推理参数
✔️ 将 HF 下载的模型转成 Ollama 用
✔️ 加载多个 LoRA 或自定义插件
✔️ 给模型设定人格或使用风格

---

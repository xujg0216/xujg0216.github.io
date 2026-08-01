# vLLM Serve 参数全解（详细版）

> 版本: vLLM 0.18.0 | 命令: `vllm serve --help=all`

---

## 目录

- [选项参数 (Options)](#选项参数-options)
- [Frontend — OpenAI 兼容前端](#frontend--openai-兼容前端)
- [ModelConfig — 模型配置](#modelconfig--模型配置)
- [LoadConfig — 权重加载配置](#loadconfig--权重加载配置)
- [AttentionConfig — 注意力配置](#attentionconfig--注意力配置)
- [StructuredOutputsConfig — 结构化输出](#structuredoutputsconfig--结构化输出)
- [ParallelConfig — 并行配置](#parallelconfig--并行配置)
- [CacheConfig — KV Cache 配置](#cacheconfig--kv-cache-配置)
- [OffloadConfig — 权重卸载](#offloadconfig--权重卸载)
- [MultiModalConfig — 多模态](#multimodalconfig--多模态)
- [LoRAConfig — LoRA 适配器](#loraconfig--lora-适配器)
- [ObservabilityConfig — 可观测性](#observabilityconfig--可观测性)
- [SchedulerConfig — 调度器](#schedulerconfig--调度器)
- [CompilationConfig — 编译与 CUDA 图](#compilationconfig--编译与-cuda-图)
- [KernelConfig — 内核选择](#kernelconfig--内核选择)
- [VllmConfig — 顶级配置容器](#vllmconfig--顶级配置容器)

---

## 选项参数 (Options)

顶层杂项参数，不属于任何配置组。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model_tag` | `str` | `None` | 位置参数，模型名或路径。不传则用 `--model` 的值（默认为 `Qwen/Qwen3-0.6B`） |
| `--config` | `str` | `None` | YAML 配置文件路径，可替代命令行传参。文档：https://docs.vllm.ai/en/latest/configuration/serve_args.html |
| `--disable-log-stats` | `bool` | `False` | 禁用统计日志（定期打印的吞吐/延迟统计）。可减少日志噪音 |
| `--enable-log-requests` | `bool` | `False` | 记录请求详情。INFO 级别记录请求 ID/参数/LoRA；DEBUG 级别记录 prompt 原文。通过 `VLLM_LOGGING_LEVEL` 控制 |
| `--fail-on-environ-validation` | `bool` | `False` | 如果环境校验失败（如显存不足），直接报错退出而非降级运行 |
| `--aggregate-engine-logging` | `bool` | `False` | 数据并行时汇总所有 engine 的统计而非分别打印 |
| `--api-server-count / -asc` | `int` | `None` | API 服务进程数。默认 = `data_parallel_size`。提高可增加 HTTP 处理能力 |
| `--gdn-prefill-backend` | `str` | `None` | GDN prefill 后端。可选 `flashinfer` / `triton`。一般不需要设 |
| `--grpc` | `bool` | `False` | 启动 gRPC 服务取代 HTTP。需 `pip install vllm[grpc]` |
| `--headless` | `bool` | `False` | 无头模式。用于多节点数据并行的从节点。见多节点 DP 文档 |
| `--shutdown-timeout` | `int` | `0` | 优雅关闭超时（秒）。`0` = 立即中止；`>0` = 等待这么久后再强制 kill |

---

## Frontend — OpenAI 兼容前端

控制 HTTP/HTTPS 服务的网络、安全、API 行为。vLLM 启动后提供与 OpenAI 完全兼容的 REST API。

### 网络

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--host` | `str` | `None` | 监听地址。`0.0.0.0` 表示接受所有网络接口的连接。不设则仅本地 |
| `--port` | `int` | `8000` | HTTP 服务端口 |
| `--uds` | `str` | `None` | Unix Domain Socket 路径。如果设置了此项，`host` 和 `port` 会被忽略。适用于同一台机器上的进程间高性能通信 |
| `--root-path` | `str` | `None` | FastAPI `root_path`。当服务运行在反向代理后面且路径被重写时使用。如 nginx 上 `/api/v1` → `/` 的场景 |

### HTTPS / SSL

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--ssl-certfile` | `str` | `None` | SSL 证书文件路径（PEM 格式） |
| `--ssl-keyfile` | `str` | `None` | SSL 私钥文件路径 |
| `--ssl-ca-certs` | `str` | `None` | CA 证书文件，用于验证客户端证书（mTLS） |
| `--ssl-cert-reqs` | `int` | `0` | 是否要求客户端证书。`0`=不需要，`1`=可选，`2`=必须（见 Python ssl 模块文档） |
| `--ssl-ciphers` | `str` | `None` | SSL 加密套件（仅 TLS 1.2 及以下）。例：`'ECDHE-RSA-AES256-GCM-SHA384'` |
| `--enable-ssl-refresh` | `bool` | `False` | 证书文件变更时自动刷新 SSL 上下文，无需重启服务 |

### 安全

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--api-key` | `list[str]` | `None` | API 密钥（可多个）。设置后客户端必须在请求头中携带 `Authorization: Bearer <key>` |
| `--allowed-origins` | `list[str]` | `['*']` | CORS 允许的源（Origin）。生产环境建议限制为具体域名 |
| `--allowed-headers` | `list[str]` | `['*']` | CORS 允许的请求头 |
| `--allowed-methods` | `list[str]` | `['*']` | CORS 允许的 HTTP 方法 |
| `--allow-credentials` | `bool` | `False` | CORS 是否允许发送凭据（Cookie、Authorization 头等） |

### API 行为

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--response-role` | `str` | `assistant` | 响应中 assistant 消息的 role 字段值 |
| `--chat-template` | `str` | 自动 | 自定义 chat 模板文件路径。覆盖模型的默认 chat template |
| `--chat-template-content-format` | `str` | `auto` | chat 模板的内容格式：`auto`、`openai`（结构化内容）、`string`（纯文本） |
| `--default-chat-template-kwargs` | `JSON` | `None` | 传递给 chat 模板的默认参数。如 `{"enable_thinking": false}` 可关闭 Qwen3 的思考模式 |
| `--trust-request-chat-template` | `bool` | `False` | 是否信任客户端在请求中传入的 chat template。有安全风险，仅受信环境使用 |
| `--return-tokens-as-token-ids` | `bool` | `False` | 响应中以 token ID 列表形式返回生成内容 |
| `--tokens-only` | `bool` | `False` | 只返回 token ID，不返回解码后的文本 |
| `--enable-auto-tool-choice` | `bool` | `False` | 当请求中提供了 tools 但未指定 `tool_choice` 时，自动选择是否调用工具 |
| `--enable-request-id-headers` | `bool` | `False` | 在响应头中附加 `X-Request-Id`，方便追踪 |
| `--enable-server-load-tracking` | `bool` | `False` | 服务端负载追踪 |
| `--enable-prompt-tokens-details` | `bool` | `False` | 在 usage 中返回 prompt token 的详细信息 |
| `--enable-force-include-usage` | `bool` | `False` | 流式输出中也强制包含最终 usage 统计 |
| `--enable-log-outputs` | `bool` | `False` | 日志中记录模型输出文本内容 |
| `--enable-log-deltas` | `bool` | `True` | 日志中记录流式输出的增量 token |
| `--enable-tokenizer-info-endpoint` | `bool` | `False` | 启用 tokenizer 信息查询的 API 端点 |
| `--exclude-tools-when-tool-choice-none` | `bool` | `False` | `tool_choice="none"` 时请求是否去除 tools 定义（减少 prompt 长度） |

### 工具调用解析器

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--tool-call-parser` | `str` | 自动 | 指定 tool call 解析器。vLLM 内置支持 30+ 模型的 tool call 格式。常用：`deepseek_v3`、`llama3_json`、`qwen3_coder`、`mistral`、`openai`、`hermes`。完整列表：`deepseek_v3/v31/v32`、`ernie45`、`functiongemma`、`gigachat3`、`glm45/glm47`、`granite/granite-20b-fc/granite4`、`hermes`、`hunyuan_a13b`、`internlm`、`jamba`、`kimi_k2`、`llama3_json`、`llama4_json`、`llama4_pythonic`、`longcat`、`minimax/minimax_m2`、`mistral`、`olmo3`、`openai`、`phi4_mini_json`、`pythonic`、`qwen3_coder`、`qwen3_xml`、`seed_oss`、`step3/step3p5`、`xlam` |
| `--tool-parser-plugin` | `str` | `""` | 自定义 tool call 解析器插件的路径 |
| `--tool-server` | `str` | `None` | 工具服务器地址，用于远程工具执行 |

### LoRA 模块预加载

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--lora-modules` | `list` | `None` | 服务启动时预加载的 LoRA 模块列表。格式：`name1=path1 name2=path2` |

### 中间件

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--middleware` | `list[str]` | `[]` | 额外的 ASGI 中间件（可多次使用），值为 Python 导入路径。如果是函数，通过 `@app.middleware('http')` 添加；如果是类，通过 `app.add_middleware()` 添加 |

### 日志与调试

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--disable-fastapi-docs` | `bool` | `False` | 禁用 `/docs` (Swagger UI)、`/redoc`、`/openapi.json`。生产环境建议开启 |
| `--enable-offline-docs` | `bool` | `False` | 使用内置的静态资源提供离线文档（适用于无互联网的隔离环境） |
| `--disable-frontend-multiprocessing` | `bool` | `False` | 禁用前端多进程模式 |
| `--disable-access-log-for-endpoints` | `str` | `None` | 对特定端点禁用 uvicorn 访问日志。逗号分隔，减少高频端点（如 `/health`）的日志噪音。例：`"/health,/metrics,/ping"` |
| `--disable-uvicorn-access-log` | `bool` | `False` | 完全禁用 uvicorn 的 HTTP 访问日志 |
| `--uvicorn-log-level` | `str` | `info` | uvicorn 日志级别：`critical` / `error` / `warning` / `info` / `debug` / `trace` |
| `--log-config-file` | `str` | `None` | Python logging 配置文件路径 |
| `--log-error-stack` | `bool` | `False` | 记录错误时附带完整堆栈信息 |
| `--max-log-len` | `int` | `None` | 每条日志的最大字符数，超过则截断 |

### HTTP 协议限制

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--h11-max-header-count` | `int` | `256` | 单请求允许的最大 HTTP 头数量（h11 解析器）。防止 HTTP 头滥用攻击 |
| `--h11-max-incomplete-event-size` | `int` | `4194304` (4MB) | 不完整 HTTP 事件（请求头或请求体）的最大字节数。防止慢速请求耗尽内存 |

---

## ModelConfig — 模型配置

控制模型的选择、加载方式、精度和行为。

### 模型选择

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--model` | `str` | `Qwen/Qwen3-0.6B` | HuggingFace 模型 ID 或本地路径。Prometheus 指标中 `model_name` 标签的默认值 |
| `--model-impl` | `str` | `auto` | 模型实现来源：`auto`=优先 vLLM 实现→fallback Transformers；`vllm`=只用 vLLM；`transformers`=只用 Transformers（无优化）；`terratorch`=TerraTorch 实现 |
| `--runner` | `str` | `auto` | Runner 类型：`auto`=自动检测；`generate`=文本生成；`draft`=投机解码的草案模型；`pooling`=嵌入/池化模型 |
| `--convert` | `str` | `auto` | 模型用途转换：`auto`=不转换；`classify`=转为分类；`embed`=转为嵌入；`none`=不转换 |

### 精度

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--dtype` | `str` | `auto` | 模型权重和激活值的精度。`auto`=FP32/FP16 模型用 FP16，BF16 模型用 BF16；`half`/`float16`=FP16（AWQ 量化时推荐）；`bfloat16`=精度与范围的平衡（推荐）；`float`/`float32`=FP32（最耗显存） |
| `--override-attention-dtype` | `str` | `None` | 单独覆盖注意力计算的精度（与权重精度分开控制） |

### 上下文与长度

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--max-model-len` | `int`/`str` | 自动 | **模型上下文窗口**（prompt + output 的 token 总数上限）。不设则自动从模型 config 读取（Qwen3-0.6B=40960）。支持缩写：`4k`=4000，`4K`=4096，`25.6k`=25600。设为 `-1` 或 `auto` = 自动选能装进 GPU 的最大值 |

### 量化

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--quantization / -q` | `str` | `None` | 权重量化方法。`None`=自动从模型 config 的 `quantization_config` 读取，没有则按 `dtype` 加载。支持的值取决于模型，常见：`awq`、`gptq`、`fp8`、`compressed-tensors`、`bitsandbytes` |
| `--allow-deprecated-quantization` | `bool` | `False` | 是否允许使用已弃用的量化方法 |

### HuggingFace 集成

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--hf-config-path` | `str` | `None` | 指定 HuggingFace config.json 的路径（通常不需要，与 model 路径不同时使用） |
| `--hf-overrides` | `dict` | `{}` | 覆盖 HF config 中的字段。如 `{"num_hidden_layers": 20}` |
| `--hf-token` | `str`/`bool` | `None` | HuggingFace API token。字符串直接作为 token；`True` = 用 `~/.cache/huggingface/token`（即 `hf auth login` 的结果） |
| `--revision` | `str` | `None` | 模型权重版本（分支名、tag 名、commit hash） |
| `--code-revision` | `str` | `None` | 模型代码版本（HuggingFace Hub 上的远程代码） |
| `--trust-remote-code` | `bool` | `False` | 是否信任并执行 HuggingFace 仓库中的自定义代码。有些模型的 tokenizer 或配置需要此选项 |
| `--config-format` | `str` | `auto` | 配置文件格式：`auto`=先试 hf 再试 mistral；`hf`=HuggingFace 格式；`mistral`=Mistral 格式 |

### Tokenizer

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--tokenizer` | `str` | 同 model | Tokenizer 名称或路径（如果和模型路径不同） |
| `--tokenizer-mode` | `str` | `auto` | Tokenizer 模式：`auto`=Mistral 模型用 `mistral_common`，其他用 `hf`；`hf`=快速 tokenizer（推荐）；`slow`=Python 实现（更准确但更慢）；`mistral`=Mistral 专用；`deepseek_v32`=DeepSeek V3.2 专用 |
| `--tokenizer-revision` | `str` | `None` | Tokenizer 的版本号（分支/tag/commit） |
| `--skip-tokenizer-init` | `bool` | `False` | 跳过 tokenizer 初始化。此时只接受 `prompt_token_ids` 输入，输出也只有 token ID |

### 生成配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--generation-config` | `str` | `auto` | 生成配置来源：`auto`=从模型路径加载 `generation_config.json`；`vllm`=不加载，用 vLLM 默认值；具体路径=从指定路径加载。**如果 generation_config.json 里设了 `max_new_tokens`，则它是服务端的全局上限** |
| `--override-generation-config` | `JSON` | `{}` | 覆盖或合并生成配置。如 `{"temperature": 0.5, "max_new_tokens": 1024}`。配合 `--generation-config auto` 时是合并；配合 `vllm` 时是替换 |

### 其他模型配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--seed` | `int` | `0` | 全局随机种子。必须全局设置是因为张量并行时不同 worker 需要采样出相同的 token |
| `--enforce-eager` | `bool` | `False` | `True`=禁用 CUDA Graph 和 torch.compile，始终用 eager 模式执行。调试时有用但严重损失性能 |
| `--enable-sleep-mode` | `bool` | `False` | 启用引擎睡眠模式（仅 CUDA/HIP）。空闲时释放 GPU 显存，收到请求时重新加载。适合与其他进程共享 GPU 的场景 |
| `--disable-cascade-attn` | `bool` | `True` | 禁用级联注意力（cascade attention）。默认 `True`（即不启用），要使用需显式设为 `False`。不影响数学正确性，只是 heuristic 优化 |
| `--disable-sliding-window` | `bool` | `False` | 禁用滑动窗口注意力。即使模型支持（如 Mistral），也强制使用全局注意力 |
| `--enable-prompt-embeds` | `bool` | `False` | 允许通过 API 直接传入文本 embedding 替代 prompt 文本。有安全风险，仅受信用户使用。形状不正确会导致引擎崩溃 |
| `--enable-return-routed-experts` | `bool` | `False` | 返回 MoE 模型中每个 token 路由到了哪个 expert |
| `--served-model-name` | `list[str]` | 同 model | 在 API `/v1/models` 中暴露的模型名。可设多个（别名）。第一个名字作为 API 响应中的 `model` 字段和 Prometheus 指标的 `model_name` 标签 |
| `--max-logprobs` | `int` | `20` | 请求中指定 `logprobs` 时，最多返回多少个 token 的对数概率。设为 `-1` 则不设上限（可能 OOM） |
| `--logprobs-mode` | `str` | `raw_logprobs` | logprobs 返回的模式：`raw_logprobs` / `raw_logits` = 未应用处理器前的值；`processed_logprobs` / `processed_logits` = 应用 temperature/top_k/top_p 等处理器后的值 |
| `--logits-processors` | `list[str]` | `None` | 自定义 logits 处理器的完整类名列表 |
| `--pooler-config` | `JSON` | `None` | Pooling 模型的 Pooler 配置，控制输出池化行为 |
| `--io-processor-plugin` | `str` | `None` | IO 处理器的插件名，模型启动时加载 |

---

## LoadConfig — 权重加载配置

控制模型权重文件的加载行为。

### 加载格式

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--load-format` | `str` | `auto` | 权重加载格式。详细选项：<br>`auto`=先试 safetensors，失败则用 PyTorch bin<br>`safetensors`=强制 safetensors 格式<br>`pt`=强制 PyTorch `.bin` 格式<br>`instanttensor`=CUDA 上的 safetensors 分布式加载，支持流水线预取和快速 Direct I/O<br>`npcache`=PyTorch 格式 + 额外存 numpy 缓存加速后续加载<br>`dummy`=随机初始化权重（仅用于代码分析和基准测试，不能推理）<br>`tensorizer`=CoreWeave 的 tensorizer 库快速加载<br>`runai_streamer`=Run:ai Model Streamer 加载<br>`runai_streamer_sharded`=Run:ai Streamer 从预分片 checkpoint 加载<br>`bitsandbytes`=使用 bitsandbytes 库加载量化权重<br>`sharded_state`=从预分片 checkpoint 加载（加速 TP 模型冷启动）<br>`gguf`=加载 GGUF 格式文件（llama.cpp 生态）<br>`mistral`=Mistral 专用的合并 safetensors 格式 |
| `--download-dir` | `str` | HF 默认缓存 | 模型权重下载目录。不设则用 HuggingFace 默认缓存路径 |
| `--ignore-patterns` | `list[str]` | `['original/**/*']` | 加载时忽略的文件模式（glob）。默认忽略 `original/` 目录避免重复加载 LLaMA 的原始权重 |

### 加载优化

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--safetensors-load-strategy` | `str` | `lazy` | safetensors 加载策略：<br>`lazy`（默认）=内存映射，按需加载。本地磁盘最优<br>`eager`=预先读入 CPU 内存。网络文件系统（Lustre/NFS）推荐，避免随机读延迟，但吃更多 CPU RAM<br>`prefetch`=预先读入 OS 页面缓存再加载。适合网络或高延迟存储<br>`torchao`=预先读入后重建为 torchao 张量子类（torchao 量化 checkpoint 专用，需 torchao ≥0.14.0） |
| `--use-tqdm-on-load` | `bool` | `True` | 加载权重时显示进度条 |
| `--pt-load-map-location` | `str`/`dict` | `cpu` | PyTorch checkpoint 加载时的 `map_location` 参数。不支持 safetensors。例：`{"": "cuda"}` 直接加载到 GPU；`{"cuda:1": "cuda:0"}` 重映射设备。命令行使用时字典内字符串需双引号 |

### 高级

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--model-loader-extra-config` | `dict` | `{}` | 传递给对应 `load_format` 的模型加载器的额外配置 |

---

## AttentionConfig — 注意力配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--attention-backend` | `str` | `None` | 注意力后端。`None`=自动选择。可选值取决于编译进来的后端，通常包括 `FLASH_ATTN`（默认）、`FLASHINFER`、`TRITON_ATTN`、`FLEX_ATTENTION` |

通过 `--attention-config / -ac` 可以传入更详细的 JSON 配置：
```bash
-ac '{"backend": "FLASH_ATTN", "flash_attn_version": 2, "use_prefill_decode_attention": false}'
```

AttentionConfig 完整字段（默认值）：
| 字段 | 默认值 | 说明 |
|---|---|---|
| `backend` | `None` | 注意力后端 |
| `flash_attn_version` | `None` | Flash Attention 版本 |
| `use_prefill_decode_attention` | `False` | prefill 和 decode 是否使用统一注意力 |
| `flash_attn_max_num_splits_for_cuda_graph` | `32` | FA CUDA 图的最大 splits 数 |
| `use_cudnn_prefill` | `False` | 是否使用 cuDNN 进行 prefill |
| `use_trtllm_ragged_deepseek_prefill` | `False` | DeepSeek ragged prefill 专用 |
| `use_trtllm_attention` | `None` | TRT-LLM 注意力 |
| `disable_flashinfer_prefill` | `True` | 禁用 FlashInfer prefill |
| `disable_flashinfer_q_quantization` | `False` | 禁用 FlashInfer 的 Q 量化 |
| `use_prefill_query_quantization` | `False` | prefill 阶段量化 Q |

---

## StructuredOutputsConfig — 结构化输出

控制推理内容解析（`<think>...</think>` 标记的拆分）和结构化输出（guided decoding）。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--reasoning-parser` | `str` | `""` | 推理内容解析器。如 `deepseek_r1`。用于拆分 `<think>推理过程</think>最终答案`，使 API 返回中 `reasoning_content` 和 `content` 分开 |
| `--reasoning-parser-plugin` | `str` | `""` | 自定义推理解析器插件的路径，可动态加载注册 |

完整的结构化输出配置通过 `--structured-outputs-config` JSON：
```bash
--structured-outputs-config '{
  "backend": "auto",
  "disable_any_whitespace": false,
  "disable_additional_properties": false,
  "reasoning_parser": "deepseek_r1",
  "enable_in_reasoning": false
}'
```

---

## ParallelConfig — 并行配置

控制分布式推理的 GPU 分配策略。vLLM 支持 **五种并行维度**：

| 并行方式 | 参数 | 默认值 | 说明 |
|---|---|---|---|
| **张量并行 (TP)** | `--tensor-parallel-size / -tp` | `1` | 每个 Transformer 层的权重分散到多 GPU |
| **流水线并行 (PP)** | `--pipeline-parallel-size / -pp` | `1` | 不同层放到不同 GPU，按流水线执行 |
| **数据并行 (DP)** | `--data-parallel-size / -dp` | `1` | 多份模型副本，不同请求分到不同副本 |
| **上下文并行 (CP)** | `--decode-context-parallel-size / -dcp` | `1` | 长序列分散到多 GPU 并行计算注意力 |
| **专家并行 (EP)** | `--enable-expert-parallel / -ep` | `False` | MoE 的 expert 权重分散到多 GPU |

### 张量并行 (TP)

将每层权重按列/行切分到多个 GPU，**减少单 GPU 显存占用，加速单请求**。

```bash
--tensor-parallel-size 2      # 2 卡跑一个 7B 模型
--tensor-parallel-size 4      # 4 卡跑一个 70B 模型
```

### 数据并行 (DP)

多份完整模型副本，**提高吞吐量**。Erlang 对 MoE 权重进行 DP 分片。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--data-parallel-size / -dp` | `int` | `1` | DP 组数。MoE 层的 expert 会按 `tp_size × dp_size` 分片 |
| `--data-parallel-size-local / -dpl` | `int` | `None` | 本节点上的 DP 副本数 |
| `--data-parallel-rank / -dpn` | `int` | `None` | 本实例的 DP rank。设置后启用外部负载均衡模式 |
| `--data-parallel-start-rank / -dpr` | `int` | `None` | 从节点的起始 DP rank |
| `--data-parallel-backend / -dpb` | `str` | `mp` | DP 后端：`mp`（multiprocessing，单机推荐）、`ray`（多机） |
| `--data-parallel-address / -dpa` | `str` | `None` | DP 集群头节点地址 |
| `--data-parallel-rpc-port / -dpp` | `int` | `None` | DP RPC 通信端口 |
| `--data-parallel-external-lb / -dpe` | `bool` | `False` | 外部负载均衡模式（Kubernetes "one-pod-per-rank" 架构） |
| `--data-parallel-hybrid-lb / -dph` | `bool` | `False` | 混合 LB 模式：vLLM 内部负载均衡到本地 DP ranks，外部 LB 跨节点 |

### 上下文并行 (CP)

将单个长序列的 KV Cache 切分到多 GPU，**处理超长上下文**。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--decode-context-parallel-size / -dcp` | `int` | `1` | Decode 阶段上下文并行组数。复用 TP 的 GPU，`tp_size` 必须能被 `dcp_size` 整除 |
| `--prefill-context-parallel-size / -pcp` | `int` | `1` | Prefill 阶段上下文并行组数 |
| `--dcp-comm-backend` | `str` | `ag_rs` | DCP 通信后端：`ag_rs`=AllGather+ReduceScatter（默认，每层 3 次 NCCL 调用）；`a2a`=All-to-All 交换局部输出 + Triton 融合（MLA 模型每层从 3 次降到 2 次） |
| `--cp-kv-cache-interleave-size` | `int` | `1` | KV cache 在 CP ranks 间的交织粒度。`1`=token 级交织；`block_size`=块级交织。需被 block_size 整除，且 ≤ block_size |

### 专家并行 (EP)

专门优化 MoE 模型的 expert 权重分布。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--enable-expert-parallel / -ep` | `bool` | `False` | 开启专家并行。MoE 层用 EP 替代 TP，非 MoE 层仍用 TP |
| `--expert-placement-strategy` | `str` | `linear` | Expert 放置策略：`linear`=连续分配（如 rank0=[0,1], rank1=[2,3]）；`round_robin`=轮转分配（如 rank0=[0,2], rank1=[1,3]）。轮转对 grouped expert 模型的负载均衡更好 |
| `--enable-elastic-ep` | `bool` | `False` | 弹性 EP，允许运行时动态调整 expert 分布 |
| `--enable-eplb` | `bool` | `False` | 专家并行负载均衡，监控并重平衡各 rank 的 expert 负载 |
| `--eplb-config` | `JSON` | 见下方 | EP 负载均衡的详细配置 |
| `--enable-ep-weight-filter` | `bool` | `False` | 每个 rank 只从磁盘加载本地的 expert 分片，显著减少大 MoE（DeepSeek/Mixtral/Kimi-K2）的存储 I/O |
| `--all2all-backend` | `str` | `allgather_reducescatter` | EP all2all 通信后端。选项：`naive`（广播）、`allgather_reducescatter`（默认）、`deepep_high_throughput`（高吞吐）、`deepep_low_latency`（低延迟）、`mori`、`nixl_ep`、`flashinfer_nvlink_two_sided`（NVLink）、`flashinfer_nvlink_one_sided`、`pplx` |

EPLB 默认配置：
```json
{
  "window_size": 1000,
  "step_interval": 3000,
  "num_redundant_experts": 0,
  "log_balancedness": false,
  "log_balancedness_interval": 1,
  "use_async": false,
  "policy": "default"
}
```

### DBO (Dual Batch Overlap)

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--enable-dbo` | `bool` | `False` | 启用双批次重叠执行。将一个批次拆成两个微批次流水线化，隐藏通信延迟 |
| `--dbo-decode-token-threshold` | `int` | `32` | 纯 decode 批次的 DBO 微批次阈值。大于此 token 数才用微批次 |
| `--dbo-prefill-token-threshold` | `int` | `512` | 包含 prefill 批次的 DBO 微批次阈值 |

### 多节点

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--nnodes / -n` | `int` | `1` | 总节点数（多机推理） |
| `--node-rank / -r` | `int` | `0` | 本节点的 rank（0 起始） |
| `--master-addr` | `str` | `127.0.0.1` | 分布式通信的 master 地址 |
| `--master-port` | `int` | `29501` | 分布式通信的 master 端口 |
| `--distributed-timeout-seconds` | `int` | `None` | 分布式操作超时（秒）。默认用 PyTorch NCCL 的默认值（600s）。多节点下载模型慢时可加大 |
| `--distributed-executor-backend` | `str` | `None` | 分布式执行后端：`mp`=multiprocessing（单机，tp×pp≤GPU 数时默认）；`ray`=Ray（多机或 TPU 必须）；`uni`=单进程；`external_launcher`=外部启动器 |
| `--disable-custom-all-reduce` | `bool` | `False` | 禁用 vLLM 的优化 all-reduce kernel，回退到 NCCL |
| `--disable-nccl-for-dp-synchronization` | `bool` | `None` | DP 同步时用 Gloo 替代 NCCL |
| `--max-parallel-loading-workers` | `int` | `None` | 模型分多批次加载时的最大并行 worker 数。防止 TP 加载大模型时 CPU RAM OOM |
| `--worker-cls` | `str` | `auto` | Worker 类的完整路径。`auto`=根据平台自动选择。自定义 worker 类注入新行为 |
| `--worker-extension-cls` | `str` | `""` | Worker 扩展类。通过动态继承注入新属性和方法到 worker 类，供 `collective_rpc` 调用使用 |
| `--ubatch-size` | `int` | `0` | 微批次大小（实验性） |
| `--ray-workers-use-nsight` | `bool` | `False` | 是否用 Nsight 分析 Ray worker |

---

## CacheConfig — KV Cache 配置

控制 PagedAttention 的关键——KV Cache 的内存分配和行为。直接影响 **显存占用** 和 **最大并发数**。

### 内存控制

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--gpu-memory-utilization` | `float` | `0.9` | **最重要的内存参数**。KV Cache 可用 GPU 显存的比例（0~1）。`0.5`=50%。这是**每个实例的限制**，不影响同 GPU 上的其他 vLLM 实例 |
| `--kv-cache-memory-bytes` | `int`/`str` | `None` | 每个 GPU 的 KV Cache 精确大小（字节），支持 `1k`/`1M` 等缩写。**设了这个后 `gpu_memory_utilization` 会被忽略**。用于需要精确控制内存的场景 |
| `--block-size` | `int` | `None`（自动） | PagedAttention 的块大小（token 数）。越大 → 管理开销越小但碎片越多。不设则由 vLLM 自动选择（通常 16 或 256） |
| `--num-gpu-blocks-override` | `int` | `None` | 强制指定 GPU block 数量（用于测试抢占行为） |

### 数据类型

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--kv-cache-dtype` | `str` | `auto` | KV Cache 存储精度。`auto`=跟随模型精度；`fp8`/`fp8_e4m3`（CUDA 11.8+）；`fp8_e5m2`；`fp8_inc`（Intel Gaudi）；`bfloat16`；`float16`。FP8 可将 KV Cache 内存减半，轻微损失精度。DeepSeekV3.2 等模型默认 FP8 |
| `--calculate-kv-scales` | `bool` | `False` | KV Cache dtype 为 FP8 时，是否在运行时动态计算缩放因子。`False`=从 checkpoint 加载（若存在），否则 scale=1.0 |

### 前缀缓存

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--enable-prefix-caching` | `bool` | `None` | 是否启用前缀缓存。相同前缀的请求（如共享 system prompt）复用 KV Cache，大幅减少 prefill 时间。vLLM 实现比 nano-vllm 的 xxhash 方案更复杂，支持全自动的 APC（Automatic Prefix Caching） |
| `--prefix-caching-hash-algo` | `str` | `sha256` | 前缀缓存哈希算法。选项：<br>`sha256`（默认）=Pickle 序列化 + SHA-256，最安全避免哈希碰撞<br>`sha256_cbor`=CBOR 序列化 + SHA-256，跨语言可复现<br>`xxhash`=Pickle + xxHash (128-bit)，更快但有碰撞风险（多租户环境有安全隐患）<br>`xxhash_cbor`=CBOR + xxHash，可复现 |

### KV Cache CPU 卸载

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--kv-offloading-size` | `float` | `None` | KV Cache CPU 卸载大小（GiB）。TP>1 时是所有 TP rank 的总和。设置后启用 KV Cache CPU 卸载，用 CPU 内存扩展 KV Cache 容量（以速度为代价） |
| `--kv-offloading-backend` | `str` | `native` | KV Cache 卸载后端：`native`（vLLM 原生 CPU 卸载）、`lmcache`（LMCache） |

### Mamba 缓存

Mamba（状态空间模型）有自己的缓存格式，不同于 Transformer 的 KV Cache。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--mamba-block-size` | `int` | `None` | Mamba 缓存块大小（token 数）。需为 8 的倍数（对齐 causal_conv1d kernel）。仅当前缀缓存开启时可设置 |
| `--mamba-cache-dtype` | `str` | `auto` | Mamba 缓存精度（卷积状态 + SSM 状态）：`auto`/`float16`/`float32` |
| `--mamba-ssm-cache-dtype` | `str` | `auto` | 仅 SSM 状态的精度（卷积状态仍用 `mamba_cache_dtype`） |
| `--mamba-cache-mode` | `str` | `none` | Mamba 缓存策略：`none`=无前缀缓存时；`all`=在每 `i × block_size` 位置缓存全部状态；`align`=仅在每步调度末尾 + `i × block_size` 位置缓存 |

### 实验性

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--kv-sharing-fast-prefill` | `bool` | `False` | 实验性。KV 共享优化 prefill（YOCO 等架构）。当前版本设置此 flag 并无实际效果 |

---

## OffloadConfig — 权重卸载

当 GPU 显存不够装模型权重时，将部分权重移到 CPU 内存。**用 CPU RAM + 带宽换取显存**。

### UVA (Unified Virtual Addressing) 零拷贝卸载

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--cpu-offload-gb` | `float` | `0` | 每个 GPU 卸载到 CPU 的大小（GiB）。设为 10 → 24GB GPU 虚拟成 34GB（可加载 13B BF16 模型=~26GB）。利用 UVA 做到零拷贝传输。需快速 CPU-GPU 互联（PCIe 4.0+） |
| `--cpu-offload-params` | `set[str]` | `set()` | 指定哪些参数名段卸载（按名称的 "." segment 匹配）。为空=自动选择参数直到达到 `cpu_offload_gb` 的显存限制。例：`"experts"` 匹配 `mlp.experts.w2_weight`；`"experts.w2_weight"` 精确匹配；`"expert"` / `"w2"` 无法匹配（必须是完整 segment） |

### Prefetch 卸载

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--offload-group-size` | `int` | `0` | 将每 N 层分为一组。每组中最后 `offload_num_in_group` 层卸载。例：`group_size=8, num_in_group=2` → 卸载第 6,7,14,15,22,23... 层 |
| `--offload-num-in-group` | `int` | `1` | 每组卸载的层数，需 ≤ `offload_group_size` |
| `--offload-params` | `set[str]` | `set()` | 指定预取卸载的参数名段。为空=卸载被选中层的全部参数。使用 segment 匹配：`"w13_weight"` 匹配 `mlp.experts.w13_weight` 但不匹配 `mlp.experts.w13_weight_scale` |
| `--offload-prefetch-step` | `int` | `1` | 预取提前步数。越大越能隐藏传输延迟，但占更多 GPU 内存 |
| `--offload-backend` | `str` | `auto` | 卸载后端：`auto`=有 `offload_group_size` 时用 `prefetch`，有 `cpu_offload_gb` 时用 `uva`；`uva`=UVA 零拷贝；`prefetch`=异步预取 |

---

## MultiModalConfig — 多模态

控制视觉/音频等多模态模型的输入处理。

### 输入控制

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--limit-mm-per-prompt` | `JSON` | `{}` | 每种模态每个 prompt 最大输入数。默认为 999。格式：<br>旧版（仅数量）：`{"image": 16, "video": 2}`<br>新版（含选项）：`{"video": {"count": 1, "num_frames": 32, "width": 512, "height": 512}}`<br>混用：`{"image": 16, "video": {"count": 1, "num_frames": 32}}` |
| `--language-model-only` | `bool` | `False` | 禁用所有多模态输入（所有模态 limit = 0）。等价于把所有模态的 limit 设为 0 |
| `--enable-mm-embeds` | `bool` | `False` | 允许直接传入多模态 embedding。有安全风险，仅受信用户。与 `limit-mm-per-prompt=0` 配合可跳过编码器加载，只接受预计算的 embedding |
| `--interleave-mm-strings` | `bool` | `False` | 启用图片和文本完全交错的多模态 prompt（需 `--chat-template-content-format=string`） |

### 编码器

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--mm-encoder-attn-backend` | `str` | `None` | 覆盖多模态编码器（视觉 Transformer）的注意力后端，如 `FLASH_ATTN` |
| `--mm-encoder-only` | `bool` | `False` | 仅运行多模态编码器，跳过语言模型。通常只用于分离式 encoder 进程 |
| `--mm-encoder-tp-mode` | `str` | `weights` | 编码器 TP 模式：`weights`=权重切分（默认）；`data`=数据切分（每 rank 有完整权重，并行处理不同输入）。仅部分模型支持 |

### 缓存与处理

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--mm-processor-cache-gb` | `float` | `4` | 多模态处理器缓存大小（GiB）。避免对相同的多模态输入重复处理。总内存消耗 = `cache_size × (api_server_count + data_parallel_size)`。设为 `0` 禁用 |
| `--mm-processor-cache-type` | `str` | `lru` | 缓存类型：`lru`=镜像 LRU 缓存；`shm`=共享内存 FIFO 缓存（跨进程共享） |
| `--mm-shm-cache-max-object-size-mb` | `int` | `128` | SHM 缓存中单个对象最大 MB（仅 `shm` 模式有效） |
| `--mm-processor-kwargs` | `JSON` | `None` | 传递给 `AutoProcessor.from_pretrained` 的覆盖参数。根据模型而异。如 Phi-3-Vision：`{"num_crops": 4}` |
| `--media-io-kwargs` | `JSON` | `{}` | 媒体 IO 处理的额外参数，按模态分别配置。例：`{"video": {"num_frames": 40}}` |
| `--skip-mm-profiling` | `bool` | `False` | 跳过多模态内存分析，加速启动。但用户需自行估算编码器激活值的内存需求 |
| `--video-pruning-rate` | `float` | `None` | 视频 Token 裁剪率 [0, 1)，决定每帧视频保留多少 token |

### 安全

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--allowed-local-media-path` | `str` | `""` | 允许 API 请求读取本地图片/视频的目录路径。有安全风险，仅受信环境使用 |
| `--allowed-media-domains` | `list[str]` | `None` | 允许的多模态输入 URL 域名白名单。不设则不限制 |

---

## LoRAConfig — LoRA 适配器

LoRA 允许基础模型在**运行时不修改权重**的情况下适配不同任务。多个 LoRA 可同时在内存中，按请求切换。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--enable-lora` | `bool` | `None` | 启用 LoRA 适配器支持。设为 `True` 后可通过 API 请求头指定使用哪个 LoRA |
| `--max-loras` | `int` | `1` | 单个批次中可同时使用的最大 LoRA 数量（不同请求可用不同 LoRA） |
| `--max-lora-rank` | `int` | `16` | 最大 LoRA rank。可选值：`1`/`8`/`16`/`32`/`64`/`128`/`256`/`320`/`512` |
| `--max-cpu-loras` | `int` | `None` | CPU 内存中可缓存的最大 LoRA 数量。需 ≥ `max_loras`。LRU 淘汰，存不下的从磁盘懒加载 |
| `--lora-dtype` | `str` | `auto` | LoRA 权重的精度：`auto`（跟随基础模型）、`float16`、`bfloat16` |
| `--fully-sharded-loras` | `bool` | `False` | 完全分片 LoRA 计算（默认仅一半分片）。在高序列长度、高 rank 或大 TP 时可能更快 |
| `--specialize-active-lora` | `bool` | `False` | 根据活跃 LoRA 数量特化 CUDA 图。会为不同的活跃 LoRA 数量（2 的幂次，最多 `max_loras`）捕获各自的 CUDA 图。提高变 LoRA 数场景的性能，但增加启动时间和显存 |
| `--enable-tower-connector-lora` | `bool` | `False` | 多模态模型（如 Qwen VL）的视觉 tower 和 connector 也接受 LoRA。实验性功能 |
| `--default-mm-loras` | `JSON` | `None` | 多模态模型特定模态的默认 LoRA 映射。如 `{"image": "/path/to/image_lora"}`。同一请求的多个模态各有 LoRA 时只生效一个 |

---

## ObservabilityConfig — 可观测性

监控和调试。

### 指标

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--otlp-traces-endpoint` | `str` | `None` | OpenTelemetry traces 上报的目标 URL（用于 Jaeger/Zipkin 等） |
| `--collect-detailed-traces` | `list[str]` | `None` | 开启详细 trace 的模块：`all` / `model` / `worker`。可能有性能影响 |
| `--kv-cache-metrics` | `bool` | `False` | 启用 KV Cache 指标（块生命周期、空闲时间、复用间隔）。采样统计，开销低 |
| `--kv-cache-metrics-sample` | `float` | `0.01` | KV Cache 指标采样率（0.0, 1.0]。默认 1% |
| `--cudagraph-metrics` | `bool` | `False` | 启用 CUDA 图指标（pad/unpad token 数、dispatch 模式频率） |
| `--enable-mfu-metrics` | `bool` | `False` | 启用 MFU 指标（MFU = Model FLOPs Utilization，衡量 GPU 利用率） |
| `--show-hidden-metrics-for-version` | `str` | `None` | 显示从指定版本起隐藏的已弃用指标。例：`0.7` 可临时恢复 v0.7 起隐藏的旧指标 |

### 追踪与日志

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--enable-layerwise-nvtx-tracing` | `bool` | `False` | 逐层 NVTX 追踪。在每个层/模块执行时打 NVTX 标记（含输入输出 shape）。需禁用 CUDA 图 |
| `--enable-logging-iteration-details` | `bool` | `False` | 详细记录每次迭代信息（context/generation 请求数、token 数、CPU 时间） |

---

## SchedulerConfig — 调度器

控制 Continuous Batching 的行为——如何将请求打包成批次、如何排序。

### 并发控制

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--max-num-seqs` | `int` | 自动 | **最大并发序列数**（单次迭代最多处理的序列数） |
| `--max-num-batched-tokens` | `int`/`str` | 自动 | **单次迭代最大处理 token 数**。影响显存峰值和计算吞吐。支持 `1k`/`1K` 等缩写。不设时自动推导 |

### Chunked Prefill

将大的 prefill 请求拆分成多个小块，与 decode 请求交错执行。避免单个长 prefill 阻塞所有 decode 请求（改善 TTFT）。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--enable-chunked-prefill` | `bool` | `True` | 启用分块预填充。当 prefill 的 token 数超过剩余 `max_num_batched_tokens` 时分块 |
| `--max-num-partial-prefills` | `int` | `1` | 最多同时进行多少个分块 prefill 的序列 |
| `--max-long-partial-prefills` | `int` | `1` | 最多同时进行多少个"长 prompt"（超阈值）的分块 prefill。设得比 `max_num_partial_prefills` 小可以让短 prompt 插队，降低短请求延迟 |
| `--long-prefill-token-threshold` | `int` | `0` | 长 prompt 的 token 数阈值。超过此值的 prefill 被视为"长" |

### 调度策略

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--scheduling-policy` | `str` | `fcfs` | 调度策略：`fcfs`=先来先服务；`priority`=按请求优先级排序（值越小越优先，平局按到达时间） |
| `--async-scheduling` | `bool` | `True` | 异步调度。GPU 执行当前批次时 CPU 预先准备下一批次的调度。避免 GPU 空闲气泡，改善延迟和吞吐 |

### 流式输出

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--stream-interval` | `int` | `1` | 流式输出缓冲区大小（token 数）。`1`=每个 token 立即推送（最流畅）；`10`=攒够 10 个一起推（减少 host 开销，可能提高吞吐） |

### 其他

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--disable-hybrid-kv-cache-manager` | `bool` | `None` | `True`=所有注意力层分配相同大小的 KV Cache（即使有 full attention 和 sliding window attention 混合）。`None`=自动根据环境和配置决定 |
| `--disable-chunked-mm-input` | `bool` | `False` | 多模态输入不分块。当 chunked prefill 开启且有混合 prompt（文本+图像 tokens 交错），确保图像 token 不被拆散 |
| `--scheduler-cls` | `str` | `None` | 自定义调度器类路径。默认用 `vllm.v1.core.sched.scheduler.Scheduler`。格式：`"mod.custom_class"` |

---

## CompilationConfig — 编译与 CUDA 图

控制 torch.compile（Inductor）和 CUDA Graph 的编译优化行为。通过 `-cc` 简写传入。

### 便捷写法

```bash
# 完全禁用（等价于 --enforce-eager）
-cc.mode=none -cc.cudagraph_mode=none

# 仅禁用 CUDA 图
-cc.cudagraph_mode=none

# 仅禁用 inductor
-cc.mode=none

# 自定义 CUDA 图捕获大小
-cc.cudagraph_capture_sizes='[1,2,4,8]'

# 完整 JSON
-cc='{"mode":3,"cudagraph_capture_sizes":[1,2,4,8,16,32],"backend":"inductor"}'
```

### 顶层编译控制

| 参数 (via JSON) | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `mode` | `int` | `None` | 编译模式：`0`(NONE)=无；`1`(DYNAMO_ONCE)=一次 dynamo；`2`(DYNAMO_AS_NEEDED)=按需；`3`(DYNAMO_FULL)=完整 dynamo+inductor |
| `backend` | `str` | `inductor` | 编译后端，默认 inductor |
| `custom_ops` | `list[str]` | `[]` | 自定义算子列表（`"all"`=全部开启） |
| `splitting_ops` | `list[str]` | `None` | 需要拆分编译的算子 |
| `compile_mm_encoder` | `bool` | `False` | 是否编译多模态编码器 |
| `compile_sizes` | `list[int]` | `None` | 为特定 batch size 编译（完全静态图，更多优化空间） |
| `compile_ranges_endpoints` | `list[int]` | `None` | 编译范围端点（动态图，适配多种 batch size） |
| `debug_dump_path` | `str` | `None` | 调试输出路径 |
| `cache_dir` | `str` | `""` | 编译缓存目录（跨进程复用编译结果） |
| `compile_cache_save_format` | `str` | `binary` | 缓存保存格式 |
| `local_cache_dir` | `str` | `None` | 本地缓存目录（节点间共享时） |
| `fast_moe_cold_start` | `bool` | `None` | MoE 快速冷启动 |
| `static_all_moe_layers` | `list` | `[]` | 静态 MoE 层列表 |

### CUDA 图

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `cudagraph_mode` | `int` | `None` | CUDA 图模式：`0`(NONE)=不使用；`1`(FULL_DECODE_ONLY)=仅 decode 阶段；`2`(FULL)=全部使用 |
| `cudagraph_capture_sizes` | `list[int]` | `None` | 手动指定捕获的 batch size。CUDA 图只能在**相同 batch size** 下回放。不设则自动生成：`[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_cudagraph_capture_size+1, 16))` |
| `max_cudagraph_capture_size` | `int` | `min(max_num_seqs×2, 512)` | 最大 CUDA 图捕获大小。限制这个值防止小内存下 OOM 和启动时间过长 |
| `cudagraph_num_of_warmups` | `int` | `0` | 捕获前预热次数 |
| `cudagraph_copy_inputs` | `bool` | `False` | 是否拷贝输入（用于非 CUDA 图模式） |
| `cudagraph_specialize_lora` | `bool` | `True` | 是否为不同 LoRA 适配器特化 CUDA 图 |

### Inductor 编译

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `inductor_compile_config` | `dict` | 见下方 | inductor 编译配置。默认：`{"enable_auto_functionalized_v2": false, "combo_kernels": true, "benchmark_combo_kernel": true}` |
| `inductor_passes` | `dict` | `{}` | 自定义 inductor passes |

### 其他

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `pass_config` | `dict` | `{}` | 自定义融合/转换 passes：`fuse_norm_quant`、`fuse_act_quant`、`fuse_attn_quant`、`enable_sp`、`fuse_gemm_comms`、`fuse_allreduce_rms` |
| `dynamic_shapes_config` | `dict` | `{"type": "backed", ...}` | 动态形状配置：`type`=动态形状类型；`evaluate_guards`=是否评估 guard；`assume_32_bit_indexing`=假设 32 位索引 |
| `use_inductor_graph_partition` | `bool` | `None` | 是否使用 inductor 图分区 |

### 便捷启动参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--cudagraph-capture-sizes` | `list[int]` | `None` | CUDA 图捕获的 batch size 列表（等价于 `-cc.cudagraph_capture_sizes`） |
| `--max-cudagraph-capture-size` | `int` | `None` | 最大 CUDA 图捕获大小（等价于 `-cc.max_cudagraph_capture_size`） |

---

## KernelConfig — 内核选择

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--moe-backend` | `str` | `auto` | MoE expert 计算内核后端。`auto`=自动选择；`triton`=Triton 融合 MoE 内核；`deep_gemm`=DeepGEMM（仅 FP8 块量化）；`cutlass`=vLLM CUTLASS 内核；`flashinfer_trtllm`=FlashInfer + TRTLLM-GEN；`flashinfer_cutlass`=FlashInfer + CUTLASS；`flashinfer_cutedsl`=FlashInfer + CuteDSL（仅 FP4）；`marlin`=Marlin（权重量化专用）；`aiter`=AMD AITer（ROCm 专用） |
| `--enable-flashinfer-autotune` | `bool` | `None` | 在内核预热时运行 FlashInfer 自动调优，找到最优 kernel 参数 |

---

## VllmConfig — 顶级配置容器

`VllmConfig` 是所有子配置的容器。vLLM 内部的统一配置对象。

### 性能模式

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--performance-mode` | `str` | `balanced` | 性能模式：<br>`balanced`=默认，延迟和吞吐的平衡<br>`interactivity`=低延迟优先（小 batch CUDA 图、延迟优化 kernel）<br>`throughput`=吞吐优先（大 CUDA 图、激进批处理、吞吐优化 kernel） |
| `--optimization-level` | `int` | `O2` | 优化级别（O0 最快启动，O3 最佳性能） |

### 投机解码

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--speculative-config` | `JSON` | `None` | 投机解码配置。需要一个较小的 draft model 快速生成候选 token，主模型并行验证。例：`{"model": "/path/to/draft-model", "num_speculative_tokens": 5, "method": "ngram"}` |

### 分析与调试

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--profiler-config` | `JSON` | 见下方 | Torch Profiler 配置。默认值包含：`profiler=None`（不启用）、`torch_profiler_dir=''`、`torch_profiler_with_stack=False`、`torch_profiler_with_flops=False`、`torch_profiler_use_gzip=True`、`torch_profiler_record_shapes=False`、`torch_profiler_with_memory=False`、`ignore_frontend=False`、`delay_iterations=0`、`max_iterations=0`、`warmup_iterations=0`、`active_iterations=5`、`wait_iterations=0` |

### 结构化输出

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--structured-outputs-config` | `JSON` | 见下方 | 结构化输出配置。默认值：`{"backend": "auto", "disable_any_whitespace": false, "disable_additional_properties": false, "reasoning_parser": "", "reasoning_parser_plugin": "", "enable_in_reasoning": false}` |

### 注意力详细配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--attention-config / -ac` | `JSON` | 见 AttentionConfig | 注意力配置 JSON。子字段见 AttentionConfig 节 |

### 编译配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--compilation-config / -cc` | `JSON` | 见 CompilationConfig | 编译配置 JSON。支持简写 `-cc.key=value`。子字段见 CompilationConfig 节 |

### 分布式 KV/权重传输

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--kv-transfer-config` | `JSON` | `None` | 分布式 KV Cache 传输配置（disaggregated prefill 场景） |
| `--ec-transfer-config` | `JSON` | `None` | 分布式 EC（Expert Cache）传输配置 |
| `--kv-events-config` | `JSON` | `None` | KV 事件发布配置 |
| `--weight-transfer-config` | `JSON` | `None` | RL 训练中的权重传输配置 |

---

## JSON 参数语法

vLLM 的 JSON 参数支持两种等价的 CLI 写法：

```bash
# 方式 1: JSON 字符串
--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'

# 方式 2: 点号展开（等价）
--json-arg.key1 value1 --json-arg.key2.key3 value2

# 列表元素用 + 追加
--json-arg '{"key4": ["value3", "value4", "value5"]}'
--json-arg.key4+ value3 --json-arg.key4+='value4,value5'
```

常见场景：
```bash
# 自定义 CUDA 图
-cc.cudagraph_capture_sizes+ 1 -cc.cudagraph_capture_sizes+='2,4,8'

# 覆盖生成配置
--override-generation-config.temperature 0.6 --override-generation-config.max_new_tokens 1024

# 投机解码
--speculative-config '{"model":"/path/to/draft","num_speculative_tokens":5}'
```

---

## 常用场景速查表

| 场景 | 关键参数 |
|---|---|
| **基础单卡** | `--max-model-len 2048 --gpu-memory-utilization 0.9 --max-num-seqs 32` |
| **24GB 单卡 7B** | `--max-model-len 8192 --gpu-memory-utilization 0.9` |
| **多卡大模型** | `--tensor-parallel-size N` |
| **超长上下文** | `--prefill-context-parallel-size 2 --decode-context-parallel-size 2` |
| **量化模型** | `--quantization awq` / `--quantization fp8` |
| **思考模型** | `--enable-reasoning --reasoning-parser deepseek_r1` |
| **LoRA 切换** | `--enable-lora --max-loras 4 --max-lora-rank 64` |
| **生产部署** | `--api-key sk-xxx --disable-fastapi-docs` |
| **多节点** | `--nnodes 2 --node-rank 0 --distributed-executor-backend ray` |
| **投机解码** | `--speculative-config '{"model":"draft-path","num_speculative_tokens":5}'` |
| **CPU 卸载** | `--cpu-offload-gb 8` |
| **KV Cache 减半** | `--kv-cache-dtype fp8` |
| **服务器监控** | `--otlp-traces-endpoint http://jaeger:4317` |
| **HTTPS** | `--ssl-certfile cert.pem --ssl-keyfile key.pem` |
| **YAML 配置** | `--config config.yaml` |

# 远程文件传输


### 1. rsync

- 功能： 用于同步和传输文件和目录

- 优势：
  * 增量传输： 只复制修改过的文件部分，节省带宽和时间。
  * 支持压缩： 可以使用`-z`选项进行数据压缩。
  * 保留属性: 可以保留文件的权限，时间戳，符号链接等
  * 可显示进度： 使用`--process`显示传输进度。

常见使用命令：
```bash
rsync -av --partial --process source/ user@remote:/path/to/desination/
```

`--partial`选项，会在中断时保留未完成的文件片段

使用`-e`选项来指定SSH连接和端口, 示例如下：
```bash
rsync -av -e "ssh -p 端口号" source_file user@remote:/path/to/destination/
```

### 2. scp
- 功能： 用于安全地复制文件到远程主机
- 优势：
  * 简单易用： 命令较为简单，适合快速传输文件
  * 安全性： 使用SSH加密传输，确保数据安全
- 限制：
  * 不支持增量传输：每次都需要完整传输文件

常用命令：
```bash
scp -P port source_file user@remote:/path/to/desination/ 
```
加入参数`-c aes128-ctr`表示使用AES 128位CTR模式来加密传输数据，可以稍微提高传输速度，相对较轻量级，特别是在处理大量小文件时。
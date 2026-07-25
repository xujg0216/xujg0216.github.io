# tar/zip

## tar

### 打包（压缩）

```bash
tar -czf archive.tar.gz /path/to/source_directory
```

参数说明：

- `-c` 创建归档
- `-z` 使用 gzip 压缩
- `-f` 指定文件名
- `-v` 显示详细过程（可选）

### 解包（解压）

```bash
tar -xzf archive.tar.gz              # 解压到当前目录
tar -xzf archive.tar.gz -C /target   # 解压到指定目录
```

- `-x` 解压
- `-C` 指定目标目录

### 排除文件

忽略特定文件或文件夹：

```bash
tar --exclude='./exclude_folder' --exclude='./exclude_file.txt' -czf archive.tar.gz -C /path/to/source_directory .
```

忽略规则较多时，使用排除列表文件：

```txt
# exclude.txt
*.log
temp/
path/to/ignore/
```

```bash
tar --exclude-from=exclude.txt -czf archive.tar.gz -C /path/to/source_directory .
```

## zip

### 打包

```bash
zip -r archive.zip mydir                           # 递归打包目录
zip -r archive.zip mydir -x '*.tmp' -x 'backup/*'  # 排除特定文件
zip -e archive.zip file.txt                        # 加密
zip -u archive.zip updatedfile.txt                 # 更新已存在的文件
zip -d archive.zip oldfile.txt                     # 从压缩包中删除文件
```

### 解压

```bash
unzip archive.zip                    # 解压到当前目录
unzip archive.zip -d /target/dir     # 解压到指定目录
unzip -l archive.zip                 # 仅列出内容，不解压
```

### 使用排除列表

```bash
zip -r archive.zip mydir $(cat exclude-list.txt | sed 's/^/-x /')
```

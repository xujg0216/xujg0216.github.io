
# Git LFS

Git LFS（Large File Storage）是Git的扩展，专门用于处理大文件（如图像、视频、大型数据集等）。使用Git LFS，可以避免将大文件直接存储在Git仓库中，而是将其存储在外部存储系统中，仅将其指针存储在Git仓库中，从而提高性能和效率。

以下是使用Git LFS的详细步骤：


### 1. 安装Git LFS
```bash
sudo apt-get update
sudo apt-get install git-lfs
```

### 2. 初始化Git LFS

```bash
git lfs install
```


这将会在系统上启动Git LFS, 并修改git 配置，以便在新建仓库中使用git lfs


### 3. 跟踪大文件

使用 `git lfs track` 命令来指定需要使用`Git LFS`存储的文件类型或具体文件。例如，如果你想要跟踪所有的PNG文件，可以运行：

```bash
git lfs track "*.png"
```
这将会创建或更新一个名为 .gitattributes 的文件，告诉Git LFS跟踪所有符合该模式的文件。


### 4. 添加并提交文件
添加想要跟踪的大文件到仓库，并像平时一样提交：
```bash
git add .
git commit -m "Add large files with Git LFS"
```
### 5. 推送到远程仓库

像平时一样将更改推送到远程仓库：

```bash
git push origin main
```

### 6. 克隆和拉取包含LFS文件的仓库
当你或其他人克隆包含LFS文件的仓库时，Git LFS会自动下载大文件：
```bash
git clone <repository-url>
```
如果仓库已经被克隆，可以运行以下命令来拉取并下载LFS文件：
```bash
git lfs pull
```

### 7. 管理LFS文件
**查看LFS文件状态**：可以使用以下命令查看哪些文件被Git LFS跟踪：
```bash
git lfs ls-files
```

**取消跟踪文件**：如果你不再需要跟踪某些文件，可以使用以下命令：
```bash
git lfs untrack "*.png"
```
这将从 `.gitattributes` 文件中移除对这些文件的跟踪。
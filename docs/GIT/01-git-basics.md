# Git 基础

## 1. Git 基础概念

1. **Git 是什么？**
    
    - Git 是一个开源的分布式版本控制系统，它能够帮助开发者跟踪文件的变化，协同工作以及管理多个版本的代码。
2. **常见的术语**
    
    - **Repository (仓库)**：保存项目文件及版本历史记录的地方。
    - **Commit**：记录对文件的修改。
    - **Branch (分支)**：不同的工作线，可以在不同的分支上并行开发。
    - **Merge (合并)**：将两个分支的代码合并到一起。
    - **Clone (克隆)**：将远程仓库的内容复制到本地。
    - **Push**：将本地的更改推送到远程仓库。
    - **Pull**：从远程仓库拉取最新的更改。

## 2. 安装 Git

1. **在 Linux 上安装 Git**
    
    ```bash
    sudo apt update
    sudo apt install git
    ```
    
2. **在 Windows 上安装 Git**
    
    - 访问 Git 官网：[https://git-scm.com/download/win](https://git-scm.com/download/win)，下载并安装。
3. **在 macOS 上安装 Git**
    
    ```bash
    brew install git
    ```
    

## 3. 配置 Git

1. **设置用户名和邮箱** 配置 Git 的用户名和邮箱是必需的，这些信息会显示在你的每个提交记录中。
    
    ```bash
    git config --global user.name "Your Name"
    git config --global user.email "youremail@example.com"
    ```
    
2. **查看配置**
    
    ```bash
    git config --list
    ```
    
3. **更改配置** 可以通过 `--global` 参数设置全局配置，也可以在项目中使用本地配置：
    
    ```bash
    git config user.name "New Name"
    git config user.email "newemail@example.com"
    ```
    
4. **设置别名** 你可以为常用的 Git 命令创建别名，简化操作：
    
    ```bash
    git config --global alias.co checkout
    git config --global alias.br branch
    git config --global alias.ci commit
    ```
    
5. **忽略文件** 使用 `.gitignore` 文件来忽略某些不需要版本控制的文件：
    
    ```bash
    echo "*.log" >> .gitignore
    ```

## 4. 创建和管理仓库

1. **初始化本地仓库**
    
    ```bash
    git init
    ```
    
2. **克隆远程仓库**
    
    ```bash
    git clone https://github.com/username/repository.git
    ```
    
3. **查看仓库状态**
    
    ```bash
    git status
    ```

## 5. 基本操作

1. **添加文件到暂存区**
    
    ```bash
    git add <file_name>    # 添加单个文件
    git add .              # 添加当前目录下所有更改过的文件
    ```
    
2. **提交更改**
    
    ```bash
    git commit -m "Your commit message"
    ```
    
3. **查看提交历史**
    
    ```bash
    git log
    ```
    
4. **查看文件的修改**
    
    ```bash
    git diff
    ```
    
5. **撤销更改**
    
    ```bash
    git restore                    # 恢复文件到最近一次提交的状态
    git reset                      # 取消暂存区所有修改，工作区不变
    git reset --soft HEAD~1        # 撤销最近一次提交，保留修改在暂存区和工作区
    git reset --mixed HEAD~1       # 撤销最近一次提交，丢弃暂存区修改，保留工作区
    git reset --hard HEAD~1        # 完全撤销最近一次提交，丢弃暂存区和工作区的修改
    ```

6. **撤销已推送的提交**

    ```bash
    git reset --hard <commit_hash>
    git push origin <branch_name> --force
    ```
    强制推送（`--force`）会覆盖远程仓库的历史，谨慎使用。

7. **删除文件**
    
    ```bash
    git rm <file_name>
    git commit -m "Remove file"
    ```

8. **合并 commit**

    合并到上一个 commit：
    ```bash
    git add .
    git commit --amend
    ```

    合并最近的多个 commit：
    ```bash
    git rebase -i HEAD~2
    git rebase -i HEAD~n
    ```
    操作步骤：
    - 执行命令后会打开编辑器
    - 将后续 commit 前的 `pick` 改为 `squash` 或 `s`
    - 保存并关闭编辑器
    - 在新的编辑器中编辑合并后的 commit 信息
    - 保存完成合并

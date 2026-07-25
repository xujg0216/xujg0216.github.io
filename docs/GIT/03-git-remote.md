# 远程仓库

## 1. 远程仓库操作

1. **查看远程仓库**
    
    ```bash
    git remote -v
    ```
    
2. **添加远程仓库**
    
    ```bash
    git remote add origin <remote_repository_url>
    ```
    
3. **推送更改**
    
    ```bash
    git push origin <branch_name>
    ```
    
4. **拉取远程更改**
    
    ```bash
    git pull origin <branch_name>
    ```
    
5. **获取最新更改但不合并**
    
    ```bash
    git fetch origin
    ```

6. **删除远程仓库**
    
    ```bash
    git remote remove origin
    ```

## 2. 工作流

### 常见工作流

- **集中式工作流**：所有开发者将代码推送到一个公共仓库，每个人在自己的本地机器上工作。
- **功能分支工作流**：每个新功能都在一个独立的分支上开发，开发完成后合并回主分支。
- **Gitflow 工作流**：将开发分成多个阶段，如 `feature`、`develop`、`release` 和 `hotfix` 等。

### Pull Request 流程

1. 创建一个新的分支进行开发。
2. 完成开发后，提交更改并推送到远程分支。
3. 在 GitHub 等平台上创建 Pull Request（PR）。
4. 进行代码审查、测试等，最终合并到主分支。

## 3. Git Tag

创建标签来标记重要的提交（如发布版本）：

```bash
git tag -a v1.0 -m "Release version 1.0"
```

推送 tag 到远程：

```bash
git push origin v1.0         # 推送单个 tag
git push origin --tags       # 推送所有本地 tag
```

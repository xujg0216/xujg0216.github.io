# 分支管理

## 1. 查看与创建分支

1. **查看现有分支**
    
    ```bash
    git branch
    ```
    
2. **创建并切换分支**
    
    ```bash
    git branch <branch_name>       # 创建分支
    git checkout <branch_name>     # 切换分支
    git checkout -b <branch_name>  # 创建并切换
    ```

3. **基于远程分支创建本地分支**

    ```bash
    git checkout -b <branch_name> origin/<branch_name>
    ```

## 2. 合并分支

```bash
git merge <branch_name>
```

### Fast-forward 合并

当目标分支没有新提交时，Git 会把当前分支指针直接移到目标分支的最新提交，不会产生额外的 merge commit：

```
合并前:
  A---B  main
       \
        C---D  feature

合并后 (git merge feature):
  A---B---C---D  main, feature
```

### 三方合并

当两个分支都有新提交时，Git 会创建一个新的 merge commit 来合并双方：

```
合并前:
  A---B---E  main
       \
        C---D  feature

合并后:
  A---B---E---F  main
       \     /
        C---D  feature
```

### 常用选项

```bash
git merge --no-ff <branch_name>   # 禁用 fast-forward，强制生成 merge commit
git merge --squash <branch_name>  # 将目标分支所有提交压缩成一个，不产生 merge commit
git merge --abort                 # 合并冲突时放弃本次合并
```

### 合并冲突

当两个分支修改了同一文件的同一行时，Git 会产生冲突。冲突文件中会标记出双方差异：

```
<<<<<<< HEAD
当前分支的内容
=======
被合并分支的内容
>>>>>>> <branch_name>
```

解决冲突后：

```bash
git add <file_name>              # 标记冲突已解决
git commit                       # 完成合并（不需要 -m，Git 会自动生成 merge commit 信息）
```

## 3. 删除分支

```bash
git branch -d <branch_name>                   # 删除本地分支
git push origin --delete <branch_name>         # 删除远程分支
```

## 4. 清理远程引用缓存

```bash
git remote prune origin
```

## 5. Git Stash

将当前的更改暂时保存起来，便于切换分支等操作：

```bash
git stash                    # 暂存当前修改
git stash list               # 查看 stash 列表
git stash pop                # 恢复最近的 stash 并删除
git stash apply              # 恢复最近的 stash 但保留
git stash drop               # 删除最近的 stash
```

## 6. Git Rebase

将一个分支的更改合并到另一个分支，通常用于保持提交历史的整洁：

```bash
git rebase <branch_name>
```

## 7. Git Cherry-pick

选择性地将某个（或某些）提交应用到当前分支：

```bash
git cherry-pick <commit_hash>                        # 挑取单个提交
git cherry-pick <hash1> <hash2>                      # 挑取多个不连续的提交
git cherry-pick <hash_a>..<hash_b>                   # 挑取连续范围的提交（不含 hash_a）
git cherry-pick <hash_a>^..<hash_b>                  # 挑取连续范围的提交（含 hash_a）
```

**使用场景**：你从 A 分支开发了一个通用功能，现在只想把这个功能合入 B 分支，而不同步 A 分支的其他改动。

### 处理冲突

cherry-pick 过程中可能遇到冲突，处理流程和 merge 类似：

```bash
# 手动修改冲突文件
git add <file_name>                 # 标记已解决
git cherry-pick --continue          # 继续应用该提交

git cherry-pick --abort             # 放弃本次 cherry-pick，恢复到操作前状态
git cherry-pick --skip              # 跳过当前这个提交，继续处理后续（批量挑取时）
```

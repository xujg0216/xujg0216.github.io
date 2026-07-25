# apt

在 Linux 中，`apt`（Advanced Package Tool）是常用的包管理工具。以下按功能分组列出常用命令。

## 1. 更新与升级

```bash
sudo apt update              # 更新包列表，不会安装或更新任何软件包
sudo apt upgrade             # 升级所有已安装包（不删除/新增包）
sudo apt full-upgrade        # 全面升级，会处理依赖关系，可能删除旧包
apt list --upgradable        # 查看可升级的包
```

## 2. 安装与删除

```bash
sudo apt install <pkg>       # 安装指定软件包
sudo apt remove <pkg>        # 删除软件包，保留配置文件
sudo apt purge <pkg>         # 完全删除，包括配置文件
```

## 3. 搜索与查看

```bash
apt search <keyword>         # 按关键字搜索包
apt show <pkg>               # 查看包的详细信息
apt list --installed         # 列出已安装的包
```

## 4. 清理

```bash
sudo apt autoremove          # 删除不再需要的依赖包
sudo apt clean               # 清理本地缓存的 deb 包文件
```

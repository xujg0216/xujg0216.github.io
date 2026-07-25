# Vscode远程免密连接


>在vscode远程连接，使用SSH公钥登录实现免密连接 
>👉 本机生成 SSH key → 把公钥拷到远端 → VS Code 用 key 登录

## 1. 本机生成SSH Key

在本机上执行
```bash
ssh-keygen -t ed25519 -C "vscode-remote"
```
一路回车即可：

- 保存路径：直接回车（默认 ~/.ssh/id_ed25519）
    
- 是否设置 passphrase：想完全无感登录就直接回车
    
  
生成后会有：

~/.ssh/id_ed25519        （私钥）
~/.ssh/id_ed25519.pub    （公钥）


## 2. 把公钥拷贝到远程设备

```bash
cat ~/.ssh/id_ed25519.pub
```

复制输出内容，然后在 **远端** 执行：
```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
vim ~/.ssh/authorized_keys
```
粘贴进去后：
```bash
chmod 600 ~/.ssh/authorized_keys
```

## 3. 配置Vscode Remote-SSH使用Key

打开配置文件
```
 Ctrl + Shift + P →
 Remote-SSH: Open SSH Configuration File
```
  
选择：
```
~/.ssh/config
```
写成这样：
```
Host remote_name
    HostName 192.168.1.100
    User root
    IdentityFile ~/.ssh/id_ed25519
```
# 网络代理

!!! Question
    在Ubuntu系统中，设置网络代理后，通常只有那些配置为使用系统代理的应用程序会通过代理连接网络。对于一些应用程序，比如终端，不会自动遵循全局代理设置，因此，需要为这些应用程序手动配置代理。

## 1. 系统级代理（环境变量）

临时生效（当前终端）：

```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export no_proxy=localhost,127.0.0.1
```

永久生效，写入 `~/.bashrc`：

```bash
echo 'export http_proxy=http://127.0.0.1:7897' >> ~/.bashrc
echo 'export https_proxy=http://127.0.0.1:7897' >> ~/.bashrc
source ~/.bashrc
```

## 2. apt 代理

编辑 `/etc/apt/apt.conf.d/proxy.conf`：

```
Acquire::http::Proxy "http://127.0.0.1:7897";
Acquire::https::Proxy "http://127.0.0.1:7897";
```

## 3. Git 代理

查看当前代理：

```bash
git config --get http.proxy
```

添加代理：

```bash
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
```

取消代理：

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```

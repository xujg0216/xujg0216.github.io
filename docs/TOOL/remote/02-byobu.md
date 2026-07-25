
# byobu
!!! note
    ssh连接linux服务器中断后，如何让命令继续在服务器运行

>使用`byobu`
`byobu` 是一个增强的终端复用器，它基于 `tmux` 和 `screen`，提供了更友好的界面和更多的功能。使用 `byobu` 可以方便地在关闭 `VSCode` 或终端后保持程序运行。以下是使用 `byobu` 的步骤：

### 安装byobu
在远程服务器安装`byobu`
```bash
sudo apt-get update
sudo apt-get install byobu
```
### 启动 byobu
```bash
byobu
```
这将启动一个新的 `byobu `会话。如果是第一次运行，`byobu` 可能会提示配置键绑定选项。

在 `byobu` 会话中运行程序：
例如，运行一个 Python 脚本：
```bash
python my_script.py
```

### 分离 byobu
分离`（detach）byobu` 会话：
按 `Ctrl + A`，然后按 `D` 键。这将分离当前会话，但不会关闭它。

### 断开 SSH 连接
可以直接关闭终端窗口或运行以下命令：
```bash
exit
```

### 重新连接byobu会话

列出所有会话
```bash
byobu list-sessions
```
使用 `byobu attach-session` 命令重新连接到指定会话：
```bash
byobu attach-session -t <session-name>
```
或者，直接连接最近的会话：
```bash
byobu attach
```
### 管理byobu会话

##### 创建一个新的会话
```bash
byobu new-session -s my_new_session
```
##### 列出所有会话
```bash
byobu list-sessions
```
##### 切换到特定会话

```bash
byobu attach-session -t <session-name>
```

##### 结束会话

连接到想结束的会话
```bash
byobu attach-session -t <session-name>
```
按 `Ctrl + D` 或运行 `exit` 命令结束会话。




### 基本快捷键
`F2`：创建新窗口。  
`F3`：切换到前一个窗口。  
`F4`：切换到下一个窗口。  
`F5`：刷新当前窗口。  
`F6`：分离当前会话。  
`F7`：进入滚动模式，可以滚动查看窗口的历史输出。  
`F8`：重命名当前窗口。  
`F9`：打开 byobu 配置菜单。  
`F10`：关闭当前窗口。  
### 高级快捷键
`Shift + F2`：拆分窗口（水平）。  
`Ctrl + F2`：拆分窗口（垂直）。  
`Shift + F3`：移动到前一个拆分窗口。  
`Shift + F4`：移动到下一个拆分窗口。  
`Ctrl + A，Shift + O`：将当前窗口放入/移出拆分布局。  
`Ctrl + A，Ctrl + F`：全屏切换当前拆分窗口。  
### 会话管理快捷键
`Ctrl + A`，`D`：分离会话。  
`Ctrl + A`，`C`：创建新窗口（与 F2 类似）。  
`Ctrl + A`，`N`：切换到下一个窗口。  
`Ctrl + A`，`P`：切换到前一个窗口。  
`Ctrl + A`，`L`：重新绘制屏幕。  
### 快捷键参考
`F9` 打开配置菜单：
通过按 `F9` 键，可以访问 `byobu` 的配置菜单，在这里可以查看和更改所有快捷键的设置。
在配置菜单中，你还可以配置状态栏、颜色方案、键绑定等选项。

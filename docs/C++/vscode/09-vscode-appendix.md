
# 九 附录
vscode通过tasks.json文件进行编译生成可执行文件， 因此实现的功能就是[[01-g++]]， [[02-compile]]中编译的步骤
>通过终端-> 配置任务 -> 创建tasks.json

下面分为两种情况： 
* [使用g++编译可执行文件](./01-g++.md)
* [使用cmake编译可执行文件](./06-cmake-compile-step.md)

### 使用g++编译可执行文件
```json
// tasks.json
{
    // https://code.visualstudio.com/docs/editor/tasks
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build",  // 任务的名字叫Build，注意是大小写区分的，等会在launch中调用这个名字
            "type": "shell",  // 任务执行的是shell命令，也可以是
            "command": "g++", // 命令是g++
            "args": [
                "'-Wall'",
                "'-std=c++17'",  //使用c++17标准编译
                "'${file}'", //当前文件名
                "-o", //对象名，不进行编译优化
                "'${fileBasenameNoExtension}.out'",  //当前文件名（去掉扩展名）
            ],
          // 所以以上部分，就是在shell中执行（假设文件名为filename.cpp）
          // g++ filename.cpp -o filename.exe
            "group": { 
                "kind": "build",
                "isDefault": true   
                // 任务分组，因为是tasks而不是task，意味着可以连着执行很多任务
                // 在build组的任务们，可以通过在Command Palette(F1) 输入run build task来运行
                // 当然，如果任务分组是test，你就可以用run test task来运行 
            },
            "problemMatcher": [
                "$gcc" // 使用gcc捕获错误
            ],
        }
    ]
}
```
上面tasks.json相当于命令`g++ -Wall -std=c++17 file.cpp -o a.out`
可以通过vscode中`终端-> 运行生成任务`或者`ctrl + shift + B`来生成可执行文件



### 使用cmake编译可执行文件
也是通过配置tasks.json文件进行编译， 与g++不同， 需要设置成cmake编译可执行文件的步骤。


```shell
# 在项目源代码根目录下，使用  cmake 指令解析 CMakeLists.txt ，生成 Makefile 和其他文件
cmake .
# 执行 make 命令，生成 target
make
```


```json
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "options": {
        "cwd": "${workspaceFolder}/build"
    },
    "tasks": [
        {
            "type": "shell",
            "label": "cmake",
            "command": "cmake",
            "args": [
                ".."
            ]
        },
        {
            "label": "make",
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "command": "make",
            "args": []
        },
        {
            "label": "build",
            "dependsOrder": "sequence",
            "dependsOn": [
                "cmake",
                "make"
            ]
        }
    ],
}
```

### 利用launch.json文件进行调试

无论是g++或者cmake， 通过建立对应的tasks.json文件生成的可执行文件， 都采用gdb进行调试， 下面是具体的launch.json中的内容
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "C++ Launch",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/yolo", // 可执行文件的路径
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": true,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "build",
            "miDebuggerPath": "/usr/bin/gdb" // gdb 调试器的路径
        }
    ]
}

```


#### 预定义变量

支持下面的预定义变量:

- **${workspaceFolder}** - 当前工作目录(根目录)
- **${workspaceFolderBasename}** - 当前文件的父目录
- **${file}** - 当前打开的文件名(完整路径)
- **${relativeFile}** - 当前根目录到当前打开文件的相对路径(包括文件名)
- **${relativeFileDirname}** - 当前根目录到当前打开文件的相对路径(不包括文件名)
- **${fileBasename}** - 当前打开的文件名(包括扩展名)
- **${fileBasenameNoExtension}** - 当前打开的文件名(不包括扩展名)
- **${fileDirname}** - 当前打开文件的目录
- **${fileExtname}** - 当前打开文件的扩展名
- **${cwd}** - 启动时task工作的目录
- **${lineNumber}** - 当前激活文件所选行
- **${selectedText}** - 当前激活文件中所选择的文本
- **${execPath}** - vscode执行文件所在的目录
- **${defaultBuildTask}** - 默认编译任务(build task)的名字

#### 预定义变量示例:

假设你满足以下的条件

1. 一个文件 `/home/your-username/your-project/folder/file.ext` 在你的编辑器中打开;
2. 一个目录 `/home/your-username/your-project` 作为你的根目录.

下面的预定义变量则代表:

- **${workspaceFolder}** - `/home/your-username/your-project`
- **${workspaceFolderBasename}** - `your-project`
- **${file}** - `/home/your-username/your-project/folder/file.ext`
- **${relativeFile}** - `folder/file.ext`
- **${relativeFileDirname}** - `folder`
- **${fileBasename}** - `file.ext`
- **${fileBasenameNoExtension}** - `file`
- **${fileDirname}** - `/home/your-username/your-project/folder`
- **${fileExtname}** - `.ext`
- **${lineNumber}** - 光标所在行
- **${selectedText}** - 编辑器中所选择的文本
- **${execPath}** - Code.exe的位置

> **Tip**: vscode的智能提示会在`tasks.json`和`launch.json` 提示所有支持的预定义变量.
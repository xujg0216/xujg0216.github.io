
# HDF5

在C语言中读取HDF5文件可以使用HDF5库，这是一个专门用于管理和存储科学数据的开源软件库

### 1.前置步骤
安装HDF5库: 如果你还没有安装HDF5库，可以通过包管理器安装。例如，在Ubuntu上可以使用以下命令：

```bash
sudo apt-get install libhdf5-dev
```

### 2.包含头文件和链接库

在编译时需要包含HDF5库的头文件并链接相应的库。

安装完成后，头文件通常位于`/usr/include/hdf5/serial`/或`/usr/local/include/`目录下。编译时需要指定这些路径。

```bash
gcc -o load_database load_database.c -I/usr/include/hdf5/serial -L/usr/lib/aarch64-linux-gnu/hdf5/serial -lhdf5
```
其中：
* `-I` 选项指定头文件的路径。
* `-L` 选项指定库文件的路径。
* `-lhdf5` 链接HDF5库。

!!! note
    要在VS Code中优先使用标准的HDF5库头文件提供的hdf5.h，可以通过以下方法来设置正确的头文件搜索路径， 
    在`.vscode/c_cpp_properties.json`添加标准HDF5库的头文件路径("/usr/include/hdf5/serial")到`includePath`列表


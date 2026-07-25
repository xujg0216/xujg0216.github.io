# docker基本命令

参考[https://docs.docker.com/reference/cli/docker/](https://docs.docker.com/reference/cli/docker/)

## 1. docker简介
Docker 是一个应用打包、分发、部署的工具
- **打包**：把软件运行所需的依赖、第三方库、软件打包到一起，变成一个安装包
- **分发**：把打包好的"安装包"上传到一个镜像仓库，其他人可以非常方便的获取和安装
- **部署**：拿着"安装包"就可以一个命令运行起来你的应用，自动模拟出一摸一样的运行环境，不管是在 Windows/Mac/Linux确保了不同机器上都是一致的运行环境
- **镜像**：镜像包含运行应用程序所需的所有内容——代码或二进制文件、运行时、依赖项以及所需的任何其他文件系统对象。可以理解为软件安装包，可以方便的进行传播和安装。**镜像是一个程序安装包，用来生成容器**
- **容器**：容器只不过是一个正在运行的进程，它还应用了一些附加的封装功能，以使其与主机和其他容器保持隔离。容器隔离的最重要方面之一是每个容器都与自己的私有文件系统进行交互；该文件系统由Docker镜像提供。**容器是镜像运行的实例，它是一个可读写的，运行中的进程环境**

## 2. 获取镜像
获取镜像的方式有两种： 
* 使用他人打包好，并通过网络（主要是docker官方的docker hub和一些类似的镜像托管网站）进行分享的镜像；
* 在本地将镜像保存为本地文件，直接使用生成的文件进行共享。后者在网络受限环境下更加方便；

### 2.1 从网络
这里主要可以借助于两条指令。一个是 `docker pull` ，另一个是 `docker run `
```bash
docker pull [OPTIONS] NAME[:TAG|@DIGEST]
```
从仓库中拉取（pull）指定的镜像到本机。
对应于这里的三种构造指令的方式，文档中给出了几种不同的拉取方式：

* NAME
  * `docker pull ubuntu`
如果不指定标签，Docker Engine会使用 :latest 作为默认标签拉取镜像。
* NAME:TAG
  * `docker pull ubuntu:14.04`
使用标签时，可以再次 `docker pull` 这个镜像，以确保具有该镜像的最新版本。
例如，`docker pull ubuntu:14.04` 将会拉取`Ubuntu 14.04`镜像的最新版本。
* NAME@DIGEST
  * `docker pull ubuntu@sha256:45b23dee08af5e43a7fea6c4cf9c25ccf269ee113168c19722f87876677c5cb2`
这个为我们提供了一种指定特定版本镜像的方法
为了保证后期我们仅仅使用这个版本的镜像，我们可以重新通过指定DIGEST（通过查看镜像托管网站里的镜像信息或者是之前的 pull 输出里的DIGEST信息）的方式 pull 该版本镜像。


```bash
docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
```

`docker run` 命令首先在指定的映像上创建一个可写的容器层，然后使用指定的命令启动它

这个实际上会自动从官方仓库中下载本地没有的镜像。更多是使用会在后面创建容器的部分介绍。

```bash
$ sudo docker images
REPOSITORY                                             TAG                 IMAGE ID            CREATED             SIZE

$ sudo docker run hello-world
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
0e03bdcc26d7: Pull complete 
Digest: sha256:49a1c8800c94df04e9658809b006fd8a686cab8028d33cfba2cc049724254202
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/

$ sudo docker images
REPOSITORY                                             TAG                 IMAGE ID            CREATED             SIZE
hello-world                                            latest              bf756fb1ae65        7 months ago        13.3kB

```

### 2.2 从本地文件加载
关于如何保存镜像在后面介绍，其涉及到的指令为 `docker save` ，这里主要讲如何加载已经导出的镜像文件。
```bash
docker load [OPTIONS]
```
例子可见：[https://docs.docker.com/engine/reference/commandline/load/#examples](https://docs.docker.com/engine/reference/commandline/load/#examples)

## 3. 创建并启动一个新容器


`docker run` 的语法非常灵活，参数也很多，我们来 **分模块详细讲解每个参数的意义和用法**



```bash
docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
```

* `OPTIONS`: 运行参数（配置容器的运行方式）
* `IMAGE`: 要运行的镜像名（如 `ubuntu:20.04`）
* `COMMAND`: 镜像内要执行的命令（可选，如 `bash`）



### 3.1 启动模式

| 参数     | 说明             | 示例                           |
| ------ | -------------- | ---------------------------- |
| `-d`   | 后台运行（detached） | `docker run -d nginx`        |
| `-it`  | 交互模式 + 伪终端     | `docker run -it ubuntu bash` |
| `--rm` | 容器退出时自动删除      | `docker run --rm ubuntu`     |

📝 说明：

* `-i`: 保持输入流打开
* `-t`: 分配终端


### 3.2 容器命名

| 参数       | 说明    | 示例                              |
| -------- | ----- | ------------------------------- |
| `--name` | 指定容器名 | `docker run --name myweb nginx` |


### 3.3 端口映射

| 参数   | 说明            | 示例                            |
| ---- | ------------- | ----------------------------- |
| `-p` | 将宿主机端口映射到容器端口 | `docker run -p 8080:80 nginx` |

含义：本机的 `8080` 对应容器的 `80`，可通过 `localhost:8080` 访问。


### 3.4 卷挂载（数据持久化）

| 参数   | 说明          | 示例                                      |
| ---- | ----------- | --------------------------------------- |
| `-v` | 将宿主机目录挂载到容器 | `docker run -v /host/data:/data ubuntu` |

含义：容器内 `/data` 目录与宿主机 `/host/data` 同步。

Docker 中的挂载方式主要有两种：绑定挂载（bind mount） 和 命名卷挂载（named volume）。这两种方式都用于将数据从宿主机"挂载"到容器中，但它们的机制、用途和管理方式不同。

#### 3.4.1 绑定挂载（Bind Mount）


```bash
-v /宿主机/目录:/容器/目录
```

**demo**
```bash
docker run -v /home/user/html:/usr/share/nginx/html nginx
```

**特点**

| 项目   | 内容                          |
| ---- | --------------------------- |
| 来源   | 宿主机的实际路径                    |
| 持久性  | 宿主机目录一直存在，容器删除也不会删除数据       |
| 管理   | 由你手动管理（Docker 不负责）          |
| 可见性  | 宿主机和容器实时同步（适合本地开发）          |
| 权限控制 | 可设置只读：`-v /src:/dst:ro`     |
| 灵活性  | 可以挂载任意路径，包括 socket、日志、配置文件等 |

**使用场景**

* 本地开发（例如代码热更新）
* 指定配置文件路径
* 挂载日志、模型、视频文件等


#### 3.4.2 命名卷挂载（Named Volume）

**demo**

```bash
docker volume create myvolume
docker run -v myvolume:/usr/share/nginx/html nginx
```

**特点**：

| 项目   | 内容                                                   |
| ---- | ---------------------------------------------------- |
| 来源   | Docker 自动创建和管理的目录（通常在 `/var/lib/docker/volumes/...`） |
| 持久性  | 与容器生命周期无关（删除容器后仍然保留）                                 |
| 管理   | 通过 `docker volume` 命令集中管理                            |
| 可见性  | 默认宿主机看不到数据（除非进入 volume 路径）                           |
| 数据隔离 | 容器之间可以共享 Volume，但默认互不影响                              |
| 可移植性 | 可以用于备份/迁移（配合 `docker volume` 命令）                     |

**使用场景**

* 数据库存储（如 MySQL、PostgreSQL）
* 需要持久化数据的服务
* 容器间共享数据（可配合 `docker-compose`）


**对比总结表**

| 对比项     | 绑定挂载（Bind Mount） | 命名卷（Named Volume）                          |
| ------- | ---------------- | ------------------------------------------ |
| 数据来源    | 宿主机上任意目录         | Docker 管理的路径 `/var/lib/docker/volumes/...` |
| 创建方式    | 自动或手动创建宿主机路径     | `docker volume create` 创建                  |
| 持久性     | 与宿主机路径相关         | 容器删除也不会删除 volume                           |
| 数据共享    | 手动配置相同宿主机路径      | 多个容器可挂载同一个 volume                          |
| 容器删除后数据 | 仍保留              | 仍保留                                        |
| 管理方便性   | 不易统一管理           | `docker volume ls/inspect/rm` 管理           |
| 文件访问性   | 宿主机上可直接访问        | 宿主机默认不可见                                   |
| 推荐场景    | 本地开发、调试、挂载配置或日志等 | 数据持久化、数据库、共享缓存                             |

**命令参考**

- 创建命名卷

```bash
docker volume create mydata
```

- 使用命名卷

```bash
docker run -v mydata:/data ubuntu
```

- 查看卷

```bash
docker volume ls
docker volume inspect mydata
```

- 删除卷

```bash
docker volume rm mydata
```




### 3.5 环境变量

| 参数   | 说明     | 示例                              |
| ---- | ------ | ------------------------------- |
| `-e` | 设置环境变量 | `docker run -e ENV=prod ubuntu` |


### 3.6 网络配置

| 参数          | 说明        | 示例                                  |
| ----------- | --------- | ----------------------------------- |
| `--network` | 指定容器连接的网络 | `docker run --network mynet ubuntu` |

可以通过 `docker network create` 创建自定义网络。


### 3.7 自动重启策略（用于生产环境）

| 参数                 | 说明        | 示例                                  |
| ------------------ | --------- | ----------------------------------- |
| `--restart=always` | 容器崩溃后自动重启 | `docker run --restart=always nginx` |

可选值：

* `no`（默认）
* `on-failure`
* `always`
* `unless-stopped`


### 3.8 完整示例：部署 Nginx 网站容器

```bash
docker run -d \
  --name my-nginx \
  -p 8080:80 \
  -v /home/user/html:/usr/share/nginx/html \
  --restart=always \
  nginx
```

📝 含义：

* 后台运行
* 容器叫 `my-nginx`
* 把本地 8080 映射到容器 80 端口
* 将宿主机的 `/home/user/html` 作为网站根目录
* 自动重启
* 使用 nginx 镜像





其实 `docker run` 等价于：

```bash
docker create ...   # 创建容器（不运行）
docker start ...    # 启动容器
```



## 4. 容器相关命令


### 4.1 容器生命周期管理（创建、启动、停止等）

`docker run` 👉 **创建 + 启动容器**

```bash
docker run -it --name mycontainer ubuntu bash
```

* 创建容器并运行
* 加 `-d` 就是后台运行

 `docker start` 👉 启动 **已创建但未运行** 的容器

```bash
docker start mycontainer
```


 `docker stop` 👉 停止正在运行的容器

```bash
docker stop mycontainer
```

* 等应用正常关闭，最长 10 秒后强制停止


`docker restart` 👉 重启容器

```bash
docker restart mycontainer
```


`docker pause / unpause` 👉 挂起 / 恢复容器进程

```bash
docker pause mycontainer
docker unpause mycontainer
```


`docker kill` 👉 强制终止容器（立即发送 SIGKILL）

```bash
docker kill mycontainer
```


`docker rm` 👉 删除容器（必须先停止）

```bash
docker rm mycontainer
docker rm -f mycontainer  # 强制删除
```


### 4.2 容器信息查看

`docker ps` 👉 查看正在运行的容器

```bash
docker ps
```

`docker ps -a` 👉 查看所有容器（包括停止的）

```bash
docker ps -a
```

`docker inspect` 👉 查看容器详细信息（JSON）

```bash
docker inspect mycontainer
```

可查看：

* 容器配置
* 网络信息
* 挂载卷
* IP、环境变量等


`docker stats` 👉 实时查看容器资源占用（CPU、内存、IO）

```bash
docker stats
```


`docker top` 👉 查看容器内的进程

```bash
docker top mycontainer
```


### 4.3 容器交互与文件管理

`docker exec` 👉 在运行中的容器内执行命令

```bash
docker exec -it mycontainer bash
```

* 类似登录容器终端
* 用于运行调试命令、查看日志、临时修改


 `docker attach` 👉 连接容器主终端

```bash
docker attach mycontainer
```

📌 注意：

* 是连接主进程（非新终端）
* Ctrl+C 会停止容器


`docker cp` 👉 容器与宿主机之间拷贝文件

**从宿主机拷贝到容器**

```bash
docker cp ./config.json mycontainer:/app/config.json
```

**从容器拷贝到宿主机**

```bash
docker cp mycontainer:/app/log.txt ./log.txt
```


`docker logs` 👉 查看容器的标准输出日志

```bash
docker logs mycontainer
docker logs -f mycontainer  # 实时查看（follow）
```


### 4.4 容器更新与修改

`docker commit` 👉 把运行中的容器保存成镜像

```bash
docker commit mycontainer mynewimage:v1
```


### 4.5 容器清理

删除所有停止的容器

```bash
docker container prune
```

 删除所有容器（危险操作）

```bash
docker rm -f $(docker ps -aq)
```

---

### 4.6 命令速查表

| 类型    | 命令                       | 说明              |
| ----- | ------------------------ | --------------- |
| 创建并运行 | `docker run`             | 创建并启动一个新容器      |
| 启动    | `docker start`           | 启动已存在容器         |
| 停止    | `docker stop`            | 停止运行的容器         |
| 强制停止  | `docker kill`            | 立即中止容器进程        |
| 重启    | `docker restart`         | 停止+启动           |
| 删除    | `docker rm`              | 删除容器            |
| 查看    | `docker ps`              | 查看运行中容器         |
| 查看全部  | `docker ps -a`           | 包括停止的容器         |
| 查看详情  | `docker inspect`         | 容器配置与状态         |
| 查看资源  | `docker stats`           | CPU、内存、I/O 实时监控 |
| 查看进程  | `docker top`             | 容器内运行的进程        |
| 登录容器  | `docker exec`            | 推荐（不会关闭主进程）     |
| 附加终端  | `docker attach`          | 不推荐（Ctrl+C 会退出） |
| 查看日志  | `docker logs`            | 标准输出内容          |
| 文件复制  | `docker cp`              | 宿主机和容器间互拷       |
| 生成镜像  | `docker commit`          | 把容器打包成镜像        |
| 清理容器  | `docker container prune` | 删除全部已停止的容器      |

---


## 5. 使用 Dockerfile 构建镜像

### 5.1 准备一个 Dockerfile

```dockerfile
FROM ubuntu:20.04
RUN apt update && apt install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "main.py"]
```

### 5.2 构建镜像

```bash
docker build -t my-python-app:v1 .
```

参数说明：

* `-t my-python-app:v1`：给镜像命名并打标签
* `.`：表示当前目录（Dockerfile 所在目录）

### 5.3 查看生成的镜像

```bash
docker images
```

输出示例：

```
REPOSITORY        TAG     IMAGE ID       CREATED          SIZE
my-python-app     v1      abcd1234...    10 seconds ago   180MB
```

### 5.4 使用镜像运行容器

```bash
docker run -it my-python-app:v1
```

## 6. 导出导入镜像

### docker save —— 导出镜像

将一个或多个镜像导出为 `.tar` 文件，包含所有层和元数据：

```bash
docker save -o nginx.tar nginx:latest                     # 导出单个镜像
docker save -o all-in-one.tar nginx:latest redis:6.0      # 导出多个镜像
```

### docker load —— 导入镜像

从 `.tar` 文件导入镜像：

```bash
docker load -i nginx.tar
```

导入后即可像本地镜像一样使用。

### 完整示例：迁移镜像

源服务器：

```bash
docker save -o myapp.tar myapp:1.0
scp myapp.tar user@target-server:/home/user/
```

目标服务器：

```bash
docker load -i /home/user/myapp.tar
docker run myapp:1.0
```

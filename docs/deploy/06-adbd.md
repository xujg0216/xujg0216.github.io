
使用`adb`连接`android`, `linux`需要在被控制设备上安装`adbd`， `android`默认安装`adbd`， 而ubuntu下默认是没有adbd服务，由于需要android下的一些配置环境导致`apt`也无法进行安装，那如何让它本身被识别成一个`adb`设备，用来调试，下面进行介绍。


#### 下载adbd二进制文件
`链接`: https://mail.bjtu.edu.cn/coremail/common/nfFile.jsp?share_link=7BF7C9AC92E94CE8993D7E7729AF4455&uid=20120017%40bjtu.edu.cn
`密码`: `yq6E`

#### 解压文件
执行以下命令：
```shell
sudo cp ./adbd.sh /etc/init.d/
sudo cp ./.usb_config /etc/init.d/

sudo cp adbd /usr/local/bin/
sudo cp adbd.service /lib/systemd/system/

```
#### 创建开机启动服务链接

在`/etc/systemd/system/multi-user.target.wants` 目录创建软链接
```shell
ln -s /lib/systemd/system/adbd.service .
```

#### 启动服务

```shell
systemctl start adbd.service
```

#### 验证服务是否启动成功

```shell
systemctl status adbd.service
```

#### 开机自启动

```shell
systemctl enable adbd.service
```


#### 关闭服务

```shell
systemctl stop adbd.service
```

#### 卸载服务

```shell
sudo rm /etc/init.d/adbd  
sudo rm /etc/init.d/.usb_config
```
以上就是在`ubuntu`下增加`adbd`服务的步骤


下面在上位机端通过`adb`进行连接

#### 启用adb, 创建ADB服务器
```shell
adb start_server
```
#### 测试ADB连接
```shell
adb devices
```

#### 连接下位机

```shell
adb connect <server_ip>:5555
```


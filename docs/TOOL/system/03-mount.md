# 挂载硬盘

## 1. 检查当前挂载

查看已挂载的文件系统：

```bash
df -h
```

## 2. 查看所有分区

使用 `lsblk` 查看所有硬盘及分区，找出未挂载的分区：

```bash
lsblk -f
```

如果某个分区没有 MOUNTPOINT，说明未挂载。例如：

```text
NAME   FSTYPE LABEL UUID                                 MOUNTPOINT
sda
├─sda1 ntfs         1A2B3C4D5E6F7G8H                     /mnt/data
├─sda2 swap         2A3B4C5D6E7F8G9H
├─sda3 ext4         3A4B5C6D7E8F9G0H                     /mnt/backup
└─sda4 ext4         4A5B6C7D8E9F0G1H
```

`/dev/sda2`、`/dev/sda4` 没有挂载点，未被挂载。

## 3. 临时挂载

!!! note
    挂载时要注意权限，默认只有 root 有权限访问。普通用户需要指定 uid/gid。

```bash
# 获取 uid 和 gid
id username

# 创建挂载点并挂载
sudo mkdir -p /mnt/sda1
sudo mount -o uid=1000,gid=1000 /dev/sda1 /mnt/sda1

# 验证
df -h | grep /mnt/sda1

# 取消挂载
sudo umount /mnt/sda1
```

## 4. 永久挂载（/etc/fstab）

临时挂载重启后失效，持久化需写入 `/etc/fstab`。

```bash
# 获取分区的 UUID
blkid /dev/sda1
```

编辑 `/etc/fstab`，添加一行：

```
UUID=1A2B3C4D5E6F7G8H /mnt/sda1 ext4 defaults,uid=1000,gid=1000 0 0
```

- 第一列：分区的 UUID（也可直接用 `/dev/sda1`，但推荐 UUID）
- 第二列：挂载点
- 第三列：文件系统类型
- 第四列：挂载选项
- 第五列：dump 备份（0=不备份）
- 第六列：fsck 检查顺序（0=不检查，1=根分区，2=其他）

保存后测试：

```bash
sudo mount -a    # 按 fstab 挂载所有，不报错即配置正确
```

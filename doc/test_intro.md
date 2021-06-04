# `FLEX`代码测试
## 通信server启动
安装完成后，为了运行或测试`FLEX`协议，首先启动消息服务器：

```console
python flex/ionic_bond/message_server.py
```
消息服务器的作用是接受发来的消息并且存储。

## 联邦配置
测试例子中的联邦配置文件均从`test/fed_config_example.py`读取。联邦配置文件定义了联邦的参与方和角色。联邦中有3种角色：`host`,`guest`和`coordinator`。在`fed_config_example`中定义了2组联邦配置：

* 一个`host`，一个`gues`，一个`coordinator`；
* 两个`guest`，一个`coordinator`。一个典型的联邦配置如下：

```json
{
    "server": "localhost:6001",
    "session": {
        "role": "host",
        "local_id": "zhibang-d-014010",
        "job_id": "test_job" },
    "federation": {
        "host": ["zhibang-d-014010"],
        "guest": ["zhibang-d-014011"],
        "coordinator": ["zhibang-d-014012"] }
}
```
其中

* `server`：定义了消息服务器的地址和端口
* `session`：定义了一次运行中本机信息，`role`为本机承担的角色，`local_id`存储机器`IP`信息为`zhibang-d-014010`，`job_id`是任务运行的序列号信息为`test_job`
* `federation`：定义了网络结构，在这个例子中有一个`host`，一个`guest`和一个`coordinator`。

FLEX支持多机和单机模式进行测试，可以分别在多台主机和一台主机上进行模拟测试。进入test目录运行测试。

##单机模式

单机模式下，联邦配置中`federation`中所有机器名都会被重定向到本机。

###批量测试
批量测试对`FLEX`中目前所有的应用协议进行测试。确保本地消息服务器已启动后，切换到`test目录，在终端运行测试

```console
./run_pytest_local_all.sh
```

在屏幕上打印的是`guest`方的运行输出结果。`host`和`coordinator`的运行输出结果分别保存在`_host_test.lo`和`_coord_test.log`中。

如果安装了tmux，可以运行

```console
./run_pytest_local_all_tmux.sh
```
将自动打开3个窗格分别运行HOST，GUEST和COORDINATOR端程序。


### 单协议测试

#### 使用PyTest
以运行安全对齐协议为例：确保本地消息服务器已启动后，切换到test目录，在终端运行测试

```console
./run_pytest_local_all.sh test/federated_sharing/sample_alignment/secure_alignment
```
在屏幕上打印的是`GUEST`方的运行输出结果。`HOST`和`COORDINATOR`的运行输出结果分别保存在`_host_test.log`和`_coord_test.log`中。

如果安装了`tmux`，可以运行

```console
./run_pytest_local_all_tmux.sh test/federated_sharing/sample_alignment/secure_alignment
```
将自动打开3个窗格分别运行`HOST`，`GUEST`和`COORDINATOR`端程序。

#### 独立运行测试程序
需要设置环境变量。以运行安全对齐协议为例：
在第一个终端，设置环境变量`ROLE`为`HOST`，进入源码目录，运行测试

```console
export COMMU_LOCALTEST=TRUE
export PYTHONPATH=$PYTHONPATH:.
python test/federated_sharing/sample_alignment/secure_alignment/host.py
```
在第二个终端，设置环境变量`ROLE`为`GUEST`，进入源码目录，运行测试

```console
export COMMU_LOCALTEST=TRUE
export PYTHONPATH=$PYTHONPATH:.
python test/federated_sharing/sample_alignment/secure_alignment/guest.py
```
在第三个终端，设置环境变量`ROLE`为`COORDINATOR`，进入源码目录，运行测试

```console
export COMMU_LOCALTEST=TRUE
export PYTHONPATH=$PYTHONPATH:.
python test/federated_sharing/sample_alignment/secure_alignment/coordinator.py
```

## 多机模式
当用户在多台主机上安装`FLEX`后，可以运行多机模式

### 批量测试
在`test`目录，修改`fed_config_example.py`中网络路由名称为待测试机器名；修改`run_pytest_dist.sh`中`HOSTS`列表为待测试机器名。
在多台服务器上分别运行测试

```console
./run_pytest_dist.sh
```

### 单协议测试

以运行安全对齐协议为例：
在第一个终端，进入源码目录，运行测试

```console
export PYTHONPATH=$PYTHONPATH:.
python test/federated_sharing/sample_alignment/secure_alignment/host.py
```
在第二个终端，进入源码目录，运行测试

```console
export PYTHONPATH=$PYTHONPATH:.
python test/federated_sharing/sample_alignment/secure_alignment/guest.py
```
在第三个终端，进入源码目录，运行测试

```console
export PYTHONPATH=$PYTHONPATH:.
python test/federated_sharing/sample_alignment/secure_alignment/coordinator.py
```

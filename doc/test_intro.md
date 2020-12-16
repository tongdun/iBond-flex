安装完成后，为了运行或测试FLEX协议，需要首先启动消息服务器：

```console
python flex/ionic_bond/message_server.py
```
消息服务器的作用是接受发来的消息并且存储。

## 联邦配置
测试例子中的联邦配置文件均从test/fed_config_example.py读取。联邦配置文件定义了联邦的参与方和角色。联邦中有3种角色：host,guest和coordinator。在fed_config_example中定义了2组联邦配置，1）一个host，一个guest，一个coordinator；2)两个guest，一个coordinator。一个典型的联邦配置如下：
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
其中，server项定义了消息服务器的地址和端口。session项定义了一次运行中本机的角色，在上述例子中，本机角色为host，id为zhibang-d-014010，程序运行的job_id是test_job。federation项定义了网络结构，在这个例子中有一个host，一个guest，一个coordinator。

联邦配置中的local_id是本机在网络中的路由名，协议运行时通过local_id去寻找相应的服务器。用户在使用FLEX协议前，需要先根据实际的服务器名或ip地址修改local_id和federation信息，根据服务器实际的角色设置role。

FLEX支持多机和单机模式进行测试，可以分别在多台主机和一台主机上进行模拟测试。进入test目录运行测试。

## 单机模式

单机模式下，联邦配置中federation中所有机器名都会被重定向到本机。

### 批量测试
批量测试对FLEX中目前所有的应用协议进行测试。确保本地消息服务器已启动后，切换到test/scripts目录，在终端运行测试

```console
./run_local_tmux.sh
```

将自动打开tmux，创建3个窗格分别运行HOST，GUEST和COORDINATOR端程序。


### 单协议测试

#### 使用PyTest
以运行安全对齐协议为例：确保本地消息服务器已启动后，切换到test/scripts目录，在终端运行测试

```console
./run_local_tmux.sh test/federated_sharing/sample_alignment/secure_alignment
```

将自动打开tmux，创建3个窗格分别运行HOST，GUEST和COORDINATOR端程序。

#### 独立运行测试程序
需要设置环境变量。以运行安全对齐协议为例：
在第一个终端，设置环境变量ROLE为HOST，进入源码目录，运行测试

```console
export COMMU_LOCALTEST=TRUE
export PYTHONPATH=$PYTHONPATH:.
python test/federated_sharing/sample_alignment/secure_alignment/host.py
```

在第二个终端，设置环境变量ROLE为GUEST，进入源码目录，运行测试

```console
export COMMU_LOCALTEST=TRUE
export PYTHONPATH=$PYTHONPATH:.
python test/federated_sharing/sample_alignment/secure_alignment/guest.py
```

在第三个终端，设置环境变量ROLE为COORDINATOR，进入源码目录，运行测试

```console
export COMMU_LOCALTEST=TRUE
export PYTHONPATH=$PYTHONPATH:.
python test/federated_sharing/sample_alignment/secure_alignment/coordinator.py
```

## 多机模式
当用户在多台主机上安装FLEX后，可以运行多机模式

### 批量测试
在test目录，修改fed_config_example.py中网络路由名称为待测试机器名；修改test/scripts/run_distrbuted.sh中HOSTS列表为待测试机器名。
在多台服务器上分别运行测试

```console
./run_distributed.sh
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

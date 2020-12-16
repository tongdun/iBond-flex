# 不经意传输协议
## 简介
不经意传输(oblivious transfer: OT)协议，是一种可保护隐私的双方通信协议，能使通信双方以一种选择模糊化的方式传输消息。协议有两个参与方，发送方和接收方。发送方拥有的消息数为n，尽管发送方有n条消息，但是执行协议后，接收方只能得到他想要得到的其中一条或多条消息。在整个过程中，发送方不能控制接收方的选择，发送方不知道接收方得到了哪几条消息，接收方也不能得到除了选择之外的其它消息。

OT协议具有多种形式，根据收发消息数量的不同可以分为：1-2（2选1），, 1-n（n选1）和 k-n（n选k）的OT协议。OT协议应用广泛，在样本对齐、安全多方计算等领域均有应用。目前FLEX中的匿踪查询使用了1-n的OT协议

## 类和函数
通过`flex.crypto.obilivious_transfer.api`的`make_ot_protocol`函数创建不经意传输的一个实例，用`client`和`server`来分别运行接收方和发送方。

`make_ot_protocol`的定义如下：
```python
def make_ot_protocol(k: int, n: int, remote_id: str) -> OneOutN_OT
```
* 输入：
    * k: 接收方需要获取的信息数，目前支持1条
    * n: 发送方拥有的信息数
    * remote_id：通信使用的对方的ID
* 输出：
    * 返回OneOutN_OT类实例

OneOutN_OT提供如下类方法：
```python
# 客户端
def client(self, index: int) -> str
# 服务端
def server(self, msg_list: List[str]) -> None
```
`client`和`server`需要分别被客户端和服务端调用，运行OT协议。
```python
def client(self, index: int) -> str
```
* 输入：
    * index: 客户端想要获取的信息的索引，例如server端有n条信息，client端将获得第index+1条信息，index的范围从0到n-1
* 输出：
    * 返回第index+1条信息，类型为字符串

```python
def server(self, msg_list: List[str]) -> None
```
* 输入：
    * msg_list: 服务端拥有的信息列表，列表内为字符串
* 输出：
    * None
    
## API调用
## 不经意传输
* Client和Server分别运行：
### Client
```python
from flex.crypto.oblivious_transfer.api import make_ot_protocol
from flex.tools.ionic import commu

# 联邦参与方信息
federal_info = {
    "server": "localhost:6001",
    "session": {
        "role": "coordinator",
        "local_id": "zhibang-d-014010",
        "job_id": 'test_job',
    },
    "federation": {
        "host": ["zhibang-d-014010"],
        "guest": ["zhibang-d-014011"],
        "coordinator": ["zhibang-d-014012"]
    }
}

# 通信初始化
commu.init(federal_info)

# 初始化OT协议
ot_protocol = make_ot_protocol(1, 10, 'zhibang-d-014011')

# 执行client方法
msg = ot_protocol.client(index=5)
print(msg)

```

### Server
```python
from flex.crypto.oblivious_transfer.api import make_ot_protocol
from flex.tools.ionic import commu

# 联邦参与方信息
federal_info = {
    "server": "localhost:6001",
    "session": {
        "role": "coordinator",
        "local_id": "zhibang-d-014011",
        "job_id": 'test_job',
    },
    "federation": {
        "host": ["zhibang-d-014010"],
        "guest": ["zhibang-d-014011"],
        "coordinator": ["zhibang-d-014012"]
    }
}

# 通信初始化
commu.init(federal_info)

# 初始化OT协议
ot_protocol = make_ot_protocol(1, 10, 'zhibang-d-014010')
msg = [str(i) for i in range(10)]

# 执行server方法
ot_protocol.server(msg)
```
代码见：[party_A.py](../../../test/crypto/oblivious_transfer/1_out_n_client.py),
[party_B.py](../../../test/crypto/oblivious_transfer/1_out_n_server.py)
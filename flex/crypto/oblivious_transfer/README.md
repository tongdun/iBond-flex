# 不经意传输协议
## 简介
* 不经意传输(`oblivious transfer`: `OT`)协议，是一种可保护隐私的双方通信协议，能使通信双方以一种选择模糊化的方式传输消息。`OT`协议具有多种形式，根据收发消息数量的不同可以分为：`1-2`（`2`选`1`）, `1-n`（`n`选`1`）和 `k-n`（`n`选`k`）的`OT`协议。
	* 应用场景： `OT`协议应用广泛，在样本对齐、安全多方计算等领域均有应用。目前FLEX中的匿踪查询使用了`1-n`的`OT`协议
	* 相关技术：
		1. 椭圆曲线加密
		2. Hash算法
	* 安全要求：
		1. 发送方拥有的消息数为`n`，尽管发送方有`n`条消息，但是执行协议后，接收方只能得到他想要得到的其中一条或多条消息。
		2. 在整个过程中，发送方不能控制接收方的选择，发送方不知道接收方得到了哪几条消息，接收方也不能得到除了选择之外的其它消息。	
	* 依赖的运行环境：
		1. hashlib
		2. secrets==1.0.2
		3. py_ecc==5.0.0

## 类和函数
不经意传输协议定义了两个参与方，分别是client(客户端)和server(服务端)：

| | client | server |
| ---- | ---- | ---- | 
| class | `OneOutN_OT` | `OneOutN_OT` |
| init | `k`, `n`, `remote_id` | `k`, `n`, `remote_id` |
| method | `client ` | `server` |

### 初始化参数

* `k`:`client`方需要的信息
* `n`:`server`方提供的`n`条信息
* `remote_id`:进行`OT`协议的非本方的参与方的ID

如：

```python
ot_protocol = make_ot_protocol(1, 10, 'zhibang-d-011041')
```

### 类方法
参与方分别使用`client `，`server`方法运行`OT`协议：

```python
# client
def client(self, index: int) -> str:
# server
def server(self, msg_list: List[str]) -> None:
```

#### 输入
`client `方法输入为：

* `index`: 客户端想要获取的信息的索引，例如server端有n条信息，client端将获得第index+1条信息，index的范围从0到n-1

`server`方法输入为：

* `msg_list`: 服务端拥有的信息列表，列表内为字符串

#### 输出
`client `方法输出为：

* 返回第index+1条信息，类型为字符串

`server`方法无输出
    
### OT协议调用示例
client(客户端)调用示例详见：[client.py](../../../test/crypto/oblivious_transfer/1_out_n_client.py)

server(服务端)调用示例详见：[server.py](../../../test/crypto/oblivious_transfer/1_out_n_server.py)
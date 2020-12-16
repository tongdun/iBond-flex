# 密钥交换协议
## 简介
密钥交换协议解决密钥分发问题，使得通信双方可以通过公开信道安全地交换共享的密钥或随机数。

FLEX协议中采用的是经典的Diffile-Hellman密钥交换算法，其安全性是依赖于计算离散对数的困难程度。针对不同等级的安全要求，可以选择使用不同长度的大素数p，常用的有2048、3072、4096、6144、8192位五种长度。

## 类和函数
通过`flex.crypto.key_exchange.api`的`make_agreement`函数创建密钥交换的一个实例。

`make_agreement`的定义如下：
```python
def make_agreement(remote_id: str, key_size: int = 2048) -> int
```
* 输入：
    * remote_id: 需要与本方进行密钥交换的参与方的ID。
    * key_size: 生成的密钥长度，单位为bit
* 输出：
    * 返回key_size比特长度的密钥

## API调用
### 协商密钥
* Party A 和 party B分别运行：
#### partyA
* conf.json
```json
{
        "server": "localhost:16001",
        "session": {
                "role": "host",
                "local_id": "zhibang-d-014010",
                "job_id": "test_job"
        },
                "federation": {
                "host": ["zhibang-d-014010"],
                "guest": ["zhibang-d-014011"]
        }
}
```
* code
```python
import json

from flex.tools.ionic import commu
from flex.crypto.key_exchange.api import make_agreement

# 加载联邦配置文件
with open('conf.json', 'r') as conf_file:
    conf = json.load(conf_file)

# 通信初始化
commu.init(conf)

# 运行密钥交换协议
k = make_agreement(remote_id='zhibang-d-014011', key_size=2048)
print(k)
```

#### partyB
* conf.json
```json
{
        "server": "localhost:16001",
        "session": {
                "role": "host",
                "local_id": "zhibang-d-014011",
                "job_id": "test_job"
        },
                "federation": {
                "host": ["zhibang-d-014010"],
                "guest": ["zhibang-d-014011"]
        }
}
```
* code
```python
import json

from flex.tools.ionic import commu
from flex.crypto.key_exchange.api import make_agreement

# 加载联邦配置文件
with open('conf.json', 'r') as conf_file:
    conf = json.load(conf_file)

# 通信初始化
commu.init(conf)

# 运行密钥交换协议
k = make_agreement(remote_id='zhibang-d-014010', key_size=2048)
print(k)
```
代码见：[party_A.py](../../../test/crypto/key_exchange/party_A.py),
[party_B.py](../../../test/crypto/key_exchange/party_B.py)
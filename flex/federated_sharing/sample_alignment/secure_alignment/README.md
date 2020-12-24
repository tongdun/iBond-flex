# 安全样本对齐
## 简介
* 第三方辅助的样本对齐协议，主要思想是利用安全密钥交换协议，参与方（不包括第三方）协商一个共同的密钥，使用该密钥加密本方待对齐的样本ID，将密文发送至第三方，第三方在密文集合上求交集并将对齐顺序信息返回给参与方，各个参与方按顺序提取本方样本集合的子集，从而得到按相同顺序排列的样本交集。
    * 应用场景: 双方样本集合大小在千万级别以下的情景。
    * 相关技术: 
        1. Diffie-Hellman密钥交换协议;  
        2. AES对称加密;
    * 算法流程图  
        ![FLEX](../../../../doc/pic/secure_alignment.png)
    * 安全要求: 
        1. 参与方不能知道其它方的用户ID，也不能泄漏交集外的用户样本信息;
        2. 第三方不能知道甚至追溯到参与方的用户ID数据.
    * 依赖的运行环境
        1. numpy>=1.18.4
        2. crypto==1.4.1
        3. gmpy2==2.0.8
        4. secrets==1.0.2
    * 协议流程，详见: [FLEX白皮书](../../../../doc/FLEX白皮书.pdf)2.2.2章节
        

## 类和函数
BF_SF协议定义了三种类型的参与方，分别是Coordinator,Guest,Host，它们对应的类函数、初始化参数、类方法如下：

| | Coordinator | Guest | Host |
| ---- | ---- | ---- | ---- |
| class | SALCoord | SALGuest | SALHost |
| init | federal_info, sec_param | federal_info, sec_param | fedral_info, sec_param |
| method | align, verify | align, verify | align, verify |

### 初始化参数
每种参与方在初始化时需要提供federal_info、sec_param两种参数。其中federal_info提供了联邦中参与方信息，sec_param是协议的安全参数。

* sec_param中需提供的参数有：
   * key_exchange_size: DH密钥交换协议的密钥长度

   如:
    ```json
    {
        "key_exchange_size": 2048
    }
    ```

### 类方法
每种参与方均提供align和verify方法，如下
```python
# Coordinator
def align(self) -> None
def verify(self) -> None
# Guest
def align(self, ids: list) -> list
def verify(self, ids: list) -> bool
# Host
def align(self, ids: list) -> list
def verify(self, ids: list) -> bool
```
#### 输入
Coordinator无需输入参数，其他参数意义如下：
* ids: 表示安全对齐协议的参与方Guest和Host，需要安全对齐的样本列表，长度为样本的数量.

例如：
```python
ids = list(map(str, range(10000, 20000)))
```

#### 输出
Coordinator无输出，若为对齐过程，则Guest方和Host方的输出为样本对齐后的ids，若为验证过程，则Guest方和Host方的输出为是否对齐。

## API调用
每种参与方均通过如下方式初始化：
```python
from flex.api import make_protocol
from flex.constants import SAL

protocol = make_protocol(SAL, federal_info, sec_param, algo_param)
```
调用时，根据federal_info中参与方角色的不同，分别返回SALCoord，SALGuest和SALHost三种类实例。

### secure_alignment调用示例
#### Host
   详见: [host.py](../../../../test/federated_sharing/sample_alignment/secure_alignment/host.py)

#### Guest
   详见: [guest.py](../../../../test/federated_sharing/sample_alignment/secure_alignment/guest.py)

#### Coordinator
   详见: [coordinator.py](../../../../test/federated_sharing/sample_alignment/secure_alignment/coordinator.py)




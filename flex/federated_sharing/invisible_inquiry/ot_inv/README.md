# 匿踪查询
## 简介
* 匿踪查询(invisible inquiry)主要是解决查询过程中如何保护查询请求方用户ID信息不为其它参与方所知。匿踪查询主要是采用混淆扩充和不经意传输（OT）两种技术手段来隐匿查询方的用户ID，让其它参与方无法轻易追踪到具体ID。
    * 应用场景:  
        应用于联邦共享或数据查询的场景中。一方为查询请求方，另一方为提供查询服务的服务方。由请求方提供用户ID信息向服务方发起查询请求，服务方发接收到请求后在本地数据库检索该ID信息，并将结果返回给查询方。
        用户ID可以是用户身份证号、手机号、姓名等具有唯一标识的信息明文或密文。如是密文，则要求各参与方必须采用相同的脱敏加密方法对明文进行加工处理。
    * 相关技术: 
        1. 不经意传输协议(oblivious transfer protocol),具体参考[OT](../../../crypto/oblivious_transfer/README.md);  
        2. AES对称加密;
    * 算法流程图  
        ![FLEX](../../../../doc/pic/invisible inquiry.png)
    * 安全要求:  
        1.查询请求方只能获得查询ID对应的查询结果，不能获得其它额外信息;  
        2.服务方不能直接知道查询ID，但允许以某个概率猜出查询ID.
    * 依赖的运行环境
        1. py_ecc==5.1.0
        2. crypto==1.4.1
        3. secrets==1.0.2
    * 协议流程，详见: [FLEX白皮书](../../../../doc/FLEX白皮书.pdf)2.1章节

## 类和函数
OT_INV协议定义了两种类型的参与方，分别是Client, Server，它们对应的类函数、初始化参数、类方法如下：

| | Client | Server |
| ---- | ---- | ---- |
| class | OTINVClient | OTINVServer |
| init | federal_info, sec_param, algo_param | fedral_info, sec_param, algo_param |
| method | exchange | exchange |

### 初始化参数
每种参与方在初始化时需要提供federal_info、sec_param和algo_param三种参数。其中federal_info提供了联邦中参与方信息，sec_param是协议的安全参数， algo_param是协议的算法参数。

* sec_param中需提供的参数有：
   * symmetric_algo: 对称加密方法

   如:
    ```json
    {
        "symmetric_algo": "aes"
    }
    ```
* algo_param中需要提供的参数有：
    * k：匿踪查询支持的单次查询数量，目前仅支持n选1的匿踪查询，因此k=1
    * n：匿踪查询混淆后的查询数量，用于混淆查询索引
    
   如：
    ```json
    {
        "k": 1,
        "n": 10
    }
    ```

### 类方法
每种参与方均提供exchange方法，如下
```python
# Client
def exchange(self, ids: Union[str, List[str]], obfuscator: Callable[[List[str], int], Tuple[List[str], int]]) -> Union[str, List[str]]
# Server
def exchange(self, query_fun: Callable[[List[str]], List[str]]) -> None
```
#### 输入
参数意义如下：
* ids: 表示查询方需要查询的id，类型可以是string或者list，由于目前仅支持n选1的匿踪查询，因此ids输入为string.
* obfuscator: 混淆函数，通过该函数，可以将ids混淆成预先设定的数量，用于隐藏真实的查询ids.
* query_fun: 表示服务方的提供的查询函数。混淆后的查询ids通过该函数可以得到对应的查询结果.

例如：
```python
ids = '50'
def obfuscator(in_list, n):
    fake_list = [random.randint(0, 100) for i in range(n-len(in_list))]
    index = random.randint(0, n-1)
    joint_list = fake_list[:index] + in_list + fake_list[index:]
    return joint_list, index
def query_fun(in_list):
    result = [str(int(i) * 100) for i in in_list]
    return result
```

#### 输出
Server无输出，Client方的输出为查询ids对应的结果。

## API调用
每种参与方均通过如下方式初始化：
```python
from flex.api import make_protocol
from flex.constants import OT_INV

protocol = make_protocol(OT_INV, federal_info, sec_param, algo_param)
```
调用时，根据federal_info中参与方角色的不同，分别返回OTINVClient，OTINVServer两种类实例。

### 匿踪查询
* Client和Server分别运行：
#### 查询方 Client
   详见: [client.py](../../../../test/federated_sharing/invisible_inquiry/client.py)

#### 服务方 Server
   详见: [server.py](../../../../test/federated_sharing/invisible_inquiry/server.py)




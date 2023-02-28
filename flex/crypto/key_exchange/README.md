# 密钥交换协议
## 简介
* FLEX协议中采用的是经典的Diffile-Hellman密钥交换算法，其安全性是依赖于计算离散对数的困难程度。针对不同等级的安全要求，可以选择使用不同长度的大素数p，常用的有2048、3072、4096、6144、8192位五种长度。
    * 应用场景：
        * 密钥交换协议解决密钥分发问题，使得参与方可以通过公开信道安全地交换共享的密钥或随机数。

    * 依赖的运行环境
        * gmpy2==2.0.8
        * secrets==1.0.2
	
## 类和函数
* `DH_KEY_EXCHANGE`协议通过`flex.crypto.key_exchange.dh_key_exchange.api`的`make_agreement`函数创建密钥交换的一个实例。其中初始化参数，类方法如下：

### 参数初始化
参与方在协议初始化时需要提供`remote_id`,`local_id`,`key_length`三个初始化参数:

* `remote_id`使用`list`的形式提供所有参与方的ID：

    ```python
    ["zhibang-d-011040", "zhibang-d-011041", "zhibang-d-011042"]
    ```
   
* `local_id`提供参与方本地的ID信息:

    ```python
    "zhibang-d-011040"
    ```
* `key_length`可选择需要的密钥长度，单位为`bit`,目前支持`[2048, 3072, 4096, 6144, 8192]bit`

### 类方法
参与方可通过`make_agreement`方法进行协议的实现，调用方法如下：

```python
def make_agreement(remote_id: List, local_id: str, key_length: int = 2048) -> int:
```

#### 输入
* `remote_id`: 需要与本方进行密钥交换的参与方的ID
* `local_id`: 参与方本地的ID信息
* `key_length`: 生成的密钥长度，单位为bit

#### 输出
* 返回`key_length`比特长度的密钥

### 密钥交换协议调用示例
参与方的调用示例详见：
[party_A.py](../../../../test/crypto/key_exchange/party_A.py),
[party_B.py](../../../../test/crypto/key_exchange/party_B.py),
[party_C.py](../../../../test/crypto/key_exchange/party_C.py)
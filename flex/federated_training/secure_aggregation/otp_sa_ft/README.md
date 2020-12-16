#  安全聚合
## 简介
* 参与方待交换的中间结果按位运算加上一个随机数，而所有参与方使用的随机数总和为0，这样在第三方就可以直接进行中间结果的聚合平均计算。协议的重点是如何生成多个随机数，并在各方不知其它方随机数的情况下能保证总和为0，这也是使用安全密钥交换和安全伪随机数生成协议的原因。
    * 应用场景: 跨样本联邦训练，各参与方将参数变量的中间结果发送到第三方，由第三方对其聚合并返还聚合结果。
    * 相关技术: 
        1. Diffie-Hellman密钥交换协议;  
        2. 安全伪随机数生成;
        3. 一次一密.
    * 算法流程图  
        ![FLEX](../../../../doc/pic/OTP-SA-FT.png)
    * 安全要求: 
        1. 不造成训练数据和标签的泄漏;
        2. 第三方也无法知道实际的模型参数中间结果.
    * 依赖的运行环境
        1. numpy>=1.18.4
        2. torch>=1.3.1
        3. gmpy2>=2.0.8
    * 协议流程，详见: [FLEX白皮书](../../../../doc/FLEX白皮书.pdf)5.5.1章节

## 类和函数
HE_SA_FT协议定义了三种类型的参与方，分别是Coordinator,Guest,Host，它们对应的类函数、初始化参数、类方法如下：

| | Coordinator | Guest | Host |
| ---- | ---- | ---- | ---- |
| class | OTPSAFTCoord | OTPSAFTGuest | OTPSAFTHost |
| init | federal_info, sec_param | federal_info, sec_param | fedral_info, sec_param |
| method | exchange | exchange | exchange |


### 初始化参数
每种参与方在初始化时需要提供federal_info和sec_param两种参数。其中federal_info提供了联邦中参与方信息，sec_param是协议的安全参数。

* sec_param中需提供的参数有：
   * key_exchange_size: 密钥交换协议密钥长度
   
   如:
    ```json
    {
        "key_exchange_size": 2048
    }
    ```

### 类方法
每种参与方均提供exchange方法，如下
```python
# Coordinator
def exchange(self)
# Guest
def exchange(self, theta: Union[list, np.ndarray, torch.Tensor])
# Host
def exchange(self, theta: Union[list, np.ndarray, torch.Tensor])
```

#### 输入
Coordinator无需输入参数，其他参数意义如下：
theta:各参与方的模型参数的中间结果

例如：
```python
theta = [[np.random.uniform(-1, 1, (2, 4)).astype(np.float32), 
          np.random.uniform(-1, 1, (2, 6)).astype(np.float32)],
         [np.random.uniform(-1, 1, (2, 8)).astype(np.float32)]]
```

#### 输出
Coordinator无输出，Guest方和Host方的输出为聚合后的中间结果。

## API调用
每种参与方均通过如下方式初始化：
```python
from flex.api import make_protocol
from flex.constants import OTP_SA_FT

protocol = make_protocol(OTP_SA_FT, federal_info, sec_param, algo_param=None)
```
调用时，根据federal_info中参与方角色的不同，分别返回OTPSAFTCoord，OTPSAFTGuest和OTPSAFTHost三种类实例。


### otp_sa_ft调用示例
#### Host
   详见[host.py](../../../../test/federated_training/secure_aggregation/otp_sa_ft/host.py)

#### Guest
   详见[guest.py](../../../../test/federated_training/secure_aggregation/otp_sa_ft/guest.py)

#### Coordinator
   详见[coordinator.py](../../../../test/federated_training/secure_aggregation/otp_sa_ft/coordinator.py)

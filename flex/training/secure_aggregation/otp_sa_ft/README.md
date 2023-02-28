#  安全聚合
## 简介
* 参与方待交换的中间结果按位运算加上一个随机数，而所有参与方使用的随机数总和为0，这样在第三方就可以直接进行中间结果的聚合平均计算。协议的重点是如何生成多个随机数，并在各方不知其它方随机数的情况下能保证总和为0，这也是使用安全密钥交换和安全伪随机数生成协议的原因。
    * 应用场景: 跨样本联邦训练，各参与方将参数变量的中间结果发送到第三方，由第三方对其聚合并返还聚合结果。
    
    * 相关技术: 
        * Diffie-Hellman密钥交换协议;  
        * 安全伪随机数生成;
        * 一次一密.
        * paillier同态加密
        
    * 算法流程图  
        * 一次一密聚合流程图：
        
        ![FLEX](../../../../doc/pic/OTP-SA-FT.png)
        
        * paillier同态加密聚合流程图：
        
        ![FLEX](../../../../doc/pic/HE-SA-FT.png)
        
    * 安全要求: 
        * 不造成训练数据和标签的泄漏;
        * 第三方也无法知道实际的模型参数中间结果.
        
    * 依赖的运行环境
        * numpy>=1.18.4
        * torch>=1.3.1
        * gmpy2>=2.0.8
        
    * 协议流程，详见: [FLEX白皮书](../../../../doc/FLEX白皮书.pdf)5.5.1章节及5.5.2章节

## 类和函数
`OTP_SA_FT`协议定义了两种类型的参与方，分别是`Coordinator`,`Party`，它们对应的类函数、初始化参数、类方法如下：

| |Coordinator | Party |
| ---- | ---- | ---- |
| class | `OTPSAFTCoord` | `OTPSAFTParty` |
| params | `federal_info`, `sec_param` | `fedral_info`, `sec_param` |
| methods | `exchange`, `confer_param` | `exchange`, `confer_param` |

` 
### 初始化参数
每种参与方在初始化时需要提供`federal_info`和`sec_param`两种参数。其中`federal_info`提供了联邦中参与方信息，`sec_param`是协议的安全参数。

* `sec_param`中需提供的参数有：
    * 使用`list`嵌套`list`形式存储加密信息，第一层`list`存储此次协议所有加密方式；第二层`list`的第一个元素表示加密的方法，第二个元素表示该加密方法需要用到的参数
 
    本协议中使用`onetime_pad`加密或者`paillier`加密，`sec_param`如下:
   
    ```python
    [['onetime_pad', {"key_length": 512}]]
    ```
  
    或者
   
    ```python
    [('paillier', {"key_length": 1024})]
    ```
 

### 类方法
每种参与方均提供`exchange`方法实现聚合:

```python
# Coordinator
def exchange(self, *args, **kwargs) -> None
# Party
def exchange(self, theta: Union[list, np.ndarray, torch.Tensor]) -> Union[list, np.ndarray, torch.Tensor]
```

#### 输入
`Coordinator`无需输入参数，其他参数意义如下：
`theta`:各参与方的模型参数的中间结果
`data`:需要约定参数的情况下参数的值

#### 输出
`Coordinator`无输出，`Party`方的输出为聚合后的中间结果。

### otp_sa_ft调用示例
参与方`Party`调用示例详见[party.py](../../../../test/training/secure_aggregation/otp_sa_ft/party1.py)

中间方`Coordinator`调用示例详见[coordinator.py](../../../../test/training/secure_aggregation/otp_sa_ft/coordinator.py)

#  线性回归
## 简介
* 跨特征联邦线性回归训练。基本思想：由于跨特征联邦线性回归中关键是计算损失函数，因此不同参与方可以先利用本地数据计算，将得到的中间结果加密后上传至第三方，由第三方汇总后解密，再反馈给各参与方。
    * 应用场景:  
		* 在跨特征联邦训练中，如果模型采用逻辑回归模型，参与方之间协同计算模型参数更新时可以采用该协议

	* 相关技术: 
		* paillier同态加密，具体参考安全公共组件[paillier](../../../crypto/paillier/README.md)的实现
		* 线性回归算法	

    * 算法流程图
        ![FLEX](../../../../doc/pic/he_linear_ft.png)

    * 安全要求
        数据交换的过程保证安全，传输的数据不会产生隐私泄漏，即其中一方无法根据接收到的密文求解或推算出另一方的明文数据。

    * 依赖的运行环境
        1. numpy>=1.18.4

    * 协议流程，详见：[FLEX白皮书](../../../../doc/FLEX白皮书.pdf)5.1章节

## 类和函数
HE_Lienar_FT协议定义了三种类型的参与方，分别是Coordinator,Guest,Host，它们对应的类函数、初始化参数、类方法如下：

| | Coordinator | Guest | Host |
| ---- | ---- | ---- | ---- |
| class |`HELinearFTCoord` | `HELinearFTGuest` | `HELinearFTHost` |
| init | `federal_info`, `sec_param` | `federal_info`, `sec_param` | `fedral_info`, `sec_param` |
| method | `exchange` | `exchange` | `exchange` |

### 初始化参数
每种参与方在初始化时需要提供`federal_info`和`sec_param`两种参数。其中`federal_info`提供了联邦中参与方信息，`sec_param`是协议的安全参数。

* `sec_param`中需提供的参数有：
   * `he_algo`: 同态加密算法名
   * `key_length`: 同态加密密钥长度

   如:
   
    ```python
    [['paillier', {"key_length": 1024}], ]
    ```

### 类方法
每种参与方均提供`exchange`方法，如下
```python
# Coordinator
def exchange(self) -> None
# Guest
def exchange(self, grad: np.ndarray, label: np.ndarray) -> np.ndarray
# Host
def exchange(self, grad: np.ndarray) -> np.ndarray
```

#### 入参说明
`Coordinator`无需输入参数，其他参数意义如下：
* `local_prediction`: 表示$`\theta x^T`$，为1维numpy.ndarray，长度等于batch大小；
* `label`: 表示label，用一维numpy.ndarray表示，长度等于batch大小，值为-1或1。

例如：
```python
local_prediction = np.random.uniform(-1, 1, (32,))
label = np.random.randint(0, 2, (32,))
```

#### 输出
`Coordinator`无输出，`Guest`方和`Host`方的输出为`loss`值。

### `HE_Linear_FT`调用示例

`Host`(参与方)调用示例详见：[host.py](../../../../test/training/linear_regression/he_linear_ft/host.py)

`Guest`(发起方)调用示例详见：[guest.py](../../../../test/training/linear_regression/he_linear_ft/guest.py)

`Coordinator`(中间方)调用示例详见：[coordinator.py](../../../../test/training/linear_regression/he_linear_ft/coordinator.py)

### 训练示例

`Host`(参与方)使用`HE_Linear_FT`协议进行完整的训练的示例详见：[test_host.py](../../../../test/training/linear_regression/he_linear_ft/test_host.py)

`Guest`(发起方)使用`HE_Linear_FT`协议进行完整的训练的示例详见：[test_guest.py](../../../../test/training/linear_regression/he_linear_ft/test_guest.py)

`Coordinator`(中间方)使用`HE_Linear_FT`协议进行完整的训练的示例详见：[test_coordinator.py](../../../../test/training/linear_regression/he_linear_ft/test_coordinator.py)

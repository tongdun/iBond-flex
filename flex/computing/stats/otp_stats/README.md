#  联邦数据统计计算协议

## 简介

* 基于秘密分享的数据统计。基本思想：
先本地计算统计量，然后对结果秘密分享，在秘密分享的碎片上一次一密的方式得到全体数据的统计量。

* 支持的统计量
    * 计数
    * 均值
    * 方差

* 协议过程：参与方提供本地数据D_i，发起联邦统计协议。本地统计量在本地完成，联邦统计量在本地统计量的基础上计算联邦统计量本地的部分，通过一次一密加密后传输到第三方，第三方不提供数据，完成各个参与方部分联邦统计量的计算，发还给各个参与方，参与方解密后得到联邦统计量。

* 依赖的运行环境
    1. numpy>=1.18.4

## 类和函数

OTP_STATS协议定义了三种类型的参与方，分别是Coordinator和Guest，它们对应的类函数、初始化参数、类方法如下：

| | Coordinator | Guest |
| ---- | ---- | ---- |
| class | OTPStatisticCoord OTPStatisticGuest |
| init | federal_info, sec_param | federal_info, sec_param |
| method | exchange | exchange |

### 初始化参数

每种参与方在初始化时需要提供federal_info和sec_param两种参数。其中federal_info提供了联邦中参与方信息，sec_param是安全参数。


### 类方法

每种参与方均提供exchange方法，如下

```python
# Coordinator
def exchange(self, stats=['std'])
# Guest
def exchange(self, data: np.ndarray, stats=['std'])
```

#### 输入

* stats: 需要计算的统计量的列表。统计量表示为'std', 'mean', 'count', 'max', 'min'。
* data: 各个Guest提供，一维numpy array。

例如：

```python
stats = ['std', 'min', 'max']
data = np.asarray([1.7, 2, np.nan, 6, -10.3])
```

#### 输出

无输出，Guest方得到的统计量记录在Guest方的data_sta字典中。

## API调用

每种参与方均通过如下方式初始化：

```python
from flex.api import make_protocol
from flex.constants import OTP_STATS

protocol = make_protocol(OTP_STATS, federal_info, sec_param=None, algo_param)
```

调用时，根据federal_info中参与方角色的不同，分别返回OTPStatisticCoord和OTPStatisticGuest类实例。

### 调用示例

Guest方调用示例详见：[guest.py](../../../../test/computing/stats/otp_stats/test_guest.py)


Coordinator方调用示例详见：[coordinator.py](../../../../test/computing/stats/otp_stats/test_coordinator.py)
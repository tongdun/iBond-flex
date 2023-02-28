#  多头共债协议

## 简介

* 基于秘密分享的多头共债。基本思想：
利用秘密分享的原理将用户的偿还能力和贷款额拆分成碎片，并在服务方之间分享和计算。第三方对计算结果进行汇总统计，并将结果反馈给发起方。

* 协议过程：该协议中，同样是由发起方向第三方发起查询服务，并向第三方提供用户ID和其偿还能力，而第三方则根据用户ID分别向服务方发起查询和判断。协议过程中，第三方会首先根据服务方数量将偿还能力（金额）拆分成碎片，并将偿还能力秘密碎片分享给所有服务方；然后服务方会将贷款额拆分成多个碎片，并将贷款额碎片分享给其它服务方；同时服务方会接收发送过来的贷款额和偿还能力碎片，对接收到的碎片进行计算，并将计算结果上传至第三方；随后第三方根据汇集的结果进行统计，得到查询判断结果，并将结果反馈给发起方。

* 算法流程图
![FLEX](../../../../doc/pic/ss_ml.png)

* 依赖的运行环境
    1. torch==1.6.0

## 类和函数

`SS_ML`协议定义了三种类型的参与方，分别是`Coordinator`,`Guest`,`Host`，它们对应的类函数、初始化参数、类方法如下：

| | Coordinator | Guest | Host |
| ---- | ---- | ---- | ---- |
| class | `SSMLCoord` | `SSMLGuest` | `SSMLHost` |
| init | `federal_info`, `sec_param`, `algo_param` | `federal_info`, `sec_param`, `algo_param` | `fedral_info`, `sec_param`, `algo_param` |
| method | `exchange` | `exchange` | `exchange` |

### 初始化参数

每种参与方在初始化时需要提供`federal_info`、`sec_param`以及`algo_param`三种参数。其中`federal_info`提供了联邦中参与方信息，`sec_param`是协议的安全参数，`algo_param`是协议的算法参数。

* `sec_param`中需提供的参数有：
   * 使用`list`嵌套`list`形式存储加密信息，第一层`list`存储此次协议所有加密方式(协议只会用到一种加密协议)；第二层`list`的第一个元素表示加密的方法(协议采用`secret_sharing`加密)，第二个元素表示该加密方法需要用到的参数(`secret_sharing`加密需要秘钥的长度`precision`)
 
		```python
		[["secret_sharing", {"precision": 4}],]
		```
   
* `algo_param`中，算法参数，故`algo_param = {}`

### 类方法

每种参与方均提供`exchange`方法，如下

```python
# Coordinator
def exchange(self)
# Guest
def exchange(self, user_id: Union[str, int], r_raw: Union[int, float])
# Host
def exchange(self, req_loan: Callable)
```

#### 输入

`Coordinator`无需输入参数，其他参数意义如下：
* `user_id`: 表示被查询用户的id，为字符串类型。
* `r_raw`: 表示额度限额，为整数或者浮点数。
* `req_loan`: 查询函数，接受u_id为参数，返回该用户额度，返回类型为整数或者浮点数。

例如：

```python
user_id = 'user_A'
r_raw = 600
def req_loan(user_id):
    return 500.0
```

#### 输出

Coordinator和Host无输出，Guest方的输出为贷款总额是否超过阈值，用0和1表示。

## API调用

每种参与方均通过如下方式初始化：

```python
from flex.api import make_protocol
from flex.constants import SS_ML

protocol = make_protocol(SS_ML, federal_info, sec_param, algo_param)
```

调用时，根据federal_info中参与方角色的不同，分别返回SSMLCoord，SSMLGuest和SSMLHost三种类实例。

### ss_ml 调用示例

Host方调用示例详见：[host.py](../../../../test/computing/multi_loan/ss_ml/test_host.py)


Guest方调用示例详见：[guest.py](../../../../test/computing/multi_loan/ss_ml/test_guest.py)


Coordinator方调用示例详见：[coordinator.py](../../../../test/computing/multi_loan/ss_ml/test_coordinator.py)
#  联邦计算协议

## 简介

* 联邦计算数据交换协议。本协议利用矩阵秘密分享的思想，首先将各参与方的特征数据分别拆分成碎片，然后参与方利用矩阵碎片信息生成秘密碎片用于共享和传播。最后在发起方这边根据获取的秘密碎片，在碎片上进行计算。

* 协议过程
提供了数据分享为碎片的函数share。在隐私数据分享为碎片后，在碎片上的操作提供了多个计算函数如add, mul等。这些计算函数大都包含了直接恢复的版本add_rec, mul_rec等，调用直接恢复函数会在原计算的基础上恢复结果碎片，得到结果。

* 算法流程图
![FLEX](../../../../doc/pic/ss_compute.png)

* 依赖的运行环境
    1. numpy>=1.18.4
    2. torch>=1.16
    2. flex.crypto.secure_multiparty_computation

* 协议流程，详见：[FLEX白皮书](../../../../doc/FLEX白皮书.pdf)w.x.y章节

## 类和函数

FEDCompute协议定义了三种类型的参与方，分别是Coordinator,Guest,Host，它们对应的类函数、初始化参数、类方法如下：

| | Coordinator | Guest | Host |
| ---- | ---- | ---- | ---- |
| class | SSComputeCoord | SSComputeGuest | SSComputeHost |
| init | federal_info, sec_param, algo_param | federal_info, sec_param, algo_param | fedral_info, sec_param, algo_param |
| method | add<br>substract<br>mul<br>matmul<br>scala_mul<br>scala_div<br>ge0<br>ge<br>le<br>add_rec<br>substract_rec<br>mul_rec<br>matmul_rec<br>scala_mul_rec<br>scala_div_rec<br>ge0_rec<br>ge_rec<br>le_rec | add<br>substract<br>mul<br>matmul<br>scala_mul<br>scala_div<br>ge0<br>ge<br>le<br>add_rec<br>substract_rec<br>mul_rec<br>matmul_rec<br>scala_mul_rec<br>scala_div_rec<br>ge0_rec<br>ge_rec<br>le_rec | add<br>substract<br>mul<br>matmul<br>scala_mul<br>scala_div<br>ge0<br>ge<br>le<br>add_rec<br>substract_rec<br>mul_rec<br>matmul_rec<br>scala_mul_rec<br>scala_div_rec<br>ge0_rec<br>ge_rec<br>le_rec |


### 初始化参数

每种参与方在初始化时需要提供federal_info、sec_param以及algo_param三种参数。其中federal_info提供了联邦中参与方信息，sec_param是协议的安全参数， algo_param是协议的算法参数。

* `sec_param`中需提供的参数有：
   * 使用`list`嵌套`list`形式存储加密信息，第一层`list`存储此次协议所有加密方式(协议只会用到一种加密协议)；第二层`list`的第一个元素表示加密的方法(协议采用`secret_sharing`加密)，第二个元素表示该加密方法需要用到的参数(`secret_sharing`加密需要秘钥的长度`precision`)
 
		```python
		[["secret_sharing", {"precision": 4}],]
		```
   
* `algo_param`中，算法参数，故`algo_param = {}`

### 类方法

每种参与方均提供add, substract, ge, le等方法，如下

```python
# Coordinator
def add(self)
# Guest
def add(self, lhs: Union[np.ndarray, torch.Tensor], rhs: Union[np.ndarray, torch.Tensor])
# Host
def add(self, lhs: Union[np.ndarray, torch.Tensor], rhs: Union[np.ndarray, torch.Tensor])
```

不带_rec结尾的函数的输入输出均为碎片，而带有rec结尾的函数的结果是碎片解析后的结果。每种参与方均提供add_rec, substract_rec, ge_rec, le_rec等方法，如下

```python
# Coordinator
def add_rec(self)
# Guest
def add_rec(self, lhs: Union[np.ndarray, torch.Tensor], rhs: Union[np.ndarray, torch.Tensor])
# Host
def add_rec(self, lhs: Union[np.ndarray, torch.Tensor], rhs: Union[np.ndarray, torch.Tensor])
```

#### 输入

Coordinator无需输入参数，其他参数意义如下：
* mat_raw: 表示参与方Guest和Host提供的特征数据，为np.ndarray或torch.Tensor形式表示；

例如：

```python
import torch
torch.manual_seed(seed=1111111)
mat_raw = torch.rand(500, 200)
```

#### 输出

Coordinator和Host无输出，Guest方的输出为联邦矩阵计算结果。

## API调用

每种参与方均通过如下方式初始化：

```python
from flex.api import make_protocol
from flex.constants import SS_COMPUTE

protocol = make_protocol(SS_COMPUTE, federal_info, sec_param, algo_param)
```

调用时，根据federal_info中参与方角色的不同，分别返回SSComputeCoord，SSComputeGuest和SSComputeHost三种类实例。

### 调用示例

Host方调用详见：[host.py](../../../../test/computing/ss_compute/host.py)


Guest方调用示例详见：[guest.py](../../../../test/computing/ss_compute/guest.py)


Coordinator方调用示例详见：[coordinator.py](../../../../test/computing/ss_compute/coordinator.py)
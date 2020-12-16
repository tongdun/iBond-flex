FLEX中目前支持多种类型的应用协议，包括联邦共享、联邦预处理、联邦计算、联邦训练、联邦预测几个大类，具体的协议如下表所示：

| 协议类型 | 协议名 | 协议描述 | 
| :----- | :----- | :-----   |
| 联邦共享 |OT_INV|匿踪查询：解决查询过程中如何保护查询请求方用户ID信息不为其它参与方所知|
| 联邦共享 |SAL |安全对齐：一种简单、高效的样本对齐协议，其目的是计算参与方之间的样本交集|
| 联邦预处理 |IV_FFS |信息价值特征选择：跨特征联邦场景下的信息价值的计算|
| 联邦训练 |HE_LINEAR_FT |跨特征联邦线性回归训练：跨特征线性回归训练中计算损失函数|
| 联邦训练 |HE_OTP_LR_FT1 |跨特征联邦逻辑回归训练：跨特征逻辑回归训练中计算损失函数|
| 联邦训练 |HE_OTP_LR_FT2 |跨特征联邦逻辑回归训练：跨特征逻辑回归训练中计算损失函数|
| 联邦训练 |OTP_SA_FT |安全聚合：基于一次一密的安全聚合|
| 联邦训练 |HE_SA_FT |安全聚合：基于同态加密的的安全聚合|
| 联邦预测 |HE_LR_FP |跨特征联邦逻辑回归预测：跨特征逻辑回归预测中间结果汇总|

# 协议初始化
FLEX设计了统一的接口来初始化各类应用协议，通过flex.api中make_protocol方法统一创建。调用方式如下：
```python
from flex.api import make_protocol
from flex.constants import *

protocol = make_protocol(protocol_name, federal_info, sec_param, algo_param)
```

# 输入

* protocol_name

  protocol_name在flex.constants中统一定义，见上表协议名列，与FLEX白皮书中协议名一致。

* federal_info

  federal_info规定了联邦参与方的信息和本方的信息。其基本格式如下：
```json
{
   "server":        "localhost:6001",
   "session": 
   {
       "role":      "guest",
       "local_id":  "zhibang-d-014011",
       "job_id":    "test_job"
   },
   "federation": 
   {
       "host":          ["zhibang-d-014010"],
       "guest":         ["zhibang-d-014011"],
       "coordinator":   ["zhibang-d-014012"]
   }
}
```
其中，server为本地服务主机和通信协议端口号；federation中规定了本次联邦的所有参与方，分为host, guest和coordinator三种角色，每种角色由多个参与方的id组成；session中规定了本方的角色-role，本地的参与方id-local_id和本次协议的job_id。
一般地，不同次协议应使用不同的job_id来区分。

在匿踪查询协议OT_INV中，需要在federal_info的session中额外提供identity关键字，identity的值有client和server两种。

* sec_param

  sec_param是协议中与安全相关的参数，规定了协议中使用的加密方式，安全密码的长度等。如线性回归HE_Linear_FT中，安全参数包括同态加密算法名和同态加密密钥长度，如下所示：
```python
{
    "he_algo": "paillier",
    "he_key_length": 1024
}
```
表示该算法使用paillier同态加密，密钥长度为1024。每种协议的默认安全参数定义在对应协议的实现代码所在文件夹中，以sec_param.json命名。协议初始化时，会先加载默认安全参数，再覆盖以用户自定义参数。

* algo_param

  algo_param是协议中与模型、算法相关的超参数。每种协议的默认超参数定义在对应协议的实现代码所在文件夹中，以algo_param.json命名。协议初始化时，会先加载默认超参数，再覆盖以用户自定义参数。并非每种协议都拥有超参，对于无需超参的协议，可不输入该参数或设置为None。

# 输出
对于每种协议，根据federal_info中role的不同，make_protocol返回不同的类实例。如当协议名为HE_LINEAR_FT时，make_protocol可返回三种不同的类实例，HELinearFTCoord、HELinearFTGuest和HELinearFTHost，分别对应role为Coordinator、Guest和Host的情况。

| 协议名 | role:coordinator | role:guest | role:host| identity:server | identity:client |
| :----- | :----- | :----- | :----- | :----- | :----- |
|OT_INV| | | |OTINVServer|OTINVClient|
|SAL |SALCoord|SALGuest|SALHost| | |
|IV_FFS | |IVFFSGuest|IVFFSHost| | |
|HE_LINEAR_FT |HELinearFTCoord|HELinearFTGuest|HELinearFTHost| | |
|HE_OTP_LR_FT1 | |HELRGuest|HELRHost| | |
|HE_OTP_LR_FT2 |HEOTPLRCoord|HEOTPLRGuest|HEOTPLRHost| | |
|OTP_SA_FT |OTPSAFTCoord|OTPSAFTGuest|OTPSAFTHost| | |
|HE_SA_FT |HESAFTCoord|HESAFTGuest|HESAFTHost| | |
|HE_LR_FP |HELRFPCoord|HELRFPGuest|HELRFPHost|

# 示例
Host端生成FMC协议实例
```python
from flex.api import make_protocol
from flex.constants import FMC


federal_info = {
   "server":        "localhost:6001",
   "session": 
   {
       "role":      "host",
       "local_id":  "zhibang-d-014010",
       "job_id":    "test_job"
   },
   "federation": 
   {
       "host":          ["zhibang-d-014010"],
       "guest":         ["zhibang-d-014011"],
       "coordinator":   ["zhibang-d-014012"]
   }
}

algo_param = {
    "mpc_precision": 4
}
    
protocol = make_protocol(FMC, federal_info, None, algo_param)
```

Server端生成OT_INV协议实例
```python
from flex.api import make_protocol
from flex.constants import OT_INV


federal_info = {
    "server": "localhost:6001",
    "session": {
        "role": "host",
        "identity": 'server',
        "local_id": "zhibang-d-014010",
        "job_id": 'test_job_100'
    },
    "federation": {
        "host": ["zhibang-d-014010"],
        "guest": ["zhibang-d-014011"],
        "coordinator": ["zhibang-d-014012"]
    }
}

sec_param = {
    "symmetric_algo": "aes",
}

algo_param = {
    'n': 10,
    'k': 1
}

protocol = make_protocol(OT_INV, federal_info, sec_param, algo_param)
```

# 类方法
大多数协议的类方法为exchange，每个参与方协同执行exchange方法完成协议。exchange函数的输入参数根据不同的协议和角色而变化。有些协议存在不止一个方法，具体的调用方式请参考各协议的api使用文档。下表列出了各协议中不同角色的方法名称。

| 协议名 | role:coordinator | role:guest | role:host| identity:server | identity:client |
| :----- | :----- | :----- | :----- | :----- | :----- |
|OT_INV| | | |exchange|exchange|
|SAL |align, verify|align, verify|align, verify| | |
|IV_FFS | |exchange|exchange| | |
|HE_LINEAR_FT |exchange|exchange|exchange| | |
|HE_OTP_LR_FT1 | |exchange|exchange| | |
|HE_OTP_LR_FT2 |exchange|exchange|exchange| | |
|OTP_SA_FT |exchange|exchange|exchange| | |
|HE_SA_FT |exchange|exchange|exchange| | |
|HE_LR_FP |exchange|exchange|exchange|


每种协议的具体调用方式见相应的README文档：
* [OT_INV ](../flex/federated_sharing/invisible_inquiry/ot_inv/README.md)
* [SAL ](../flex/federated_sharing/sample_alignment/secure_alignment/README.md)
* [IV_FFS ](../flex/federated_preprocessing/federated_feature_selection/iv_ffs/README.md)
* [HE_LINEAR_FT ](../flex/federated_training/linear_regression/he_linear_ft/README.md)
* [HE_OTP_LR_FT1 ](../flex/federated_training/logistic_regression/he_otp_lr_ft1/README.md)
* [HE_OTP_LR_FT2 ](../flex/federated_training/logistic_regression/he_otp_lr_ft2/README.md)
* [OTP_SA_FT ](../flex/federated_training/secure_aggregation/otp_sa_ft/README.md)
* [HE_SA_FT ](../flex/federated_training/secure_aggregation/he_sa_ft/README.md)
* [HE_LR_FP ](../flex/federated_prediction/logistic_regression/he_lr_fp/README.md)
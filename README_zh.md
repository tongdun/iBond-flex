<div>

![FLEX logo](doc/pic/FLEX_logo.png)

<div>

[英文版链接](README.md)

FLEX(Federated Learning EXchange，FLEX)是同盾科技AI研究院为知识联邦体系设计并打造的一套标准化的联邦协议。
FLEX协议约定了联邦过程中参与方之间的数据交换顺序，以及在交换前后采用的数据加解密方法。只要参与各方能够遵守这些约定，
就可以安全地加入到联邦中提供数据或使用联邦服务。

FLEX协议包括两层：
1. 应用协议：这一层协议是面向联邦算法的，为联邦算法提供多方数据交换的应用支撑。
协议中会约定多方间数据交换的顺序和采用的具体密码算法。联邦过程中采用的通信协议也会被封装在这里。
2. 公共组件：是上层应用协议所依赖的基础密码算法和安全协议，比如同态加密、秘密分享等。

<div style="text-align: center;">
<br>

![FLEX协议总览](doc/pic/FLEX-structure-zh.png)

FLEX协议总览
</div>

本项目实现了FLEX白皮书中的应用协议和公共组件，其中通信接口使用同盾科技AI研究院自研的Ionic Bond协议接口，本项目仅给出了一种简单实现作为参考。

### 安装教程

FLEX协议可源码直接运行，支持Python3.6以上运行环境，并设置环境变量

```bash
export PYTHONPATH="/path/to/flex"
```

安装基本的依赖库，以Ubuntu系统为例：

```bash
apt install libgmp-dev, libmpfr-dev, libmpc-dev
pip install numpy, gmpy2, pycryptodome, scikit_learn, py_ecc, pandas
```

即可运行大部分的协议。

若需要使用基于安全多方计算相关的协议，还需安装：

```bash
pip install torch, tensorflow
```

用户也可通过FLEX中提供的工具将其安装到系统中.在源码目录运行：

```bash
pip install .
```

进行安装，然后可通过`from flex.api import *`来调用协议。

### 运行测试
FLEX提供了基本的测试代码，主要用于测试协议中的通信是否正常运行。通常情况下，用户需在三台主机上安装FLEX协议，分别扮演Coordinator, Guest, Host角色，用户需根据实际的主机hostname/ip修改federal_info，然后运行测试程序。FLEX也提供了单机模式，用于在一台机器上模拟测试协议运行。具体的测试说明见[test_intro](doc/test_intro.md)。

### API与文档
对于上层协议，通过统一的api进行调用，典型的流程是通过make_protocol得到协议实例，使用exchange方法来执行协议。以安全聚合为例：

```python
from flex.api import make_protocol
from flex.constants import OTP_SA_FT

# 初始化
protocol = make_protocol(OTP_SA_FT, federal_info, sec_param, algo_param)
# 执行
protocol.exchange(theta)
```

其中，federal_info为联邦参与方信息，sec_param为协议的安全参数，规定了协议中使用的密码方法、密钥长度等，algo_param为算法超参数，也可以为空。Theta是协议执行时的输入，具体的参数说明及协议使用请参考[api_intro](doc/api_intro.md)。

对于公共组件部分，通过flex.crypto中相应的模块api进行调用，如paillier同态加密算法的调用如下：

```python
from flex.crypto.paillier.api import generate_paillier_encryptor_decryptor

# 生成加密器和解密器
pe, pd = generate_paillier_encryptor_decryptor(n_length = 2048)
# 加密
en_x = pe.encrypt(x)
en_y = pe.encrypt(y)
# 求和
en_z = en_x + en_y
# 解密
z = pd.decrypt(en_z)
```

各公共组件的具体使用请参考[crypto_intro](doc/crypto_api_intro.md)。

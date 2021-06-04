# Secure Multiparty Computation Library
## 背景介绍
安全多方计算主要采用秘密分享(secret sharing: SS)技术，通过某种方法将秘密拆分，从N个信道同时发送，即使有信道存在恶意者，也无法恢复秘密。FLEX协议中提供的秘密分享支持任意数量参与方之间的秘密分享，还支持加法、乘法、点乘、比较等常用运算。

## 使用示例
### 多方计算配置

配置文件在config.conf中

```python
# Party数目，不包括Arbiter
NUM_PARTY = 2
# 域大小，目前只能是2^64
FIELD = 2**64
# 小数点精度，PRECISION=0时整数
PRECISION = 3
# 通信模块
IOMODULE = IBOND
```

初始化

```python
# On both parties and arbiter
import smpc
job_id = 'test_job_1'
mpc = smpc.SMPCTH('config.conf', job_id)
```

在安全多方计算协议中，每个参与方有自己的world_id。world_id的取值范围是0--NUM_PARTY-1。另外有一个Arbiter辅助参与计算，其world_id=NUM_PARTY。所有的计算方另外有role属性，party的role是PARTY，Arbiter的role是ARBITER。

```python
print('My world_id:', mpc.world_id)
print('My role:', mpc.role)
```

### 支持的函数

#### 原始数据 -> 秘密碎片

函数的输入输出如下

```python
def share_secrets(self, secret, field_size=None):
    """Share secret to other parties

    Arguments:
        secret: the secret to be shared
            for Arbiter: could be None
    Returns:
        [sh0, sh1, ... , shN]
            for Arbiter: None
    """
    pass

```

例子

```python
# On party
raw_data = ...
shares = mpc.share_secrets(raw_data)  # shares是一个长度为NUM_PARTY的列表

# On Arbiter
shares = mpc.share_secrets(None)  # shares是None
# 注意即使Arbiter没有任何操作也推荐调用和party的同样的函数，否则需要自行管理mpc.tag使得tag在所有参与方同步
```

#### 秘密碎片 -> 原始数据

函数的接口如下

```python
def reconstruct(self, x_sh):
    """Reconstruct from share of secret. It will not cutoff to field,
    so you need to do it yourself when needed.

    Args:
        x_sh: to be reconstructed
            for Arbiter: could be None
    Returns:
        x: the secret
            for Arbiter: None
    """
    pass
```

例子

```python
# On party
x_sh = shares[0]
x_raw = mpc.reconstruct(x_sh)

# On Arbiter
x_sh = None
x_raw = mpc.reconstruct(x_sh)
```

#### 点乘

函数接口

```python
def mul(self, x_sh, y_sh):
    """Elementwise multiply of x and y

    Arguments:
        x_sh: LHS of multiply
            for Arbiter: x_sh be zeros of the same size
        y_sh: RHS of multiply
            for Arbiter: y_sh be zeros of the same size

    Returns:
        share of mul(x,y)
            for Arbiter: None
    """
    pass
```

例子

```python
# On party
matrix = torch.randint(100, [100, 100])
shares = mpc.share_secrets(matrix)
x_sh, y_sh = shares
xy_sh = mpc.mul(x_sh, y_sh)

# On Arbiter
mpc.share_secrets(None)
x_sh = torch.zeros([100, 100])  # Arbiter初始化矩阵为同样大小
y_sh = x_sh
xy_sh = mpc.mul(x_sh, y_sh)
```

#### 矩阵乘

函数接口

```python
def matmul(self, x_sh, y_sh):
    """Elementwise multiply of x and y

    Arguments:
        x_sh: LHS of multiply
            for Arbiter: x_sh be zeros of the same size
        y_sh: RHS of multiply
            for Arbiter: y_sh be zeros of the same size

    Returns:
        share of mul(x,y)
            for Arbiter: None
    """
    pass
```

例子

```python
# On party
matrix = torch.randint(100, [100, 100])
shares = mpc.share_secrets(matrix)
x_sh, y_sh = shares
xy_sh = mpc.mul(x_sh, y_sh)

# On Arbiter
mpc.share_secrets(None)
x_sh = torch.zeros([100, 100])  # Arbiter初始化矩阵为同样大小
y_sh = x_sh
xy_sh = mpc.matmul(x_sh, y_sh)
```

#### 选择

函数接口

```python
def select_share(self, alpha_sh, x_sh, y_sh):
    """ Performs select share protocol
    If the bit alpha_sh is 0, x_sh is returned
    If the bit alpha_sh is 1, y_sh is returned

    Args:
        x_sh: the first share to select
        y_sh: the second share to select
        alpha_sh: the bit to choose between x_sh and y_sh
            for Arbiter: x_sh, y_sh, alpha_sh should be same shape

    Return:
        z_sh: such that
        z = (1 - alpha) * x + alpha * y
            for Arbiter: None
    """
    pass

```

例子
```python
# On party
matrix = torch.randint(100, [100, 100])
shares = mpc.share_secrets(matrix)
x_sh, y_sh = shares
xy_sh = mpc.mul(x_sh, y_sh)
alpha = torch.LongTensor([0])
alpha_sh = mpc.share_secret_from(alpha, 0)  # share from Party 0
ret = mpc.select_share(alpha_sh, x_sh, y_sh)

# On Arbiter
mpc.share_secrets(None)
x_sh = torch.zeros([100, 100])  # Arbiter初始化矩阵为同样大小
y_sh = x_sh
xy_sh = mpc.matmul(x_sh, y_sh)
mpc.share_secret_From(None, 0)
alpha_sh = torch.zeros([100, 100])
mpc.select_share(alpha_sh, x_sh, y_sh)
```

#### 比较

函数接口

```python
def relu_deriv(self, a_sh):
    """Compute a>=0

    Args:
        a_sh (AdditiveSharingTensor): the tensor of study
            for Arbiter: zero with the same shape
    Return:
        the converted share
            for Arbiter: None
    """
    pass
```

例子
```python
# On party
a = torch.arange(-100, 100).unsqueeze(-1)
a_sh = mpc.share_secret_from(a, 0)
ret_sh = mpc.relu_deriv(a_sh)

# On Arbiter
a_sh = torch.zeros([200, 1])
ret_sh = mpc.relu_deriv(a_sh)
```

#### 最小值

函数接口

```python
def minpool(self, x_sh):
    """Compute min of list of values

    Args:
        x_sh (AdditiveShareingTensor): the tensor of study
            for Arbiter: zero with the same shape

    Return:
        share of min value, index of this value in x_sh
            for Arbiter: None
    """
    pass
```

例子
```python
# On party
a = torch.arange(-100, 100).unsqueeze(-1)
a_sh = mpc.share_secret_from(a, 0)
ret_sh = mpc.minpool(a_sh)

# On Arbiter
a_sh = torch.zeros([200, 1])
ret_sh = mpc.minpool(a_sh)
```

#### Relu激活函数

函数接口

```python
def relu(self, a_sh):
    """Compute Reul(a)

    Args:
        a_sh (AdditiveSharingTensor): the tensor of study
            for Arbiter: zero with the same shape
    Return:
        the converted share
            for Arbiter: None
    """
    pass
```

例子
```python
# On party
a = torch.arange(-100, 100).unsqueeze(-1)
a_sh = mpc.share_secret_from(a, 0)
ret_sh = mpc.relu(a_sh)

# On Arbiter
a_sh = torch.zeros([200, 1])
ret_sh = mpc.relu(a_sh)
```

#### sigmoid激活函数

函数接口

```python
def sigmoid(self, a_sh):
    """Compute sigmoid(a_sh)
    by piecewise linear approximation

    Args:
        a_sh (AdditiveSharingTensor): the tensor of study
            for Arbiter: zero with the same shape
    Return:
        the converted share
            for Arbiter: None
    """
    pass
```

例子
```python
# On party
a = torch.arange(-100, 100).unsqueeze(-1)
a_sh = mpc.share_secret_from(a, 0)
ret_sh = mpc.sigmoid(a_sh)

# On Arbiter
a_sh = torch.zeros([200, 1])
ret_sh = mpc.sigmoid(a_sh)
```

### Lowlevel functions

share_zero_from

share_secret_from

broadcast_to_parties_from

broadcast_slice_to_parties_from

broadcast_to_arbiter

private_compare

private_compare_unsigned

_cutoff

msb

share_convert

### Application functions

logistic

multihead_sum_compare

multihead_max
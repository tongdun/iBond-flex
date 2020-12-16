# 安全伪随机数生成器
## 简介
联邦数据交换过程中经常需要参与方在一个共同种子的条件下，来生成大量的随机数。安全伪随机数生成器是满足该要求的一类算法，其生成的伪随机序列既满足统计学伪随机性，也满足密码学安全伪随机性，即不能以显著大于50%的概率在多项式时间内推算出序列的其它任何部分。

NIST.SP.800-90标准中规定了四种安全伪随机数生成算法Hash_DRBG，HMAC_DRBG，CTR_DRBG和Dual_EC_DRBG。目前FLEX协议中采用的是HMAC_DRBG方法，底层主要使用了Hash算法和Hmac算法。

* 依赖的运行环境
    1. hashlib
    2. hmac

## 类和函数
安全伪随机数生成器通过`flex.crypto.csprng.api`中的`generate_csprng_generator`函数来创建，调用`generate`方法来生成伪随机数，调用`reseed`方法来重置种子。

`generate_csprng_generator`定义如下：
```python
def generate_csprng_generator(entropy: Union[int, bytes], personalization_string: bytes = b"",
                              method=CRYPTO_HMAC_DRBG) -> CRYPTO_HMAC_DRBG
```
* 输入：
    * entropy: 伪随机数生成器的种子，可以是int或bytes
    * personalization_string: 额外的输入，与entropy一起影响伪随机串的生成，,类型为bytes，可以为空
    * method: 生成伪随机数的方法，目前支持CRYPTO_HMAC_DRBG
* 输出：
    * 当method为CRYPTO_HMAC_DRBG时，返回CRYPTO_HMAC_DRBG类实例
    
CRYPTO_HMAC_DRBG类提供两种类方法，如下：
```python
# 生成伪随机数串
def generate(self, num_bytes: int) -> bytes
# 重置种子
def reseed(self, entropy: bytes)
```
其中，generate用于生成指定长度的伪随机串，reseed是用于重置伪随机发生器。
```python
def generate(self, num_bytes: int) -> bytes
```
* 输入：
    * num_bytes: 返回的随机串长度，单位为byte
* 输出：
    * 字节串，长度为num_bytes
    
```python
def reseed(self, entropy: bytes)
```
* 输入：
    * entropy: 用于重置伪随机发生器的种子，用byte串表示。推荐当生成伪随机数次数超过2**48次后重置生成器
* 输出：
    * 无输出
    
## API调用
### 生成伪随机数生成器
```python

from flex.crypto.csprng.api import generate_csprng_generator
from flex.constants import *

# 生成伪随机数生成器实例
drbg = generate_csprng_generator(entropy=b'542435464554342576476747656736767657676545234546', personalization_string=b'', method=CRYPTO_HMAC_DRBG)

# 调用generate方法生成伪随机数串，返回长度为num_bytes的字节串
onetime_key = drbg.generate(num_bytes=2**16)
print(onetime_key)
```

### 重置种子
```python
# 通过entropy重置伪随机数生成器，当生成伪随机数次数超过2**48次后推荐重置生成器
drbg.reseed(b'e4243546455434c576476747656736767657676d4523454a')
# 调用generate方法生成新的伪随机数串
print(drbg.generate(num_bytes=2**7))
```
代码见：[example.py](../../../test/crypto/csprng/example.py)

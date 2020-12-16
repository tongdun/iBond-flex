# 格式保留加密算法
## 简介
格式保留加密是一种可以保证密文与明文具有相同的格式与长度的加密方式。常用于数据去标识化或脱敏。FLEX中的格式保留加密采用了NIST.SP.800-38G标准中的AES-FF1算法，将其当作一种随机置换算法，来生成布隆过滤器内的随机置换。为提高计算效率，选用了base为2的特例，来支持比特长度为64位以下的明文的加密。

* 依赖的运行环境
    1. Crypto
    2. numpy

## 类和函数
通过`flex.crypto.fpe.api`的`generate_fpe_encryptor`函数创建格式保留加密的一个实例，调用`encrypt`方法进行加密，调用`decrypt`方法进行解密。其中`encrypt`和`decrypt`支持批量的加解密。

`generate_fpe_encryptor`定义如下：
```python
def generate_fpe_encryptor(key: Union[int, bytes], n: int, t: Union[int, bytes] = b'', method: str = CRYPTO_FF1,
                           encrypt_algo: str = CRYPTO_AES) -> FF1Radix2Encryptor
```
* 输入：
    * key: 密钥，可以是int或bytes，支持三种长度：16，24和32，单位为字节
    * n: 输入和输出的最大比特长度，类型为int
    * t: tweak, 0 <= tweak <= maxTlen，类型为bytes，默认为空 
    * method: 格式保留加密的方法，目前支持CRYPTO_FF1
    * encrypt_algo: 采用的对称加密算法，目前支持CRYPTO_AES，即采用AES算法。
* 输出：
    * 当method为CRYPTO_FF1时，返回FF1Radix2Encryptor类实例
    
FF1Radix2Encryptor类提供两种类方法，如下：
```python
# 加密
def encrypt(self, x: Union[int, np.ndarray]) -> Union[int, np.ndarray]
# 解密
def decrypt(self, x: Union[int, np.ndarray]) -> Union[int, np.ndarray]
```
其中encrypt为加密函数，decrypt为解密函数。
```python
def encrypt(self, x: Union[int, np.ndarray]) -> Union[int, np.ndarray]
```
* 输入：
    * x: 要加密的明文，用十进制表示，可以是一个int，也可以是numpy.ndarray。注意x范围为0到2^n - 1
* 输出：
    * 密文，int或者numpy.ndarray，与x格式相同
    
```python
def decrypt(self, x: Union[int, np.ndarray]) -> Union[int, np.ndarray]
```
* 输入：
    * x: 要解密的密文，用十进制表示，可以是一个int，也可以是numpy.ndarray。注意x范围为0到2^n - 1
* 输出：
    * 密文，int或者numpy.ndarray，与x格式相同


## API调用
### 生成加（解）密器，加密与解密
```python
import numpy as np
from flex.crypto.fpe.api import generate_fpe_encryptor

# 设置密钥
key = b'1234567890123456'
# 表示输入和输出范围为0-(2**15-1)
n = 15
t = b''
# 生成格式保留加密加密器
encryptor = generate_fpe_encryptor(key, n, t, method='ff1', encrypt_algo='aes')

# 加密和解密单个数字
x = 32767
y = encryptor.encrypt(x)
z = encryptor.decrypt(y)

# 加密一组数字，支持numpy.ndarray
input = np.array([i for i in range(2**15)], dtype=np.uint32)
output = encryptor.encrypt(input)
```
代码见：[example.py](../../../test/crypto/fpe/example.py)

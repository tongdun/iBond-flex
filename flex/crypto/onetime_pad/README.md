# 一次一密
## 简介
一次一密(one-time pad)是一种理想的加密方案，该方案中每个消息用不同的密钥加密，每个密钥只使用一次。一次一密会使用乱码本记录一个大的不重复的真随机密钥集。实际应用中，发送方会对所发消息加密，然后销毁乱码本中用过部分。接收方有一个同样的乱码本，并依次使用乱码本上的每个密钥去解密密文，接收方在解密消息后销毁乱码本中用过的部分。新的消息则用乱码本的新的密钥加解密。

FLEX协议中使用一次一密时，会先将待加密的数值转换为整数，然后与密钥相加或相减，解密时需要再将整数转换为浮点数。

## 类和函数
通过`flex.crypto.onetime_pad.api`的`generate_onetime_pad_encryptor`函数创建一次一密的一个实例，用`encrypt`和`decrypt`来进行加密解密，用`decode`进行解码操作。

`generate_onetime_pad_encryptor`的定义如下：
```python
def generate_onetime_pad_encryptor(secret_key: Union[int, bytes]) -> OneTimePadEncryptor
```

* 输入：
    * secret_key: 用于生成一次一密的种子密钥
* 输出：
    * 返回OneTimePadEncryptor类实例
    
OneTimePadEncryptor提供`encrypt`和`decrypt`功能
```python
def encrypt(self, x: Union[list, np.ndarray, torch.Tensor], alpha: int = 1) -> OneTimePadCiphertext
```
* 输入：
    * x: 待加密的明文，可以是numpy.ndarray, torch.Tensor, 或者包含两者的嵌套list
    * alpha: 整数，默认为1，也可以是负数
* 输出：
    * 返回OneTimePadCiphertext实例，其密文值形式上满足x' + alpha*noise，其中x'是将x编码后得到的整数

```python
def decrypt(self, x: OneTimePadCiphertext, alpha: int = 1) -> Union[list, np.ndarray, torch.Tensor]
```
* 输入：
    * x: 待解密的密文，OneTimePadCiphertext格式
    * alpha: 整数，默认为1，也可以是负数
* 输出：
    * 返回与原始明文相同的格式，计算公式为首先计算x - alpha*noise，再对其解码
    
同一OneTimePadEncryptor实例可多次加密，每次加密将自动更新加密所需的随机数，当进行解密时，将使用上一次加密生成的随机数，不产生新的随机数。
    
## API调用
## 生成加（解）密器，加密与解密
```python
import numpy as np
import torch
from flex.crypto.onetime_pad.api import generate_onetime_pad_encryptor
from test.utils import almost_equal


secret_key = b'86da313426ac3e638'
# 生成加密器
encryptor = generate_onetime_pad_encryptor(secret_key)

x = [np.array([0.54356], dtype=np.float32),
     [torch.tensor([0.54356, 0.45345], dtype=torch.float32), np.array([[455432], [54352]], dtype=np.int32)]]
     
# 对x进行加密，alpha为2，得到的密文值为encode(x) + alpha * noise
en_x = encryptor.encrypt(x, alpha=2)

# 对密文en_x进行解密，alpha为2，得到decode(en_x - alpha * noise)
z = encryptor.decrypt(en_x, alpha=2)
assert almost_equal(x, z)
```

## 密文相加，解码
```python
# 生成加密器
encryptor2 = generate_onetime_pad_encryptor(secret_key)

y = [np.array([0.4325], dtype=np.float32),
     [torch.tensor([0.5434, 0.5631], dtype=torch.float32), np.array([[65643452], [542794]], dtype=np.int32)]]
     
# 对y进行加密，alpha为-2
en_y = encryptor2.encrypt(y, alpha=-2)

# 引文x，y加密使用同一个secret_key，且alpha分别为2和-2，两个密文的noise部分抵消，可直接对相加后的密文进行解码操作
z = (en_x + en_y).decode()

assertAlmostEqual(iterative_add(x, y), z)
```

## 密文相加，解密
```python
# 生成加密器
encryptor3 = generate_onetime_pad_encryptor(secret_key)

# 加密
en_y = encryptor3.encrypt(y, alpha=-1)

# 密文相加，并解密，这里en_x的alpha为2，en_y的alpha为-1，因此在解密时需将alpha设为1
z = encryptor.decrypt(en_x + en_y, alpha=1)

assertAlmostEqual(iterative_add(x, y), z)
```
代码见：[example.py](../../../test/crypto/onetime_pad/example.py)
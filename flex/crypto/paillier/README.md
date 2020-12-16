# Paillier加密库
## 背景介绍
同态加密(homomorphic encryption: HE)提供了在不泄漏原始数据内容的情况下，可以对加密数据进行处理的功能。同态加密根据加密算法不同可以分为加法同态（只支持加减运算），乘法同态（只支持乘除法运算）以及全同态加密（支持加减乘除，指数，对数等运算）。目前FLEX协议中支持Paillier半同态算法。

## 类和函数
通过`flex.crypto.paillier.api`的`generate_paillier_encryptor_decryptor`函数创建paillier同态加密的一个实例，用`encrypt`和`decrypt`来进行加密解密，用`parallel_ops`进行并行计算。

`generate_paillier_encryptor_decryptor`的定义如下：
```python
def generate_paillier_encryptor_decryptor(n_length: int = 1024, seed: int = None) -> (PaillierEncryptor, PaillierDecryptor)
```

* 输入：
    * n_length: 生成的密钥长度，目前支持1024，2048
    * seed：种子，同样的种子可以生成同样的密钥，默认为None，则随机生成公私钥
* 输出：
    * 返回PaillierEncryptor, PaillierDecryptor类实例
    
```python
PaillierEncryptor.encrypt(value: Union[np.ndarray, int, float], precision: int = None, random_value: int = None) -> Union[np.ndarray, PaillierEncryptedNumber]
```
* 输入：
    * value: 待加密的明文，支持int, float和numpy.ndarray
    * precision：编码的精度，可以设为None，则自动选择合适的值
    * random_value: 用于混淆
* 输出：
    * 返回PaillierEncryptor, PaillierDecryptor类实例

```python
PaillierDecryptor.decrypt(encrypted_number: Union[np.ndarray, PaillierEncryptedNumber]) -> Union[np.ndarray, int, float]
```
    
## API调用
## 生成加密器和解密器
```python
import numpy as np
from flex.crypto.paillier.api import generate_paillier_encryptor_decryptor

pe, pd = generate_paillier_encryptor_decryptor()

plain_param1 = np.random.random(100).astype(np.float32)
encrypted_param1 = pe.encrypt(plain_param1)
plain_param2 = pd.decrypt(encrypted_param1)
```

## 将加密器进行序列化并传输
```python
import pickle
s = pickle.dumps(pe)
pe = pickle.loads(s)
```

## 支持numpy.array的直接加密和解密（并自动多进程并发处理）
```python
plain_param1 = np.random.random((128, 4)).astype(np.float32)
encrypted_param1 = pe.encrypt(plain_param1)
plain_param2 = pd.decrypt(encrypted_param1)
```

## 密文计算的并行加速
```python
from flex.crypto.paillier import parallel_ops
pe, pd = generate_paillier_encryptor_decryptor()

x = np.random.random(size=(1000, 2))
y = np.random.randint(1000, size=(1000, 2))
en_x = pe.encrypt(x)
en_y = pe.encrypt(y)

en_result = parallel_ops.add(en_x, y)
result = pd.decrypt(en_result)
assertAlmostEqual(x+y, result)

en_result = parallel_ops.add(en_x, en_y)
result = pd.decrypt(en_result)
assertAlmostEqual(x+y, result)

en_result = parallel_ops.mul(en_x, y)
result = pd.decrypt(en_result)
assertAlmostEqual(x*y, result)
```

## 密文对象
* 经过加密器得到的是密文对象PaillierEncryptedNumber的实例（或者是实例的array）
* 密文对象通过运算符重载，支持的计算有（Paillier支持的）：
    * 密文对象 +|- 密文对象 -> 密文对象
    * 密文对象 +|- 标量    -> 密文对象
    * 密文对象 *|/ 标量    -> 密文对象

代码见：[example.py](../../../test/crypto/paillier/example.py)


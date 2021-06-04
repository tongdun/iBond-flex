# Paillier加密
## 简介
* 同态加密(homomorphic encryption: HE)提供了在不泄漏原始数据内容的情况下，可以对加密数据进行处理的功能。同态加密根据加密算法不同可以分为加法同态（只支持加减运算），乘法同态（只支持乘除法运算）以及全同态加密（支持加减乘除，指数，对数等运算）。目前FLEX协议中支持Paillier半同态算法。
	* 应用场景：不泄漏原始数据内容的情况下，需要对数据进行处理的情况。

## 类和函数
通过`flex.crypto.paillier.api`的`generate_paillier_encryptor_decryptor`函数创建paillier同态加密的一个实例：

| | Party |
| ---- | ---- |
| class | `PaillierEncryptor`, `PaillierDecryptor` |
| init | `n_length`, `seed` |
| method | `encrypt`, `decrypt` |

### 初始化参数
* `n_length`: 生成的密钥长度，目前支持1024，2048
* `seed`：种子，同样的种子可以生成同样的密钥，默认为None，则随机生成公私钥

如：

```python
pe, pd = generate_paillier_encryptor_decryptor(n_length: int = 1024, seed: int = None)
```

### 类方法
提供`encrypt`和`decrypt`方法进行加解密：

```python
PaillierEncryptor.encrypt(value: Union[np.ndarray, int, float], 
                          precision: int = None, random_value: int = None) -> Union[np.ndarray, PaillierEncryptedNumber]:
PaillierDecryptor.decrypt(encrypted_number: Union[np.ndarray, PaillierEncryptedNumber]) -> Union[np.ndarray, int, float]:
```

#### 输入
`encrypt`方法的输入为：

* `value`: 待加密的明文，支持int, float和numpy.ndarray
* `precision`：编码的精度，可以设为None，则自动选择合适的值
* `random_value`: 用于混淆

`decrypt`方法的输入为：

* `encrypted_number`：加密后的密文

#### 输出
`encrypt`方法的输出为加密后的密文

`decrypt`方法的输出为解密后的明文
    
### Paillier同态加密调用示例
#### 密文对象
* 经过加密器得到的是密文对象PaillierEncryptedNumber的实例（或者是实例的array）
* 密文对象通过运算符重载，支持的计算有（Paillier支持的）：
    * 密文对象 +|- 密文对象 -> 密文对象
    * 密文对象 +|- 标量    -> 密文对象
    * 密文对象 *|/ 标量    -> 密文对象

详见：[example.py](../../../test/crypto/paillier/example.py)


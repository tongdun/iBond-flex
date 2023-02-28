# 格式保留加密算法
## 简介
* 格式保留加密是一种可以保证密文与明文具有相同的格式与长度的加密方式。FLEX中的格式保留加密采用了NIST.SP.800-38G标准中的AES-FF1算法，将其当作一种随机置换算法，来生成布隆过滤器内的随机置换。为提高计算效率，选用了base为2的特例，来支持比特长度为64位以下的明文的加密。
	* 应用场景：常用于数据去标识化或脱敏。
	* 依赖的运行环境：
		1. Crypto
   		2. numpy

## 类和函数
通过`flex.crypto.fpe.api`的`generate_fpe_encryptor`函数创建格式保留加密的一个实例：调用`encrypt`方法进行加密，调用`decrypt`方法进行解密。其中`encrypt`和`decrypt`支持批量的加解密。

| | Party |
| ---- | ---- |
| class | `FF1Radix2Encryptor` |
| init | `key`, `n`, `t`, `method`, `encrypt_algo` |
| method | `encrypt`, `decrypt` |

### 初始化参数

* `key`: 密钥，可以是int或bytes，支持三种长度：16，24和32，单位为字节
* `n`: 输入和输出的最大比特长度，类型为int
* `t`: tweak, 0 <= tweak <= maxTlen，类型为bytes，默认为空 
* `method`: 格式保留加密的方法，目前支持`CRYPTO_FF1`
* `encrypt_algo`: 采用的对称加密算法，目前支持`CRYPTO_AES`，即采用AES算法。
如：

```python
encryptor = generate_fpe_encryptor(b'1234567890123456', 15, b'',
                                   method=CRYPTO_FF1, encrypt_algo=CRYPTO_AES)
```

### 类方法    
FF1Radix2Encryptor类提供两种类方法，如下：

```python
# encryot
def encrypt(self, x: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
# decrypt
def decrypt(self, x: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
```

其中encrypt为加密函数，decrypt为解密函数。

#### 输入
`encrypt`方法输入为：

* `x`: 要加密的明文，用十进制表示，可以是一个int，也可以是numpy.ndarray。注意x范围为0到2^n - 1

`decrypt`方法输入为：

* `x`: 要解密的密文，用十进制表示，可以是一个int，也可以是numpy.ndarray。注意x范围为0到2^n - 1

#### 输出
`encrypt`方法输出为：

* 密文，int或者numpy.ndarray，与x格式相同
    
`decrypt`方法输出为：

* 明文，int或者numpy.ndarray，与x格式相同

### 格式保留加密算法调用示例
详见：[example.py](../../../test/crypto/fpe/example.py)

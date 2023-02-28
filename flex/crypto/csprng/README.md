# 安全伪随机数生成器
## 简介
* 联邦数据交换过程中经常需要参与方在一个共同种子的条件下，来生成大量的随机数。NIST.SP.800-90标准中规定了四种安全伪随机数生成算法 `Hash_DRBG`， `HMAC_DRBG`， `CTR_DRBG`和`Dual_EC_DRBG`。目前FLEX协议中采用的是`HMAC_DRBG`方法。
	* 应用场景：参与方需要一个共同的种子的场景。
	* 相关技术：
		1. Hash算法
		2. Hmac算法
	* 安全要求：
		1. 生成的伪随机序列满足统计学伪随机性
		2. 满足密码学安全伪随机性，即不能以显著大于50%的概率在多项式时间内推算出序列的其它任何部分。
	* 依赖的运行环境：
   		1. hashlib
   		2. hmac

## 类和函数
安全伪随机数生成器通过`flex.crypto.csprng.api`中的`generate_csprng_generator`函数来创建：

| | Party |
| ---- | ---- |
| class | `HMacDRBG` |
| init | `entropy`, `personalization_string`, `method` |
| method | `generate`, `reseed` |

### 初始化参数

* `entropy`: 伪随机数生成器的种子，可以是int或bytes
* `personalization_string`: 额外的输入，与`entropy`一起影响伪随机串的生成，类型为bytes，可以为空
* `method`: 生成伪随机数的方法，目前支持`CRYPTO_HMAC_DRBG`
     
如：

```python
drbg = generate_csprng_generator(entropy=b'542435464554342576476747656736767657676545234546',
                                 personalization_string=b'', method=CRYPTO_HMAC_DRBG)
```

### 类方法
提供`generate`方法来生成伪随机数，`reseed`方法来重置种子，如下：

```python
# generate random string
def generate(self, num_bytes: int) -> bytes:
# reset seed
def reseed(self, entropy: bytes):
```

#### 输入
`generate`方法输入为：

* num_bytes: 返回的随机串长度，单位为byte
    
`reseed`方法输入为：

* entropy: 用于重置伪随机发生器的种子，用byte串表示。推荐当生成伪随机数次数超过2**48次后重置生成器

#### 输出
`generate`方法输出为字节串，长度为num_bytes.

`reseed`方法无输出
    
### 安全伪随机数生成器调用示例
详见：[example.py](../../../test/crypto/csprng/example.py)

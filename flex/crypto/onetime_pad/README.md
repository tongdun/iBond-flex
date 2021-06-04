# 一次一密
## 简介
* 一次一密(one-time pad)是一种理想的加密方案，该方案中每个消息用不同的密钥加密，每个密钥只使用一次。一次一密会使用乱码本记录一个大的不重复的真随机密钥集。实际应用中，发送方会对所发消息加密，然后销毁乱码本中用过部分。接收方有一个同样的乱码本，并依次使用乱码本上的每个密钥去解密密文，接收方在解密消息后销毁乱码本中用过的部分。新的消息则用乱码本的新的密钥加解密。FLEX协议中使用一次一密时，会先将待加密的数值转换为整数，然后与密钥相加或相减，解密时需要再将整数转换为浮点数。
	* 依赖的运行环境：
   		1. numpy==1.18.5
   		2. torch==1.6.0

## 类和函数
通过`flex.crypto.onetime_pad.api`的`generate_onetime_pad_encryptor`函数创建一次一密的一个实例：

| | Party |
| ---- | ---- |
| class | `OneTimePadEncryptor` |
| init | `secret_key` |
| method | `encrypt`, `decrypt` |

### 初始化参数
* `secret_key`: 用于生成一次一密的种子密钥

如：

```python
encryptor = generate_onetime_pad_encryptor(b'1' * 48)
```

### 类方法
OneTimePadEncryptor提供`encrypt`和`decrypt`功能

```python
def encrypt(self, x: Union[list, np.ndarray, torch.Tensor], alpha: int = 1) -> OneTimePadCiphertext:
def decrypt(self, x: OneTimePadCiphertext, alpha: int = 1) -> Union[list, np.ndarray, torch.Tensor]:
```

#### 输入
`encrypt`方法输入为：

* `x`: 待加密的明文，可以是numpy.ndarray, torch.Tensor, 或者包含两者的嵌套list
* `alpha`: 整数，默认为1，也可以是负数

`decrypt`方法输入为：

* `x`: 待解密的密文，OneTimePadCiphertext格式
* `alpha`: 整数，默认为1，也可以是负数

#### 输出
`encrypt`方法输出为：

* 返回OneTimePadCiphertext实例，其密文值形式上满足x' + alpha*noise，其中x'是将x编码后得到的整数

`decrypt`方法输出为：

* 返回与原始明文相同的格式，计算公式为首先计算x - alpha*noise，再对其解码

同一OneTimePadEncryptor实例可多次加密，每次加密将自动更新加密所需的随机数，当进行解密时，将使用上一次加密生成的随机数，不产生新的随机数。
    
## 一次一密调用示例
详见：[example.py](../../../test/crypto/onetime_pad/example.py)
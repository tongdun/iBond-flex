FLEX中公共组件主要包括联邦数据安全交换过程中使用到的基础性密码算法、工具及安全协议。目前公共组件部分支持如下算法、协议：

| 算法/协议名 | 算法/协议描述 | 
| :----- | :-----   |
|Paillier同态加密算法|支持Paillier算法的密文加、数乘密文操作|
|密钥交换协议|采用Diffile-Hellman密钥交换算法协商一个共同的随机数/密钥|
|安全伪随机数生成|输入种子密钥，多次生成指定长度的伪随机数序列|
|一次一密|使用一次一密加密方式，密文为y = encode(x) + alpha*noise, alpha为整数(正、负或零),支持密文的加法, 解码|
|格式保留加密|加密后密文与明文的空间相同，选用base为2的特例算法，支持比特长度为64位以下的明文的加密|
|不经意传输协议|发送方有n个消息，接收方想得到其中k个消息(1 <= k <= N)。协议保证发送方不能控制接收方的选择，发送方不知道接收方得到了哪几条消息，接收方也不能得到除了选择之外的其它消息。|

算法/协议的介绍见[Crypto README](../flex/crypto/README.md)

# 协议初始化
公共组件可通过crypto模块内对应的api进行调用，每种组件的调用方法如下：

## Paillier同态加密算法

```python
from flex.crypto.paillier.api import generate_paillier_encryptor_decryptor

pe, pd = generate_paillier_encryptor_decryptor(n_length = 1024, seed = None)
```

### 输入

* n_length

  密钥长度，支持1024，2048位，默认值为1024位

* seed

  种子，输入同样的种子可以生成同样的公钥，密钥，默认为None

### 示例
具体的API调用请参考[Paillier README](../flex/crypto/paillier/README.md)


## 密钥交换协议

```python
from flex.tools.ionic import commu
from flex.crypto.key_exchange.api import make_agreement

commu.init(conf)
secret_key = make_agreement(remote_id, key_size = 2048)
```

### 输入

* remote_id

  进行密钥交换的远程参与方的ID

* key_size

  密钥长度，支持2048, 3072, 4096, 6144, 8192位

### 示例
具体的API调用请参考[KeyExchange README](../flex/crypto/key_exchange/README.md)

## 安全伪随机数生成

```python
from flex.crypto.csprng.api import generate_csprng_generator
from flex.constants import *

drbg = generate_csprng_generator(entropy, personalization_string = b"", method=CRYPTO_HMAC_DRBG)
```

### 输入

* entropy

  用于初始化伪随机生成器的种子

* personalization_string

  附加的输入，与entropy一样用于初始化伪随机生成器，默认为b""

* method

  使用的随机生成方法，目前支持CRYPTO_HMAC_DRBG

### 示例
具体的API调用请参考[CSPRNG README](../flex/crypto/csprng/README.md)

## 一次一密

```python
from typing import Union
from flex.crypto.onetime_pad.api import generate_onetime_pad_encryptor

encryptor = generate_onetime_pad_encryptor(secret_key)
```

### 输入

* secret_key

  用于初始化一次一密算法

### 示例
具体的API调用请参考[OneTimePad README](../flex/crypto/onetime_pad/README.md)

## 格式保留加密

```python
from crypto.fpe.api import generate_fpe_encryptor

encryptor = generate_fpe_encryptor(key, n, t, method='ff1', encrypt_algo='aes')
```

### 输入

* key

  加密密钥，支持密钥长度为16，24，32字节

* n

  输入/输出的最大比特位长度（基数2）

* t

  tweak，t的长度在0到maxTlen之间

* method

  使用的格式保留加密方法，默认为ff1

* encrypt_algo

  使用的加密方法，默认为aes加密

### 示例
具体的API调用请参考[FPE README](../flex/crypto/fpe/README.md)

## 不经意传输协议

```python
from flex.crypto.oblivious_transfer.api import make_ot_protocol
from flex.tools.ionic import commu

commu.init(federal_info)
ot_protocol = make_ot_protocol(k, n, remote_id)
```

### 输入

* k

  n条信息中选择k条信息，目前支持k为1

* n

  server提供的信息数

* remote_id

  进行通信的远程参与方的ID

### 示例
具体的API调用请参考[ObliviousTransfer README](../flex/crypto/oblivious_transfer/README.md)

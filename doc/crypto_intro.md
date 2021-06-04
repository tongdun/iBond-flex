# `FLEX`公共组件
FLEX中公共组件主要包括联邦数据安全交换过程中使用到的基础性密码算法。目前公共组件部分支持如下算法（算法原理详见：[`Crypto README`](../flex/crypto/README.md)）：

|算法名 |算法描述 | 算法原理及使用示例| 
| :----- | :-----   | :----- |
|`Paillier`同态加密算法|支持`Paillier`算法的密文加、数乘密文操作| [`Paillier Readme`](../flex/crypto/paillier/README.md)|
|密钥交换协议|采用`Diffile-Hellman`密钥交换算法协商一个共同的随机数/密钥| [`DH Readme`](../flex/crypto/key_exchange/README.md)|
|安全伪随机数生成|输入种子密钥，多次生成指定长度的伪随机数序列| [`Csprng Readme`](../flex/crypto/csprng/README.md)|
|一次一密|使用一次一密加密方式，密文为```y = encode(x) + alpha*noise```, `alpha`为整数(正、负或零),支持密文的加法, 解码| [`OTP Readme`](../flex/crypto/onetime_pad/README.md)|
|格式保留加密|加密后密文与明文的空间相同，选用`base`为2的特例算法，支持比特长度为64位以下的明文的加密| [`FPE Readme`](../flex/crypto/fpe/README.md)|
|布隆过滤器:|使用`bloom`过滤器表示集合,支持过滤器内的置换、`&`和`==`操作| [`BF Readme`](../flex/crypto/id_filter/README.md)|
|不经意传输协议|发送方有n个消息，接收方想得到其中k个消息(`1 <= k <= N`)。协议保证发送方不能控制接收方的选择，发送方不知道接收方得到了哪几条消息，接收方也不能得到除了选择之外的其它消息。| [`OT Readme`](../flex/crypto/oblivious_transfer/README.md)|
|安全多方计算协议|无可信第三方的情况下，安全地计算一个约定函数| [`SMPC Readme`](../flex/crypto/smpc/README.md)|
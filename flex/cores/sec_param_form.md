## JSON for `sec_param` of FLEX
安全参数按照如下的形式(list中每一个元素是一个list)

```python
[['paillier', {'key_length':1024}], ['aes', {'key_length':128}]]
```

其中，list的第一个元素表示加密的方法，第二个元素表示该加密方法需要的参数。若不使用任何的加密方式，传输一个空的list即可。

现公共组件支持的加密方法和参数对支持形式如下

| 公共组件样例 | 参数说明 |
| :----- | :----- | 
|`['paillier', {'key_length': 1024}]`| `paillier`的`key_length`支持1024和2048
| `['aes', {'key_length': 128}]`| `aes`的`key_length`支持128，192和256
| `['sm4', {'key_length': 128}]`| `sm4`的`key_length`支持128
| `['secp256k1', {'key_length': 256}]`| `secp256k1`的`key_length`支持256
| `['md5', {}]`| 无
| `['sm3', {}]`| 无
| `['secret_sharing', {'precision': 3}]`|十进制小数点后有效数字
|`['onetime_pad', {'key_length': 512}]`|一次一密
|`['OT', {'n':10, 'k':1}]`|`OT`支持n选k(参数n可以为任意大于看的值，k现只支持1)

注：在一次安全协议中可以重复调用同一个公共组件，样例如：

```python
[['aes', {'key_length':2048}],]
```

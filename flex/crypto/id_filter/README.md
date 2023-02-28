# 布隆过滤器
## 简介
* 布隆过滤器(bloom filter: BF)，是一种空间效率很高的随机数据结构，以比特数组的形式表示，用来判断一个元素是否存在集合内。
	* 应用场景：布隆过滤器具有运行速度快、占用内存小的特点，因此常应用于海量数据的处理。
	* 相关技术：
		1. 格式保留加密算法
	* 依赖的运行环境：
   		1. Crypto
   		2. numpy

## 类和函数
通过`flex.crypto.id_filter.api`的`generate_id_filter`函数创建布隆过滤器的一个实例：
用`update`来更新过滤器中的用户ID集合，用`permute`和`inv_permute`进行置换，逆置换操作。

| | Party |
| ---- | ---- |
| class | `IDFilter` |
| init | `log2_bitlength`, `src_filter` |
| method | `update`, `permute`, `inv_permute`|

### 初始化参数

* `log2_bitlength`: 布隆过滤器长度的2的对数，支持大小为7到64
* `src_filter`: 若为None，则创建一个全零的过滤器；若不为None，则从该filter来创建过滤器。`src_filter`为numpy.ndarray，可以是np.bool类型和np.uint8类型。

如：

```python
generate_id_filter(log2_bitlength: int = 31, src_filter: Union[None, np.ndarray] = None)
```

### 类方法
IDFilter提供如下几种类方法：

```python
# update filter
def update(self, ids: Union[list, int]):
# permute filter
def permute(self, secret_key: bytes):
# inverse permute
def inv_permute(self, secret_key: bytes):
```

#### 输入
`update`方法输入为：

* `ids`：用于更新过滤器的ID列表

`permute`，`inv_permute`方法输入为：

* `secret_key`：用于置换及逆置换的密钥

#### 输出
`update`方法无输出

`permute`，`inv_permute`方法输出为置换，逆置换后的`filter`

### 布隆过滤器调用示例
详见：[example.py](../../../test/crypto/id_filter/example.py)
# 安全样本对齐(无第三方)
## 简介
* 基于`Blind-RSA`方案的去中心化的样本对齐协议，主要利用了RSA加密和Hash技术，其中一个参与方本地生成公私钥对，用私钥对本地数据加密，并将公钥发送给另一参与方加密数据，加密数据经过交互和再次加密后，另一参与方可以在密文上比较样本的交集，并通过索引回溯原始集合的交集，并将交集发送给有公私钥的参与方，双方返回相同的交集集合。
    
    * 应用场景: 
        * 双方样本集合大小在百万级别以下，或双方样本数量差距过大的情景;  
        
    * 相关技术: 
        * RSA公钥加密技术; 
        
    * 算法流程图  
        ![FLEX](../../../../doc/pic/sharing/RSA_SAL.png)
        
    * 安全要求: 
        * 参与方不能知道其它方的用户ID，也不能泄漏交集外的用户样本信息;
        
    * 依赖的运行环境
        * numpy>=1.18.4
        * crypto==1.4.1
        * gmpy2==2.0.8
        * rsa==4.7.2
        
    * 协议流程，详见: [FLEX白皮书](../../../../doc/FLEX白皮书.pdf)2.2.5章节
        
## 类和函数
`RSA_SAL`协议定义了两种类型的参与方，分别是`Guest`, `Host`，它们对应的类函数、初始化参数、类方法如下：

| | Guest | Host |
| ---- | ---- | ---- |
| class | `RSASalGuest`| `RSASalHost` |
| init | `federal_info`, `sec_param` | `fedral_info`, `sec_param` |
| method | `align` | `align` |

### 初始化参数
每种参与方在初始化时需要提供`federal_info`、`sec_param`和`algo_param`三种参数。其中`federal_info`提供了联邦中参与方信息，`sec_param`是协议的安全参数， `algo_param`是协议的算法参数。
  
* `sec_param`中需提供的参数有：
   * 使用`list`嵌套`list`形式存储加密信息，第一层`list`存储此次协议所有加密方式；第二层`list`的第一个元素表示加密的方法，第二个元素表示该加密方法需要用到的参数
 
   本协议中使用RSA加密，`sec_param`如下:
   
    ```python
    [['rsa', {'key_length': 2048}]]
    ```
  
* 本协议中不需要`algo_param`，所以提供`algo_param`为空

### 类方法
每个参与方均提供`align`方法，如下

```python
# Host
def align(self, ids: List, *args, **kwargs) -> List
# Guest
def align(self, ids: List, *args, **kwargs) -> List
```

#### 输入
参数意义如下：
* `ids`: 表示`Blind-RSA`安全对齐协议的参与方`Host`和`Guest`，需要安全对齐的样本列表，长度为样本的数量.

例如：

```python
ids = list(range(1000))
```

#### 输出
`Host`和`Guest`方的输出都为样本对齐后的`ids`。

### ecdh_sal调用示例

`Host`调用示例详见: [host.py](../../../../test/sharing/sample_alignment/rsa_sal/host.py)

`Guest`调用示例详见: [Guest.py](../../../../test/sharing/sample_alignment/rsa_sal/guest.py)



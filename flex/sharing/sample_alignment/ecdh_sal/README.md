# 安全样本对齐(无第三方)
## 简介
* 基于`ECDH`(`Elliptic-curve Diffie-Hellman`)方案的无第三方辅助的样本对齐协议，主要是利用了椭圆曲线加密技术以及Diffie-Hellman安全交换协议的思想，参与方本地生成各自密钥，并使用该密钥加密本方待对齐的样本ID，将密文发送至另一参与方，另一参与方用本地生成的密钥再次加密密文后发还给原参与方，各个参与方得到本地和对方加密两次的密文，在密文上做对比运算，按顺序提取本方样本集合的子集，从而得到按相同顺序排列的样本交集。
    * 应用场景: 
        * 双方样本集合大小在百万级别以下的情景。
        
    * 相关技术: 
        * Diffie-Hellman密钥交换协议;  
        * ECC椭圆曲线加密技术.  
        
    * 算法流程图  
        ![FLEX](../../../../doc/pic/sharing/ECDH_SAL.png)
        
    * 安全要求: 
        * 参与方不能知道其它方的用户ID，也不能泄漏交集外的用户样本信息;
        
    * 依赖的运行环境
        * numpy>=1.18.4
        * crypto==1.4.1
        * gmpy2==2.0.8
        * secrets==1.0.2
        
    * 协议流程，详见: [FLEX白皮书](../../../../doc/FLEX白皮书.pdf)2.2.4章节
        
## 类和函数
`ECDH_SAL`协议定义了两种类型的参与方，分别是`Guest`, `Host`，它们对应的类函数、初始化参数、类方法如下：

| | Guest | Host |
| ---- | ---- | ---- |
| class | `ECDHSalGuest`| `ECDHSalHost` |
| init | `federal_info`, `sec_param` | `fedral_info`, `sec_param` |
| method | `align` | `align` |

### 初始化参数
每种参与方在初始化时需要提供`federal_info`、`sec_param`和`algo_param`三种参数。其中`federal_info`提供了联邦中参与方信息，`sec_param`是协议的安全参数， `algo_param`是协议的算法参数。
  
* 本协议中未使用现有的加密方案，`sec_param`为空，本协议中不需要`algo_param`，所以提供`algo_param`为空

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
* `ids`: 表示`ECDH`安全对齐协议的参与方`Host`和`Guest`，需要安全对齐的样本列表，长度为样本的数量.

例如：

```python
ids = list(range(1000))
```

#### 输出
`Host`和`Guest`方的输出都为样本对齐后的`ids`。

### ecdh_sal调用示例

`Host`调用示例详见: [host.py](../../../../test/sharing/sample_alignment/ecdh_sal/host.py)

`Guest`调用示例详见: [Guest.py](../../../../test/sharing/sample_alignment/ecdh_sal/guest.py)



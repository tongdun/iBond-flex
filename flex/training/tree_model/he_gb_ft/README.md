# 跨特征树模型节点分裂协议

## 简介
跨特征场景下，寻找最优分裂点使得树节点在此次分裂达到的增益最大。

## 类和函数
`HE_GB_FT`协议定义了两种类型的参与方，分别是`Guest`(发起方)和`Host`(参与方)，它们对应的类函数、初始化参数、类方法如下：

| | `Guest` | `Host` | 
| ---- | ---- | ---- |
| `class` | `HEGBFTGuest` | `HEGBFTHost` | 
| `params` | `federal_info`, `sec_param`, `algo_param` | `federal_info`, `sec_param`, `algo_param`| 
| `methods` |`pre_exchange`, `exchange` | `pre_exchange`, `exchange` |

### 参数初始化
发起方和参与方在协议初始化时都需要提供`federal_info`, `sec_param`和`algo_param`三个初始化参数。其中，`federal_info`提供了联邦中参与方信息，`sec_param`提供了协议的安全参数信息，`algo_param`提供了协议的函数参数信息。

 * `sec_param`参数释义：
   * 使用`list`嵌套`list`形式存储加密信息，第一层`list`存储此次协议所有加密方式(树节点分裂协议只会用到一种加密协议)；第二层`list`的第一个元素表示加密的方法(树节点分类协议采用`paillier`加密)，第二个元素表示该加密方法需要用到的参数(`paillier`加密需要秘钥的长度`key_length`)
   
	  ```python
   	  [["paillier", {"key_length": 1024}],]
     ```
     
 * `algo_param`中需提供的参数有：
   * `min_leaf_samples`: 树节点分裂叶子结点最小样本数目；
   * `lambda_`: 控制树复杂度超参；
   * `gain`: 计算增益的方式(模型支持两种：`gini` 基尼系数计算增益，`grad_hess` 通过一阶、二阶导计算增益)；
	
   		```python
    	{
        	"min_num_samples": 50,
        	"lambda_": 0.01
    	}
   	```
   
### 类方法
每种参与方均提供`pre_exchange`和`exchange`方法，如下

```python
# HEGBFTGuest
pre_exchange(self, label: np.ndarray)
exchange(self, data: Dict, is_category: Optional[Dict], *args, **kwargs)
 
# HEGBFTHost
pre_exchange(self, *args, **kwargs)
exchange(self, data: Dict, is_category: Optional[Dict], *args, **kwargs)
```

#### 输入
* `pre_exchange`的参数意义：
	* `label`：`guest`输入明文的标签信息，`host`无此输入

* `exchange`的参数意义：
	* `data`：每一个特征在此轮的计数(`count`)，一阶导(`grad`)和二阶导(`hess`)的信息
	* `is_category`：特征是否为类别型特征判断信息

#### 输出
* `pre_exchange`的输出：`guest`方无输出，`host`方输出标签的加密信息

* `exchange`的参数意义：
	* `max_gain`：最优的增益(`guest`直接输出，使用特征为该`host`则返回否则返回`None`)
	* `party_id`: 最优分割`IP`信息
	* `best_feature`: 最优分割特征(`guest`直接输出，使用特征为该`host`则返回否则返回`None`)
	* `best_split_point`: 最优分割特征的分裂点信息(`guest`直接输出，使用特征为该`host`则返回否则返回`None`)
	* `weight`: 父节点的权重(`guest`直接输出，使用特征为该`host`则返回否则返回`None`)


### 调用示例
### `HE_GB_FT`调用示例
`Host`(参与方)调用示例详见：[host.py](../../../../test/training/tree_model/he_gb_ft/host.py)

`Guest`(发起方)调用示例详见: [guest.py](../../../../test/training/tree_model/he_gb_ft/guest.py)


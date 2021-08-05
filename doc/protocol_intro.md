<h1><code>FLEX</code>支持协议</h1>
FLEX中目前支持多种类型的应用协议，包括联邦共享、联邦预处理、联邦计算、联邦训练和联邦预测5个大类，具体的协议如下表所示：
<table>
	<tr>
	    <th>协议类型</th>
	    <th>协议名</th>
	    <th>协议描述</th>
	    <th>协议原理及示例</th>
	</tr>
	<tr>
	    <td rowspan="3">联邦共享</td>
	    <td><code>OT-INV</code></td>
	    <td>匿踪查询：解决查询过程中如何保护查询请求方用户ID信息不为其它参与方所知</td>
	    <td><a href="../flex/sharing/invisible_inquiry/ot_inv/README.md" target="_blank">Readme</a></td>
	</tr>
	<tr>
	    <td><code>BF-SF</code></td>
	    <td>样本过滤：在数据样本量大时，快速滤除大量交集外样本，以保证后续样本对齐可以快速完成</td>
	    <td><a href="../flex/sharing/sample_alignment/sample_filtering/README.md" target="_blank">Readme</a></td>
	</tr>
	<tr>
	    <td><code>SAL</code></td>
	    <td>安全对齐：一种简单、高效的样本对齐协议，其目的是计算参与方之间的样本交集</td>
	    <td><a href="../flex/sharing/sample_alignment/secure_alignment/README.md" target="_blank">Readme</a></td>
	</tr>
	<tr>
	    <td rowspan="2">联邦预处理</td>
	    <td><code>HE-DT-FB</code></td>
	    <td>联邦决策树分箱：无标签的一方利用有标签一方的标签信息进行特征离散化处理</td>
	    <td><a href="../flex/preprocessing/binning/he_dt_fb/README.md" target="_blank">Readme</a></td>
	</tr>
	<tr>
	    <td><code>IV-FFS</code></td>
	    <td>信息价值特征选择：跨特征联邦场景下的信息价值的计算</td>
	    <td><a href="../flex/preprocessing/feature_selection/iv_ffs/README.md" target="_blank">Readme</a></td>
	</tr>
	<tr>
	    <td rowspan="3">联邦计算</td>
	    <td><code>HE-ML</code></td>
	    <td>多头共债：基于同态加密的多头风险计算</td>
	    <td><a href="../flex/computing/multi_loan/he_ml/README.md" target="_blank">Readme</a></td>
	</tr>
	<tr>
	    <td><code>SS-ML</code></td>
	    <td>多头共债：基于秘密分享的多头风险计算</td>
	    <td><a href="../flex/computing/multi_loan/ss_ml/README.md" target="_blank">Readme</a></td>
	</tr>
	<tr>
	    <td><code>OTP-STATS</code></td>
	    <td>联邦数据统计量：基于一次一密的联邦统计量计算</td>
	    <td><a href="../flex/computing/stats/otp_stats/README.md" target="_blank">Readme</a></td>	
	</tr>
	<tr>
	    <td rowspan="6">联邦训练</td>
	    <td><code>HE-LINEAR-FT</code></td>
	    <td>跨特征联邦线性回归训练：跨特征线性回归训练中计算损失函数</td>
	    <td><a href="../flex/training/linear_regression/he_linear_ft/README.md" target="_blank">Readme</a></td>
	</tr>
	<tr>
	    <td><code>HE-OTP-LR-FT1</code></td>
	    <td>跨特征联邦逻辑回归训练：跨特征逻辑回归训练中计算损失函数</td>
	    <td><a href="../flex/training/logistic_regression/he_otp_lr_ft1/README.md" target="_blank">Readme</a></td>
	</tr>
	<tr>
	    <td><code>HE-OTP-LR-FT2</code></td>
	    <td>跨特征联邦逻辑回归训练：跨特征逻辑回归训练中计算损失函数</td>
	    <td><a href="../flex/training/logistic_regression/he_otp_lr_ft2/README.md" target="_blank">Readme</a></td>
	</tr>
	<tr>
	    <td><code>OTP-NN-FT</code></td>
	    <td>跨特征联邦神经网络训练：跨特征神经网络参数更新</td>
	    <td><a href="../flex/training/neural_network/otp_nn_ft/README.md" target="_blank">Readme</a></td>
	</tr>
	<tr>
	    <td><code>OTP-SA-FT</code></td>
	    <td>安全聚合：支持一次一密和同态加密的安全聚合</td>
	    <td><a href="../flex/training/secure_aggregation/otp_sa_ft/README.md" target="_blank">Readme</a></td>
	</tr>
	<tr>
	    <td><code>HE-FM-FT</code></td>
	    <td>跨特征联邦因子分解机FM训练：跨特征因子分解机FM训练中参数更新</td>
	    <td><a href="../flex/training/factorization_machines/he_fm_ft/README.md" target="_blank">Readme</a></td>
	</tr>
	<tr>
		<td rowspan="2">联邦预测</td>
	    <td><code>HE-LR-FP</code></td>
	    <td>跨特征联邦逻辑回归预测：跨特征逻辑回归预测中间结果汇总</td>
	    <td><a href="../flex/prediction/logistic_regression/he_lr_fp/README.md" target="_blank">Readme</a></td>
	</tr>
	<tr>
	    <td><code>HE-FM-FP</code></td>
	    <td>跨特征联邦因子分解机FM预测：跨特征联邦因子分解机FM预测中间结果汇总</td>
	    <td><a href="../flex/prediction/factorization_machines/he_fm_fp/README.md" target="_blank">Readme</a></td>
	</tr>
</table>

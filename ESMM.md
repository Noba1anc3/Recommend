## Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate

[Alibaba Inc](https://dl.acm.org/doi/pdf/10.1145/3209978.3210104)

> Ma X, Zhao L, Huang G, et al. Entire space multi-task model: An effective approach for estimating post-click conversion rate[C]//The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2018: 1137-1140. 

### Abstract

传统转化率预测模型使用点击过的展现样本来训练，但在所有可展现的完整空间上进行推理。这导致了样本选择偏差（SSB）问题。此外，数据稀疏（DS）问题导致模型难以训练。

> Conventional CVR models are trained with samples of clicked impressions while utilized to make inference on the entire space with samples of all impressions. This causes a sample selection bias problem. Besides, there exists an extreme data sparsity problem, making the model fitting rather difficult.  

本文中作者在全新的角度建模转化率：充分利用**用户行为序列模式，例如：展现 -> 点击 -> 转化**

> In this paper, we model CVR in a brand-new perspective by making good use of sequential pattern of user actions, i.e., impression → click → conversion. 

ESMM能解决SSB和DS问题：

- **在完整空间上直接建模转换率**
- **应用表征迁移学习策略**

> The proposed Entire Space Multi-task Model (ESMM) can eliminate the two problems simultaneously by i) modeling CVR directly over the entire space, ii) employing a feature representation transfer learning strategy. 

### Introduction

#### Background

转换率建模指的是估计点击后转换的任务，例如**pCVR = p (转换 | 点击，展现)**

> CVR modeling refers to the task of estimating the post-click conversion rate, i.e., pCVR = p(conversion|click,impression). 

![1652167046846](C:\Users\yi\AppData\Roaming\Typora\typora-user-images\1652167046846.png)

- 样本选择偏差问题：传统CVR的预估模型是在点击过的展现样本上训练，在推理时使用整个样本空间。训练样本和实际数据不服从同一分布，不符合机器学习中训练数据和测试数据同分布的设定。直观的说，会产生转化的用户不一定都是进行了点击操作的用户，如果只使用点击后的样本来训练，会导致CVR学习有偏。

- 数据稀疏问题：点击样本在整个样本空间中只占了很小一部分，而转化样本更少，高度稀疏的训练数据使得模型的学习变得相当困难。 

  > ii) data sparsity (DS) problem. In practice, data gathered for training CVR model is generally much less than CTR task.  

#### Methods

针对以上问题，目前有一些解决方法：

- **过采样方法**[8]复制了罕见的类示例，这有助于减轻数据的稀疏性，但对采样率很敏感。

  > Oversampling method [8] copies rare class examples which helps lighten sparsity of data but is sensitive to sampling rates. 

- **All Missing As Negative**（AMAN）采用随机抽样策略，选择未点击的印象作为负面示例[5]。通过引入未观察到的例子，它可以在一定程度上消除SSB问题，但会导致持续低估的预测。

  > All Missing As Negative (AMAN) applies random sampling strategy to select un-clicked impressions as negative examples [5]. It can eliminate the SSB problem to some degree by introducing unobserved examples, but results in a consistently underestimated prediction. 

- **无偏方法**[7]解决了CTR建模中的SSB问题，方法是通过拒绝采样来拟合观测值的真实潜在分布。然而，在按拒绝概率的除法对样本加权时，可能会导致数值的不稳定。

  > Unbiased method [7] addresses SSB problem in CTR modeling by fitting the truly underlying distribution from observations via rejection sampling. However, it might encounter numerical instability when weighting samples by division of rejection probability. 

#### Summary Up

在ESMM中，引入了后展现点击率![](https://latex.codecogs.com/svg.image?pCTR )和后展现点击转换率![](https://latex.codecogs.com/svg.image?pCTCVR )预测这两个辅助任务。ESMM没有直接用点击的展现样本训练CVR模型，而是**将pCVR视作一个中间变量，乘以pCTR等于pCTCVR**。![](https://latex.codecogs.com/svg.image?pCTCVR )和![]( https://latex.codecogs.com/svg.image?pCTR )都是通过所有展现的样本在整个空间内进行估计的，因此得到的pCVR也适用于整个空间。这表明SSB问题已被解决。此外，CVR网络的特征表示参数与CTR网络共享。后者用更丰富的样本进行训练。这种参数迁移学习有助于显著缓解DS问题。

> In ESMM, two auxiliary tasks of predicting the post-view click-through rate (CTR) and post-view click through&conversion rate (CTCVR) are introduced. Instead of training CVR model directly with samples of clicked impressions, ESMM treats pCVR as an intermediate variable which multiplied by pCTR equals to pCTCVR. Both pCTCVR and pCTR are estimated over the entire space with samples of all impressions, thus the derived pCVR is also applicable over the entire space. It indicates that SSB problem is eliminated. Besides, parameters of feature representation of CVR network is shared with CTR network. The latter one is trained with much richer samples. This kind of parameter transfer learning helps to alleviate the DS trouble remarkably. 

### The Proposed Approach

#### Notation

点击后CVR建模是为了估计![]( https://latex.codecogs.com/svg.image?pCTR&space;=&space;p(y&space;=&space;1|x) )

> Post-click CVR modeling is to estimate the probability of pCVR = p(z = 1|y = 1, x). 

- ![]( https://latex.codecogs.com/svg.image?pCTR&space;=&space;p(y&space;=&space;1|x) )

- ![]( https://latex.codecogs.com/svg.image?pCVR&space;=&space;p(z&space;=&space;1|y&space;=&space;1,&space;x) )

- ![]( https://latex.codecogs.com/svg.image?pCTCVR&space;=&space;p(y&space;=&space;1,&space;z&space;=&space;1|x) )

  其中，![]( https://latex.codecogs.com/svg.image?z )和![]( https://latex.codecogs.com/svg.image?y )分别代表转化和点击

![1652168444962](C:\Users\yi\AppData\Roaming\Typora\typora-user-images\1652168444962.png)

#### CVR Modeling and Challenges

ESMM模型结构如下所示：

![1652168583701](C:\Users\yi\AppData\Roaming\Typora\typora-user-images\1652168583701.png)

- SSB

  传统CVR建模通过引入辅助特征空间![]( https://latex.codecogs.com/svg.image?X_c )基于如下近似进行: ![](https://latex.codecogs.com/svg.image?p(z=1|y=1,x))≈![]( https://latex.codecogs.com/svg.image?q(z=1|x_c) ).  ![]( https://latex.codecogs.com/svg.image?\forall&space;x_c\in&space;X_c ), 存在![]( https://latex.codecogs.com/svg.image?(x=x_c,y_x=1) ), 其中![]( https://latex.codecogs.com/svg.image?x\in&space;X ), ![]( https://latex.codecogs.com/svg.image?y_x )是![]( https://latex.codecogs.com/svg.image?x )的点击标签。在推理阶段，对![](https://latex.codecogs.com/svg.image?p(z=1|y=1,x))的计算基于![]( https://latex.codecogs.com/svg.image?q(z=1|x_c) )进行，其假设是对任意的![]( https://latex.codecogs.com/svg.image?(x,y_x=1) ), 都有![]( https://latex.codecogs.com/svg.image?x\in&space;X_c ). 然而，该假设有很大概率会被违反，其原因在于![]( https://latex.codecogs.com/svg.image?X_c )只是![]( https://latex.codecogs.com/svg.image?X )的一小部分。

  **它受到很少发生的点击事件的随机性的严重影响，其概率在展现空间的不同区域有所不同。此外，当没有足够多的观测数据时，![]( https://latex.codecogs.com/svg.image?X_c)样本空间可能与![]( https://latex.codecogs.com/svg.image?X)有很大差异。**

  > It is affected heavily by the randomness of rarely occurred click event, whose probability varies over regions in space ![]( https://latex.codecogs.com/svg.image?X). Moreover, without enough observations in practice, space ![]( https://latex.codecogs.com/svg.image?X_c ) may be quite different from ![]( https://latex.codecogs.com/svg.image?X).  

  这会导致训练样本的数据分布与真实数据分布产生便宜，影响CVR建模的泛化性。

   This would bring the drift of distribution of training samples from truly underling distribution and hurt the generalization performance for CVR modeling.  

- DS

  表一表明**CVR任务的可用样本只有CTR任务的4%**

  > Table 1 shows the statistics of our experimental datasets, where number of
  > samples for CVR task is just 4% of that for CTR task.  

  ![1652170826655](C:\Users\yi\AppData\Roaming\Typora\typora-user-images\1652170826655.png)

  另一个CVR建模的挑战是**延迟反馈**。

  > It is worth mentioning that there exists other challenges for CVR modeling, e.g. delayed feedback [1].   
  >

#### Entire Space Multi-Task Model

- Modeling over entire space

  ![]( https://latex.codecogs.com/svg.image?p(z=1|x,y=1)=\frac{p(z=1,y=1|x)}{p(y=1|x)} )

  该算式表明，拥有![]( https://latex.codecogs.com/svg.image?pCTR)和![]( https://latex.codecogs.com/svg.image?pCTCVR)的估计之后，我们可用基于整个输入空间样本得到![]( https://latex.codecogs.com/svg.image?pCVR)，这直接解决了SSB问题。

  > Eq.(2) tells us that with estimation of pCTCVR and pCTR, pCVR can be derived over the entire input space X, which addresses the sample selection bias problem directly.  

  然而，![]( https://latex.codecogs.com/svg.image?pCTR)是一个很小的数字，除以![]( https://latex.codecogs.com/svg.image?pCTR)可能导致数值不稳定性。ESMM使用乘法的形式避免该问题。在ESMM中，![]( https://latex.codecogs.com/svg.image?pCVR)只是一个中间变量，![]( https://latex.codecogs.com/svg.image?pCTR)和![]( https://latex.codecogs.com/svg.image?pCTCVR)是ESMM模型在完整空间中估计的主要因素。

  > However, pCTR is a small number practically, divided by which would arise numerical instability. ESMM avoids this with the multiplication form. In ESMM, pCVR is just an intermediate variable which is constrained by the equation of Eq.(1). pCTR and pCTCVR are the main factors ESMM actually estimated over entire space.  

  这确保了估计出![]( https://latex.codecogs.com/svg.image?pCVR)的在0到1的范围之内，**而在基于除法的方法当中，该值可能超过1.**

  > it ensures the value of estimated pCVR to be in range of [0,1], which in DIVISION method might exceed 1.  

  ESMM模型的损失函数如下，它包含了来自CTR任务和CTCVR任务的损失值。

  ![]( https://latex.codecogs.com/svg.image?L(\theta&space;_{ctr},&space;\theta&space;_{cvr})=\sum_{i=1}^{n}l(y_i,&space;f(x_i;\theta&space;_{ctr})%29&plus;\sum_{i=1}^{n}l(y_i&space;\&&space;z_i,f(x_i;\theta&space;_{ctr})f(x_i;\theta&space;_{cvr})%29 )

- Feature representation transfer

  参数共享机制使CVR网络能够从未发生点击的展现中学习，并在减轻数据稀疏问题中提供了很大的帮助。

  > This parameter sharing mechanism enables CVR network in ESMM to learn from un-clicked impressions and provides great help for alleviating the data sparsity trouble. 

### Experiments

#### Experimental Setup

##### Data

收集了淘宝推荐系统的流量日志，随机选取1%样本作为公开数据集，整个数据集为产品数据集。

> We collect traffic logs from Taobao’s recommender system and release a 1% random sampling version of the whole dataset, whose size still reaches 38GB (without compression). We refer to the released dataset as Public Dataset and the whole one as Product Dataset. 

**按时间对数据集进行划分**，前一半数据为训练集，后一半数据为测试集。

> Both of the two tasks split the first 1/2 data in the time sequence to be training set while the rest to be test set.  

##### Methods

- Base Model
  - ESMM模型图左侧的CVR结构
- AMAN
  -  采用负采样策略，采样比例设置为10%，20%，50%和100% 
- 过采样
  -  复制正样本来减少训练数据的稀疏性 
- 无偏
  - 基于拒绝采样拟合真实潜在分布，pCTR作为拒绝概率
- 基于除法的ESMM
  - 预测pCTR和pCTCVR，然后相除得到pCVR
- ESMM-NS
  - CVR与CTR部分不共享嵌入层参数
- ESMM

##### Hyper-parameters

- ReLU激活函数
- 嵌入向量维度：18
- 多层感知机输出维度：360，200，80，2
- Adam优化器（β1=0.9, β2 = 0.999, ϵ = 10−8)
- 指标：AUC

#### Results on Public Dataset

![1652174547773](C:\Users\yi\AppData\Roaming\Typora\typora-user-images\1652174547773.png)

只有AMAN模型在CVR任务上表现的低于BASE模型。

过采样和无偏方法在CVR和CTCVR任务上均表现的好于BASE模型。

#### Results on Product Dataset

**随着训练样本量的增加，所有方法都有所改进。这表明了数据稀疏性的影响。**

> all methods show improvement with the growth of volume of training samples. This indicates the influence of data sparsity.  

![1652186839388](C:\Users\yi\AppData\Roaming\Typora\typora-user-images\1652186839388.png)

通过对整个数据集的训练，ESMM在CVR任务上获得2.18%的绝对AUC增益，在CTCVR任务上获得2.32%的绝对AUC增益。**对于AUC增加0.1%的工业应用来说，这是一个显著的改进。**

> Trained with the whole dataset, ESMM achieves absolute AUC gain of 2.18% on CVR task and 2.32% on CTCVR task over BASE model. This is a significant improvement for industrial applications where 0.1% AUC gain is remarkable.  
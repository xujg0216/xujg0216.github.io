# triplet loss


## 1. database初始化

为每个`queries`通过`KNN`计算`hard_positive`， `soft_positive`

## 2. epoch内分批加载,构造子数据集

在每个epoch中， 一共随机`sample`5000个`queries`， 为避免一次性加载过多样本导致内存不足， 通过`cache_refresh_rate`分批加载数据。
```python
sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False)
```

`queries_num`为查询样本的总数量， 从中随机选择`cache_refresh_rate`一次性加载， 同时不允许重复抽取相同的查询样本。例如`cache_refresh_rate`为1000。

随机从database中选择`neg_samples_num`个为负样本 , 例如也为1000
```python
sampled_database_indexes = np.random.choice(self.database_num, self.neg_samples_num, replace=False)
positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes]

positives_indexes = [p for pos in positives_indexes for p in pos]
```

找到随机的`queries`对应的`positive`， 将`positive_idx`与负样本加载在一起， 组成三元组。
```python
database_indexes = list(sampled_database_indexes) + positives_indexes

database_indexes = list(np.unique(database_indexes))

subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
```

将一个cache的anchor， positive, negative组成一个子数据集， 计算这些样本的特征。储存在cache内
```python
subset_ds = Subset(self, positives_indexes + list(sampled_queries_indexes + self.database_num))

cache = self.compute_cache(args, model, subset_ds, (len(self), args.features_dim))
```
调用子数据集进行样本的特征表示。
```python
def compute_cache(args, model, subset_ds, cache_shape):

	"""Compute the cache containing features of images, which is used to

	find best positive and hardest negatives."""

	subset_dl = DataLoader(dataset=subset_ds, num_workers=args.num_workers,
	
	batch_size=args.infer_batch_size, shuffle=False,
	
	pin_memory=(args.device == "cuda"))
	
	model = model.eval()
	
	# RAMEfficient2DMatrix can be replaced by np.zeros, but using
	
	# RAMEfficient2DMatrix is RAM efficient for full database mining.
	
	cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32)
	
	with torch.no_grad():
	
	for images, indexes in tqdm(subset_dl, ncols=100):
	
	images = images.to(args.device)
	
	features = model(images)
	
	cache[indexes.numpy()] = features.cpu().numpy()
	
	return cache
```
>cache内存储了1000个queries， 1000个负样本（有可能部分为正样本）， 以及这1000个queries对应的正样本， 经过网络模型的特征表示

然后计算`best_positive`和`negative`的索引
* `best_positive`是将`queries`的`positive`与模型的输出通过`faiss`计算
* `negative`是随机从`1000`个负样本中选择的， 但是`soft_positive`之外的`negs_num_per_query`个
以上就组成了一个`queries`的全局索引`triplets_global_indexes`， 由1个`anchor`, 1个`positive`， 10个`negative`组成。
一个子数据集的全局索则为`[1000, 12]`

`__getitem__`中则为：
```python
def __getitem__(self, index):

	if self.is_inference:
	
	# At inference time return the single image. This is used for caching or computing NetVLAD's clusters
	
		return super().__getitem__(index)

	query_index, best_positive_index, neg_indexes = torch.split(self.triplets_global_indexes[index], (1, 1, self.negs_num_per_query))
	
	query = self.query_transform(path_to_pil_img(self.queries_paths[query_index]))
	
	positive = self.resized_transform(path_to_pil_img(self.database_paths[best_positive_index]))
	
	negatives = [self.resized_transform(path_to_pil_img(self.database_paths[i])) for i in neg_indexes]
	
	images = torch.stack((query, positive, *negatives), 0)
	
	triplets_local_indexes = torch.empty((0, 3), dtype=torch.int)
	
	for neg_num in range(len(neg_indexes)):
	
		triplets_local_indexes = torch.cat((triplets_local_indexes, torch.tensor([0, 1, 2 + neg_num]).reshape(1, 3)))
	
	return images, triplets_local_indexes, self.triplets_global_indexes[index]
```

## 3. 模型前向传播与损失计算

对一个`batch`内的1个`anchor`, 1个`positive`， 10个`negative`进行模型推理

一个`batch`调用推理后， 计算损失。

```python
criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")
```

1. **`nn.TripletMarginLoss`**:
    
    - 这是 PyTorch 中的一个内置损失函数，用于计算三元组损失。它的目标是通过最小化三元组的损失，拉近正样本之间的距离，推远负样本之间的距离。
2. **`margin=args.margin`**:
    
    - `margin` 是三元组损失中的一个重要参数，表示正样本和负样本之间的最小边距。具体来说，损失函数会强制正样本（`anchor` 和 `positive`）之间的距离小于负样本（`anchor` 和 `negative`）之间的距离加上这个 `margin` 的值。
    - 换句话说，如果负样本距离比正样本距离更小，超过了 `margin`，损失将会增大，促使模型进行调整。
    - `args.margin` 是一个可调的超参数，决定了这个边距的大小。
3. **`p=2`**:
    
    - `p` 参数指定了距离度量的类型。`p=2` 表示使用欧氏距离（Euclidean distance）来计算样本之间的距离。如果你将 `p` 设置为 1，则会使用曼哈顿距离（Manhattan distance）。
    - 在深度学习中，欧氏距离通常是最常用的距离度量方法，用于度量样本之间的相似性。
4. **`reduction="sum"`**:
    
    - `reduction` 参数指定了如何汇总每个三元组损失的值。这里的 `"sum"` 表示将所有的损失值相加。这意味着在计算损失时，会对每个三元组的损失进行求和。
    - 除了 `"sum"`，还有 `"mean"`（求平均）和 `"none"`（不做任何汇总，返回每个三元组的损失值）。通常，`"sum"` 会用于批量处理，`"mean"` 会在计算平均损失时使用。

三元组损失函数的目标是最小化以下的损失：

$L = \sum_{i=1}^{N} \max(d(a_i, p_i) - d(a_i, n_i) + \text{margin}, 0)$

其中：

- $d(a_i, p_i)$ 是锚点（anchor）样本与正样本（positive）之间的距离，
- $d(a_i, n_i)$ 是锚点（anchor）样本与负样本（negative）之间的距离，
- `margin` 是一个可调的边距超参数，
- $N$ 是三元组的数量。

`loss`为
```python
loss = criterion_triplet(anchor, positive, negative)
```



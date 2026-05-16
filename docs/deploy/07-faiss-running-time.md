```python
from time import time
import faiss
from tqdm import tqdm
import numpy as np
```


```python
database_input = np.random.randn(30000, 4096)
max_recall_values = 1
faiss_index = faiss.IndexFlatL2(4096)
faiss_index.add(database_input)
```


```python
faiss_index
```




    <faiss.swigfaiss_avx512.IndexFlatL2; proxy of <Swig Object of type 'faiss::IndexFlatL2 *' at 0x76d74863c360> >




```python
t = []
for _ in tqdm(range(10000)):
    query_input = np.random.randn(1, 4096)
    t1 = time()
    distances, predictions = faiss_index.search(query_input, max_recall_values)
    t2 = time()
    t.append(t2 - t1)
print("Average faiss.IndexFlatL2 running time on cpu: ", sum(t)/10000)
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [02:04<00:00, 80.39it/s]

    Average faiss.IndexFlatL2 running time on cpu:  0.012288888859748841


    



```python
res = faiss.StandardGpuResources()
faiss_index = faiss.GpuIndexFlatL2(res, 4096)
faiss_index.add(database_input)
```


```python
faiss_index
```




    <faiss.swigfaiss_avx512.GpuIndexFlatL2; proxy of <Swig Object of type 'faiss::gpu::GpuIndexFlatL2 *' at 0x76d6bb652930> >




```python
t_gpu = []
for _ in tqdm(range(10000)):
    query_input = np.random.randn(1, 4096)
    t1 = time()
    distances, predictions = faiss_index.search(query_input, max_recall_values)
    t2 = time()
    t_gpu.append(t2 - t1)
print("Average faiss.IndexFlatL2 running time on RTX 4060: ", sum(t_gpu)/10000)
```

    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:21<00:00, 471.49it/s]

    Average faiss.IndexFlatL2 running time on RTX 4060:  0.002055554723739624


    


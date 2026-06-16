# Creating representations in Deep Isolated Forest manually by hand
Purpose: To deeply understand the underlying mechanism and algorithm used in the CERE step.  

## Step 1: Generate a random dataset
Because this exercise is being done by hand and aims to just build the model on a small scale and visualise the output, we limit the dataset to have 3 features each and 15 datapoints to remain computationally possible. Hence our dataset can be represented by a 15 x 3 matrix. Each element in the dataset is limited to integers between -50 and 50 inclusive, once again with the purpose to remain computationally simple.

We use np.random.randint to generate the dataset.

```python
import numpy as np
rng = np.random.default_rng(42)
dataset = rng.integers(-50, 51, size=(15, 3))
print(dataset)

# Output
[[-41  28  16]
 [ -6  -7  36]
 [-42  20 -30]
 [-41   3  48]
 [ 24  26  22]
 [ 29   1 -38]
 [ 34  -5   0]
 [-13 -32  43]
 [ 28  15 -10]
 [ 33   5  -6]
 [ -5 -28 -41]
 [  6  39 -44]
 [ 36  33 -23]
 [ 13 -34  26]
 [ 20 -15 -44]]
```

Hence, this is our dataset for this exercise.

## Step 2: Create iForest(X, t, v) --> Algorithm 1
Input variables
1. X - Input data
2. t - Number of trees
3. v - Sub-sampling size for each tree

The steps to create iForest are as follows:  
1. Initialise Forest
2. Set height limit = ceil(log_x(v))
3. for i = 1 to t:
   1. X' = sample(X, v)
   2. Initialise a iTree(X', 0, l) and add to the iForest
4. return iForest

For our exercise, I have set  
X = dataset  
t = 5
v = 4 (Deliberately chosen so that l = 2)


Here I have come up with the index for the sampled datapoints for each tree

```python
import numpy as np

rng = np.random.default_rng(42)
data_index = np.array([
    rng.choice(np.arange(15), size=4, replace=False)
    for _ in range(5)
])

print(data_index)

# Output
[[ 6  1  9 10]
 [ 8  2  1  7]
 [ 8  7 10  1]
 [ 4 11  2 12]
 [ 3 13  5  6]]
```

## Step 3: Create and train iTrees --> Algorithm 2
Input variables
1. X - Input data (for each tree)
2. e - current tree height
3. l - height limit

The steps to create iForest are as follows:  
1. Initialise tree
2. if e >= l or |X| <= 1:
   1. return exNode(Size <- |X|)
3. else
   1. let Q be a list of attributes in X
   2. randomly select a q from Q
   3. randomly select a split point p between min and max value of attribute q in X
   4. X_l <- filter(X, q < p)
   5. X_r <- filter(X, q >= p)
   6. return inNode{left <- iTree(X_l, e+1, l), right <- iTree(X_r, e+1, l), splitatt <- q, splitval <- p}

```python
for i in range(data_index.shape[0]):
    print(dataset[data_index[i, :]])
    print()

# Output
[[ 34  -5   0]
 [ -6  -7  36]
 [ 33   5  -6]
 [ -5 -28 -41]]

[[ 28  15 -10]
 [-42  20 -30]
 [ -6  -7  36]
 [-13 -32  43]]

[[ 28  15 -10]
 [-13 -32  43]
 [ -5 -28 -41]
 [ -6  -7  36]]

[[ 24  26  22]
 [  6  39 -44]
 [-42  20 -30]
 [ 36  33 -23]]

[[-41   3  48]
 [ 13 -34  26]
 [ 29   1 -38]
 [ 34  -5   0]]
```

Randomly selecting the feature, for first and second split.
```python
import numpy as np

rng = np.random.default_rng(42)
feature_select = rng.integers(0, 3, size=(5, 2))
print(feature_select)

# Output
[[0 2]
 [1 1]
 [1 2]
 [0 2]
 [0 0]]
```

## Training the trees
### **Tree 1**  
Training Dataset:  
[[ 34  -5   0]  
 [ -6  -7  36]  
 [ 33   5  -6]  
 [ -5 -28 -41]]  

| Layer  |  ExNode   |          X | SFeat | SValue | LNode | RNode |
| :---:  | :------:  | ---------: | ----: | ----:  | ----: | ----: |
| 0      |   False   | 0, 1, 2, 3 | 0     | 18     | 1, 3  | 0, 2  |
| 1,1    |   False   | 1, 3       | 2     | -13    | 3     | 1     |
| 1,2    |  False    | 0, 2       | 2     | -1     | 2     | 0     |
| 1,1,1  |  True     | 3          | -     | -      | -     | -     |
| 1,1,2  |  True     | 1          | -     | -      | -     | -     |
| 1,2,1  |  True     | 2          | -     | -      | -     | -     |
| 1,2,2  |  True     | 0          | -     | -      | -     | -     |

---

### **Tree 2**  
Training Dataset:  
[[ 28  15 -10]  
 [-42  20 -30]  
 [ -6  -7  36]  
 [-13 -32  43]]  

| Layer | ExNode |          X | SFeat | SValue | LNode | RNode |
| :---: | :----: | ---------: | ----: | -----: | ----: | ----: |
|   0   |  False | 0, 1, 2, 3 |     1 |     10 |  2, 3 |  0, 1 |
|  1,1  |  False |       2, 3 |     1 |    -20 |     3 |     2 |
|  1,2  |  False |       0, 1 |     1 |     17 |     0 |     1 |
| 1,1,1 |  True  |          3 |     - |      - |     - |     - |
| 1,1,2 |  True  |          2 |     - |      - |     - |     - |
| 1,2,1 |  True  |          0 |     - |      - |     - |     - |
| 1,2,2 |  True  |          1 |     - |      - |     - |     - |

---

### **Tree 3**  
Training Dataset:  
[[ 28  15 -10]  
 [-13 -32  43]  
 [ -5 -28 -41]  
 [ -6  -7  36]]  

| Layer | ExNode |          X | SFeat | SValue | LNode |   RNode |
| :---: | :----: | ---------: | ----: | -----: | ----: | ------: |
|   0   |  False | 0, 1, 2, 3 |     1 |    -30 |     1 | 0, 2, 3 |
|  1,1  |  True  |          1 |     - |      - |     - |       - |
|  1,2  |  False |    0, 2, 3 |     2 |     20 |  0, 2 |       3 |
| 1,2,1 |  True  |       0, 2 |     - |      - |     - |       - |
| 1,2,2 |  True  |          3 |     - |      - |     - |       - |

---

### **Tree 4**  
Training Dataset:  
[[ 24  26  22]  
 [  6  39 -44]  
 [-42  20 -30]  
 [ 36  33 -23]]  

| Layer | ExNode |          X | SFeat | SValue | LNode | RNode |
| :---: | :----: | ---------: | ----: | -----: | ----: | ----: |
|   0   |  False | 0, 1, 2, 3 |     0 |     15 |  1, 2 |  0, 3 |
|  1,1  |  False |       1, 2 |     2 |    -35 |     1 |     2 |
|  1,2  |  False |       0, 3 |     2 |    -10 |     3 |     0 |
| 1,1,1 |  True  |          1 |     - |      - |     - |     - |
| 1,1,2 |  True  |          2 |     - |      - |     - |     - |
| 1,2,1 |  True  |          3 |     - |      - |     - |     - |
| 1,2,2 |  True  |          0 |     - |      - |     - |     - |

---

### **Tree 5**  
Training Dataset:  
[[-41   3  48]  
 [ 13 -34  26]  
 [ 29   1 -38]  
 [ 34  -5   0]]   

| Layer | ExNode |          X | SFeat | SValue | LNode |   RNode |
| :---: | :----: | ---------: | ----: | -----: | ----: | ------: |
|   0   |  False | 0, 1, 2, 3 |     0 |    -10 |     0 | 1, 2, 3 |
|  1,1  |  True  |          0 |     - |      - |     - |       - |
|  1,2  |  False |    1, 2, 3 |     0 |     30 |  1, 2 |       3 |
| 1,2,1 |  True  |       1, 2 |     - |      - |     - |       - |
| 1,2,2 |  True  |          3 |     - |      - |     - |       - |


Now all 5 trees have been trained, we move on to use it to find outliers from similar data.


## Testing the trees (Detecting outliers in a test set)
### Generating the test dataset
We use np.random.randint to generate the test dataset.

```python
import numpy as np
rng = np.random.default_rng(42)
test_dataset = rng.integers(-50, 51, size=(10, 3))
print(test_dataset)

# Output
[[-41  28  16]
 [ -6  -7  36]
 [-42  20 -30]
 [-41   3  48]
 [ 24  26  22]
 [ 29   1 -38]
 [ 34  -5   0]
 [-13 -32  43]
 [ 28  15 -10]
 [ 33   5  -6]]
```

Hence, this is our dataset for this exercise.

---

### Calculating the path length of each test datapoint for each tree
For this exercise, n = v = 4. Hence we get c(4) = 2*H(3) - 2(3)/4 = 2.1667

| datapoint      | T1 | T2 | T3 | T4 | T5 | E[h(x)] | s(x, n) |
| :------------: |:--:|:--:|:--:|:--:|:--:| :-----: | :-----: |
| [-41  28  16]  | 2  | 2  | 3  | 2  | 1  |   2.0   |  0.527  |
| [ -6  -7  36]  | 2  | 2  | 2  | 2  | 3  |   2.2   |  0.495  |
| [-42  20 -30]  | 2  | 2  | 3  | 2  | 1  |   2.0   |  0.527  |
| [-41   3  48]  | 2  | 2  | 2  | 2  | 1  |   1.8   |  0.562  |
| [ 24  26  22]  | 2  | 2  | 2  | 2  | 3  |   2.2   |  0.495  |
| [ 29   1 -38]  | 2  | 2  | 3  | 2  | 3  |   2.4   |  0.464  |
| [ 34  -5   0]  | 2  | 2  | 3  | 2  | 2  |   2.2   |  0.495  |
| [-13 -32  43]  | 2  | 2  | 1  | 2  | 1  |   1.6   |  0.599  |
| [ 28  15 -10]  | 2  | 2  | 3  | 2  | 3  |   2.4   |  0.464  |
| [ 33   5  -6]  | 2  | 2  | 3  | 2  | 2  |   2.2   |  0.495  |

---

### Identifying the anomalous point

From this conclusion, we notice that all the points have an anomalous score of only about 0.464 to 0.599, which does not have much variance meaning that most of them are normal instances. Since all the points return s ~ 0.5, the entire sample does not have any distince anomalies.   
However, for the purpose of this exercise, we take the datapoint with the largest anomalous score to be the anomaly, which in this case is the datapoint [-13 -32  43], which has the largest anomaly score of 0.599 ~ 0.6.

Hence, we can conclude that datapoint [-13 -32  43] is the anomaly in the test set.

---

### Plotting the datapoints to see the anomaly in 3d space
Here are the datapoints plotted out in 3d space.  
Green --> Normal point  
Red --> Anomalous point  

**Top view**
![Top View](images/top-view.png)

**Bottom view**
![Bottom View](images/bottom-view.png)

**Side view**
![Top View](images/side-view.png)
![Top View](images/side-view-2.png)

### Analysis of anomaly
From the images above, the point in red can even visually be classified as the most anomalous of the dataset, as it is slightly isolated from the rest of the dataset. This shows the power of the IForest algorithm, where even though the number of trees, max depth of trees and subsampling size is extremely small (For ease of computation), we still get an extremely easy to compute, AND accurate result of the anomaly.


## Conclusion
Though we were able to identify the anomaly from the dataset, there are some issues that arose from the way this exercise was conducted, in an attempt to reduce computational requirement as the exercise was done by hand. Here are 3 improvements that should be made when running the algorithm using a computer instead.


### 1. Generate Data from a Meaningful Distribution

One major limitation of the current implementation is that the training and testing datasets were generated using uniformly random integers between -50 and 50. This means that every datapoint is equally likely to occur and there is no underlying population mean, cluster structure, or notion of normal behaviour. Since Isolation Forest relies on anomalies existing in sparse regions relative to a dense population of normal instances, the model struggles to distinguish anomalies when the entire dataset is effectively random. As a result, the anomaly scores obtained were all relatively similar, indicating little separation between normal and anomalous observations. To address this issue, the dataset could instead be generated from a realistic distribution such as a Gaussian distribution centred around a population mean, with a small number of artificially injected outliers. This would create a meaningful distinction between common and rare observations, allowing the Isolation Forest to isolate true anomalies more effectively. This improves the quality of the underlying data rather than just improving the model's estimation process.

### 2. Increase the Subsampling Size

The current model uses a subsampling size of only four datapoints per tree, resulting in a maximum tree depth of two. This severely restricts the complexity of the isolation trees and limits the possible path lengths that can be assigned to observations. Consequently, many datapoints receive similar path lengths and anomaly scores, reducing the model's ability to differentiate between moderately unusual observations and genuine outliers. Increasing the subsampling size to values such as 32, 64, or 128 would allow each tree to grow deeper and capture more detailed structure within the dataset. This would produce a wider range of path lengths and anomaly scores, enabling finer distinctions between different levels of abnormality. As a result, the model would be expected to identify anomalies more accurately and provide greater separation between normal and anomalous instances.

### 3. Increase the Number of Trees

The current forest consists of only five isolation trees, making the anomaly scores highly dependent on a small number of random feature selections and split points. Since Isolation Forest is inherently random and stochastic, different random choices can significantly influence the path length of a datapoint when only a few trees are used. This introduces a high degree of variance into the final anomaly scores and may cause certain observations to appear anomalous purely due to chance. Increasing the number of trees to a more typical value such as 100 or more would reduce the influence of individual random splits and provide a more reliable estimate of the expected path length. This would lead to more stable anomaly scores, greater consistency across different runs of the algorithm, and improved confidence in the identified anomalies.

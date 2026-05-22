# Training Summary
---
## Topics covered
- Python Syntax
  - Basic syntax and internal workings of python
  - OOPs concepts
  - NumPy library
  - Pandas library
  - Matplotlib library
- Random Forest and Decision Trees
  - Prediction of the Price bracket to which given device belongs
- Anomaly detection using Deep Isolation Forest
- Optimised Deep Isolation Forest
  - Electricity Theft Detection
  - Anomaly detection on NPDCL electricity consumption data

## 1. Python — Core Language & OOP
### 1.1 Functions — Arguments, Recursion, Anonymous

#### Argument Types

```python
def func(parameters):
    pass
```

The regular python syntax for functions is to take in paramters, positional and named. However, python also allows to accept n number of parameters into a function

```python
# *args unpacks to tuple, **kwargs to dict
def fn(*args, **kwargs):
    print(args, kwargs)
```

#### Recursion

Recursion relies on the **call stack** — each call adds a frame to the stack and once the top frame returns a value, it passes the value to the frame below and returns, until all frames return a value and the function ends. Python's default recursion limit is 1000.

```python
# Example: Factorial
def factorial(n):
    return 1 if n <= 1 else n * factorial(n - 1)
```

#### Lambda (Anonymous Functions)

```python
# Single-expression functions, often used to be passed as arguments
sorted_data = sorted(records, key=lambda r: r["salary"], reverse=True)
```

---

### 1.2 File Handling — Text, Pickle, JSON

```python
with open("data.txt", "r") as f:
    lines = f.readlines()          # List of lines (keeps \n)
    # or: content = f.read()       # Whole file as string
    # or: for line in f:           # Memory-efficient iteration

with open("out.txt", "w") as f:
    f.writelines(lines)            # Note: No auto-newline
```

```python
# Pickle
import pickle

model = {"weights": [0.1, 0.3], "bias": 0.5}
with open("model.pkl", "wb") as f:   # binary write
    pickle.dump(model, f)

with open("model.pkl", "rb") as f:   # binary read
    loaded = pickle.load(f)
```


```python
# JSON
import json

data = {"name": "Alice", "scores": [95, 87, 91]}
json_str = json.dumps(data)       # dict → string
back = json.loads(json_str)                  # string → dict

with open("config.json", "w") as f:
    json.dump(data, f, indent=2)             # write to file
with open("config.json") as f:
    config = json.load(f)                    # read from file
```

---

### 1.3 Modules, Packages, and Libraries

```
module     → a single .py file
package    → a directory with __init__.py containing multiple modules
library    → a collection of packages/modules (e.g. NumPy, Pandas)
```

```python
# Module
import math
from math import sqrt, pi          # selective import
import numpy as np                 # alias
```

---

### 1.4 Exceptions

```python
# Catching specific exceptions — always prefer specific over bare `except:`
try:
    result = int(user_input)
except ValueError as e:
    print(f"Invalid input: {e}")
except (TypeError, KeyError):
    pass
else:
    print("Succeeded")        # runs only if no exception raised
finally:
    print("Always runs")      # cleanup — runs regardless
```
---

### 1.5 Shallow Copy vs Deep Copy

```python
import copy

original = [[1, 2], [3, 4]]

shallow = copy.copy(original)       # New outer list, SAME inner objects
deep    = copy.deepcopy(original)   # Fully independent
```

---

### 1.6 Object-Oriented Programming

#### Instance vs Class Variables & the `__dict__` Lookup Chain

```python
class Employee:
    raise_amount = 1.04          # Class variable — shared across all instances

    def __init__(self, first, last, pay):
        self.first = first       # Instance variable — unique per object
        self.pay = pay
```

When `emp_1.raise_amount` is accessed, Python first checks `emp_1.__dict__` (instance), then `Employee.__dict__` (class). This is the **attribute lookup chain**.

#### Regular, Class, and Static Methods

```python
class Employee:
    raise_amount = 1.04

    def apply_raise(self):              # Regular — receives instance as self
        self.pay = int(self.pay * self.raise_amount)

    @classmethod
    def set_raise_amt(cls, amount):     # Class method — receives class as cls
        cls.raise_amount = amount

    @staticmethod
    def is_workday(day):                # No implicit first arg — a utility function in class namespace
        return day.weekday() < 5
```

---

### 1.7 Iterators

An **iterator** is an object where:
- `__iter__()` — returns `self`
- `__next__()` — returns the next value or raises `StopIteration`

```python
# Under the hood of every for loop
iterable = [1, 2, 3]
it = iter(iterable)
print(next(it))           # 1
print(next(it))           # 2
```

---

### 1.8 Generators

Generators use `yield` to produce values lazily, stopping execution between calls.

**infinite sequence:**

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fibs = fibonacci()
[next(fibs) for _ in range(8)]   # [0, 1, 1, 2, 3, 5, 8, 13]
```
---

## 2. NumPy

**Why NumPy is faster than lists:**
- **Fixed type** — a Python list stores a pointer + type metadata per element; NumPy stores one dtype descriptor at the array level, then raw data. Fewer bytes to read, no per-element type checking.
- **Contiguous memory** — list elements are scattered in heap memory (accessed via pointers). NumPy stores elements side-by-side, enabling **SIMD** (Single Instruction Multiple Data) vectorised CPU operations.

---

### 2.1 Array Basics

```python
import numpy as np

a = np.array([1, 2, 3], dtype='int64')
b = np.array([[1., 2., 3.], [4., 5., 6.]])

b.ndim      # 2 — number of dimensions
b.shape     # (2, 3) — rows, columns
b.dtype     # float64
b.itemsize  # 8 — bytes per element
b.size      # 6 — total elements
b.nbytes    # 48 — total bytes (size * itemsize)
```

---

### 2.2 Indexing and Slicing

```python
a = np.array([[1,2,3,4,5,6,7],
              [8,9,10,11,12,13,14]])

a[1, 5]        # element at row 1, col 5 → 13
a[0, :]        # full row 0
a[:, 2]        # full column 2
a[0, 1:6:2]    # row 0, cols 1-5 step 2 → [2, 4]

# 3D: index works outside-in [depth, row, col]
b = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])
b[0, 1, 1]    # 4
```

---

### 2.3 Array Initialisation

```python
np.zeros((2, 3))             # all zeros
np.ones((2, 3), dtype='int32')
np.full((2, 2), 7.0)         # filled with a constant
np.identity(3)               # 3×3 identity matrix
```

---

### 2.4 Mathematics

```python
a = np.array([1, 2, 3, 4])

# Element-wise (broadcasting scalar over array)
a + 2;  a * 2;  a ** 2

# Element-wise between arrays (must be same shape, or broadcastable)
a + np.array([10, 20, 30, 40])

# Linear algebra
np.linalg.det(np.identity(3))   # determinant → 1.0
```

---


### 2.5 Reshaping and stacking

```python
a = np.array([[1,2,3,4],[5,6,7,8]])

a.reshape((4, 2))      # same elements, different shape (must be compatible)

# Stacking
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
np.vstack([v1, v2])
np.hstack([v1, v2])
```

---

### 2.6 Boolean Masking & Advanced Indexing

```python
a = np.array([[1, 54, 99], [12, 2, 84]])

a > 50                  # boolean array
a[a > 50]               # [54, 99, 84]

np.any(a > 50, axis=0)  # any match per column
np.all(a > 50, axis=1)  # all match per row

# Index with explicit list
b = np.arange(1, 10)
b[[0, 2, 7]]            # [1, 3, 8]
```

---

## 3. Pandas

### 3.1 Creating & Inspecting DataFrames

```python
import pandas as pd
import numpy as np

# From a dict — keys become column names
people = {"first": ["Corey", "Jane"], "last": ["Schafer", "Doe"],
          "email": ["corey@email.com", "jane@email.com"]}
df = pd.DataFrame(people)

# From CSV
df = pd.read_csv("survey.csv", index_col="Respondent")
```

---

### 3.2 Selecting Rows and Columns

```python
# Single column
df["email"]

# Multiple columns
df[["last", "email"]]

# .iloc — integer-based
df.iloc[0]           # row 0
df.iloc[0:2, 0:2]    # rows 0-1, cols 0-1 (end exclusive)

# .loc
df.loc[0, "email"]
df.loc[0:2, "Hobbyist":"Employment"]
```
---

### 3.3 Filtering Rows

```python
# Boolean filter mask
filt = (df["last"] == "Doe")
df.loc[filt]                      # preferred — allows column selection too
df.loc[filt, "email"]             # filtered rows + single column
```

---

### 3.4 Adding and Removing Rows/Columns

```python
# Add column
df["full_name"] = df["first"] + " " + df["last"]

# Split a column back out (expand=True → returns DataFrame, not list of strings)
df[["first", "last"]] = df["full_name"].str.split(" ", expand=True)

# Drop columns / rows
df.drop(columns=["first", "last"], inplace=True)
df.drop(index=4, inplace=True)

# Drop rows based on condition
df.drop(index=df[df["last"] == "Doe"].index, inplace=True)
```

---

### 3.5 Reading and Writing Data

```python
# CSV
df = pd.read_csv("data.csv", index_col="Respondent")
df.to_csv("output.csv")
df.to_csv("output.tsv", sep="\t")

# Excel
df.to_excel("output.xlsx")
df = pd.read_excel("output.xlsx", index_col="Respondent")
```

---

## 4. Matplotlib
### 4.1 Line Plots & Formatting

```python
plt.plot(x, y, color="#587b9a", linestyle="--", marker="o",
         linewidth=2, label="Python Devs")

plt.plot(x, y, "b:")   # blue dotted line

plt.xlabel("Age")
plt.ylabel("Salary (USD)")
plt.title("Median Salary by Age")
plt.show()
```
---

### 4.2 Bar Charts

```python
# Single category
plt.bar(x, y, color="#444444", label="All Devs")

# Horizontal bar — better when category labels are long
plt.barh(languages, popularity)
```
---

### 4.3 Histograms

```python
plt.hist(ages, bins=10, edgecolor="black")

# Custom bin edges
plt.hist(ages, bins=[10, 20, 30, 40, 50, 60], edgecolor="black")

# Log scale (useful when some bins have very low counts)
plt.hist(ages, bins=bins, log=True)
```
---

### 4.4 Scatter Plots

```python
plt.scatter(x, y, s=100, c=colors, edgecolor="black",
            linewidth=1, cmap="Greens", alpha=0.75)

cbar = plt.colorbar()
cbar.set_label("Satisfaction Level")

```
---

## 5. Random Forest & Decision Trees — Device Price Bracket Prediction

---

### 5.1 Theory — Decision Trees

A Decision Tree is a supervised learning model that recursively partitions the feature space into regions, assigning a class label (classification) or value (regression) to each region. Every node in the tree asks a binary question about one feature, **is feature <= n?**; and every leaf holds a prediction.

#### How Splits Are Determined - Training

During the training phase, each tree randomly selects $\sqrt(\# features)$ to sample from. At each node, the algorithm randomly selects a feature and based on the feature, it calculates the **gini impurity** at every possible split threshold to find the partition that maximises **information gain** — the reduction in impurity after the split.

**Gini Impurity** measures the probability that a randomly chosen element from a node would be incorrectly classified if labelled at random according to the class distribution in that node:

$$G(S) = 1 - \sum_{k=1}^{K} p_k^2$$

where $p_k$ is the proportion of class $k$ in sample set $S$. A perfectly pure node (all one class) gives $G = 0$; a maximally impure node gives $G = 0.5$ for binary problems.

**Weighted Gini** after splitting node $S$ into left ($S_L$) and right ($S_R$) subsets:

$$G_{weighted}(S_L, S_R) = \frac{|S_L|}{|S|} \cdot G(S_L) + \frac{|S_R|}{|S|} \cdot G(S_R)$$

**Information Gain** is the reduction in impurity achieved by the split:

$$IG = G(S) - G_{weighted}(S_L, S_R)$$

The split with the highest $IG$ is chosen. This search is run independently at every node.

> **Entropy** is an alternative impurity measure: $H(S) = -\sum_k p_k \log_2(p_k)$. Gini is generally preferred in practice — it is faster to compute (no logarithm) and tends to produce similar results to entropy-based splitting.

#### Stopping Conditions

Without constraints, a tree will grow until every leaf is pure — perfectly memorising the training data (overfitting). Hence, we implement stopping conditions, which are:

| Condition | Effect |
|-----------|--------|
| `max_depth` reached | Limits tree height |
| Node is pure ($G = 0$) | No split can improve things |
| Fewer than `min_samples` in node | Split becomes unreliable on small groups, leaning towards overfitting |
| $IG \leq$ `min_info_gain` threshold | Split is not meaningfully better than random |

> Note: `max_depth`, `min_samples`, `min_info_gain` are all hyperparameters in the model which are optmised based on hyperparameter tuning.
---

### 5.2 Theory — Random Forests

A random forest is a collection of N trained decision trees, `N_TREES`. A single decision tree has high variance — small changes in the training data can produce very different trees. A Random Forest addresses this in two ways: **bagging** and **feature subsampling**.

#### Bagging (Bootstrap Aggregating)

$N$ trees are trained on $N$ independently sampled bootstrap datasets — each created by sampling the training data **with replacement** to the same size as the original. Because sampling is with replacement, each bootstrap dataset contains roughly 63.2% unique rows (the rest are repeats), and each omits different rows, so each tree sees a slightly different view of the data.

Once all the trees are trained and unseen data is passed through the forest, the final prediction of the random forest is calculated by combining the predictions of the individual trees by **majority vote** (classification) or averaging (regression):

$$\hat{y} = \text{mode}\{f_1(x), f_2(x), \ldots, f_N(x)\}$$

#### Feature Subsampling (Random Subspace Method)

At each split within a tree, only a **random subset of $m$ features** is considered as candidates, rather than all features. This is the key step that *decorrelates* the trees — without it, all trees would repeatedly pick the same dominant features at the top and end up highly correlated, defeating the purpose of ensembling.

Standard choice: $m = \lfloor\sqrt{p}\rfloor$ for classification, where $p$ is the total number of features.

#### Why Ensembles of Decision Trees Reduce Variance

If individual trees each have variance $\sigma^2$ and are uncorrelated, averaging $N$ of them reduces variance to $\sigma^2/N$. In practice trees are correlated (they share the same training distribution), so the reduction is:

$$\text{Var(ensemble)} = \rho\sigma^2 + \frac{1-\rho}{N}\sigma^2$$

where $\rho$ is the pairwise correlation between trees. Feature subsampling directly reduces $\rho$, which is why it is the single most important hyperparameter to understand in a Random Forest.

#### Key Hyperparameters

| Parameter | Role |
|-----------|------|
| `n_trees` | More trees → lower variance, however, diminishing returns past ~100–200 |
| `max_depth` | Controls individual tree complexity |
| `min_samples_split` | Minimum samples required to attempt a split |
| `n_features` ($m$) | Subspace size per split — controls tree correlation |
| `min_info_gain` | Minimum gain threshold — prunes uninformative splits |

---

### 5.3 The Dataset

The dataset contains **694 printer/MFD (multi-function device) records** across 23 columns. The target variable is `listprice` (USD), which ranges from \$25 to \$200. The task is to replace this continuous price with a **percentile bin label** under three binning scenarios and predict the percentile bin that a new printer will land in based on its features.

**Feature overview:**

| Type | Columns |
|------|---------|
| Identifiers (dropped) | `guid`, `model` |
| Numeric | `width`, `depth`, `height`, `weight`, `speedMono`, `speedColor`, `printDPI`*, `standardinputcapacity`, `maximuminputcapacity`, `powerActive` |
| Boolean | `connNetwork`, `hasWifi`, `hasA3`, `copy`, `scan`, `fax` |
| Categorical | `make` (18 brands), `deviceType` (2), `printTechnology` (Inkjet/Laser), `hasDuplex` (True/False/Optional) |

*`printDPI` was stored as a string in `"W x H"` format and required parsing.

---

### 5.4 Exploratory Data Analysis

#### Step 1 — Setup & Data Integrity

```python
dev_prices = pd.read_csv("device_price_dataset.csv")
dev_prices = dev_prices.replace('\xa0', ' ', regex=True)
dev_prices_raw = dev_prices.copy()
```

The raw copy is preserved before any changes are made.

**`printDPI` parsing** — the column stored resolution as a string `"4800 x 1200"`, making it unusable as a numeric feature. It was converted to a single effective DPI by taking the geometric mean of the two components:

```python
dev_prices['printDPI'] = round((
    dev_prices['printDPI']
    .str.replace(' ', '', regex=False).str.lower().str.split('x')
    .apply(lambda x: int(x[0]) * int(x[1]))
) ** 0.5, 2)
```

#### Step 2 — Quality Checks

**No null values** were found across all 694 rows. No exact duplicate rows were found, and all `guid` values were unique.

**Impossible range checks** flagged several columns with zero values where zero is physically impossible (e.g., `width`, `speedColor`, `standardinputcapacity`, `maximuminputcapacity`). These rows were removed:

```python
dev_prices = dev_prices[dev_prices['width'] != 0]
dev_prices = dev_prices[dev_prices['speedColor'] != 0]
dev_prices = dev_prices[dev_prices['standardinputcapacity'] != 0]
dev_prices = dev_prices[dev_prices['maximuminputcapacity'] != 0]
```

**`weight` zeros** were handled differently — rather than removing rows, zeros were imputed with the column mean, as weight being missing (encoded as 0) is more likely than weight being invalid:

```python
dev_prices.loc[dev_prices['weight'] == 0, 'weight'] = dev_prices['weight'].mean()
```

**Outlier detection** was performed using both the IQR method (flag values beyond $Q_1 - 1.5 \cdot IQR$ or $Q_3 + 1.5 \cdot IQR$) and Z-scores (flag $|z| > 3$). Boxplots were generated for all numeric columns to visually confirm outlier distributions. The IQR function was generalised across all numeric columns at once:

```python
def find_outliers(df):
    numeric_cols = df.select_dtypes(include='number').columns
    summary = {}
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        count = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        summary[col] = {'outlier_count': count, 'lower': Q1-1.5*IQR, 'upper': Q3+1.5*IQR}
    return pd.DataFrame(summary).T.sort_values('outlier_count', ascending=False)
```

**Categorical consistency checks** revealed that `make` contained 18 brands with heavy imbalance — 5 brands (Hewlett-Packard, Canon, Epson, Brother, Lexmark) accounted for the vast majority of records. Minority brands were consolidated into an `'Others'` group to prevent rare categories from introducing noise:

```python
dev_prices.loc[~dev_prices['make'].isin(
    ['Hewlett-Packard', 'Canon', 'Epson', 'Brother', 'Lexmark']
), 'make'] = 'Others'
```

`hasDuplex` had three values: `True`, `False`, and `Optional`. This is intentionally kept as a 3-class categorical rather than forced to boolean.

#### Step 3 — Univariate Analysis

For every numeric column, the following statistics were computed and plotted:

- Histogram with mean and median lines overlaid
- Boxplot showing spread and outlier positions
- Skewness

Key findings:
- `listprice` is approximately uniform across its range (\$25–\$200), with mean ≈ median ≈ \$129 — indicating no strong price-point clustering.
- `speedMono` and `speedColor` are right-skewed — a small number of high-speed devices pull the distribution.
- `maximuminputcapacity` is heavily right-skewed with extreme outliers (enterprise-class paper trays).
- `powerActive` shows a bimodal pattern consistent with the Inkjet vs Laser split.

**Near-zero variance check** was run to flag columns with no discriminating power. All retained columns passed this check.

#### Step 4 — Bivariate Analysis

**Numeric vs Numeric — Correlation Matrix (Pearson & Spearman):**

High-correlation pairs were flagged at $|r| > 0.85$:

- `width` and `speedColor` were found to be strongly correlated with other features — dropped in modelling.
- `standardinputcapacity` and `maximuminputcapacity` showed strong correlation — `standardinputcapacity` was dropped as `maximuminputcapacity` had higher correlation with the target.
- `printDPI` had low correlation with `listprice`, contributing little predictive signal.

```python
# Identify which feature to keep from correlated pairs
for _, row in high_corr_df.iterrows():
    f1, f2 = row['feature_1'], row['feature_2']
    r1 = abs(dev_prices[f1].corr(dev_prices[target]))
    r2 = abs(dev_prices[f2].corr(dev_prices[target]))
    keep = f1 if r1 >= r2 else f2
    drop = f2 if r1 >= r2 else f1
    print(f'Keep: {keep}  |  Drop: {drop}')
```

**VIF (Variance Inflation Factor)** was computed to detect multicollinearity. Features with $VIF > 5$ were flagged as candidates for removal, providing a regression-based cross-check on the correlation-based approach.

**Categorical vs Numeric (ANOVA / t-test):**
- `printTechnology` (Inkjet vs Laser) showed a statistically significant difference in `listprice` (ANOVA $p < 0.05$) — Laser devices command higher prices.
- `hasA3` was significant — A3-capable devices are priced higher.
- `connNetwork` showed some association but weaker significance.

**Categorical vs Categorical (Cramér's V):**
- Cramér's V was computed for all categorical pairs. `copy`, `scan`, and `deviceType` showed strong mutual association (all MFDs copy and scan, making these near-constant or redundant).
- `hasWifi` and `connNetwork` were highly associated — `hasWifi` was dropped.
- `copy` and `scan` were dropped for the same reason.

---

### 5.5 Feature Selection & Final Feature Set

Following EDA, the following columns were dropped before modelling:

| Column | Reason |
|--------|--------|
| `guid`, `model` | Identifiers — no predictive value, near-100% cardinality |
| `deviceType` | Near-constant after quality filtering |
| `hasWifi` | Redundant with `connNetwork` (high Cramér's V) |
| `width` | High correlation with other physical dimensions |
| `printDPI` | Low correlation with target; noisy after parsing |
| `speedColor` | High correlation with `speedMono` |
| `standardinputcapacity` | Dominated by `maximuminputcapacity` |
| `copy`, `scan` | Near-constant (all MFDs) + high categorical association |
| `listprice` | Target variable — replaced by `percentile_bin` |

**Final features used for modelling (13 columns):**

```
Numeric (7): depth, height, weight, speedMono, maximuminputcapacity, powerActive
Categorical (6): printTechnology, hasDuplex, make, connNetwork, hasA3, fax
```

---

### 5.6 Target Engineering — Percentile Binning

The `listprice` column was converted into a discrete class label using `pd.qcut` — quantile-based binning:

```python
dev_prices["percentile_bin"] = pd.qcut(
    dev_prices["listprice"],
    q=int(100 / PERC_INTV),
    labels=False,
    duplicates='drop'
)
```

Three scenarios were evaluated:

| Scenario | Interval | No. of Bins | Difficulty |
|----------|----------|-------------|-----------|
| 1 | 5-percentile | 20 classes | High |
| 2 | 10-percentile | 10 classes | Medium |
| 3 | 20-percentile | 5 classes | Lower |

A higher percentile binning results in fewer, wider target classes — easier to classify but less informative, and may risk underfitting. Finer binning gives more information but makes the boundary between adjacent bins nearly indifferentiable to the model (a device at the 5th vs 6th percentile of price may have almost identical features) - risks overfitting.

---

### 5.7 Model — Random Forest (built from scratch)

The Random Forest was implemented from scratch in Python (no scikit-learn) using the Singleton, Node, and LeafNode class structure.

#### Preprocessing

Since the Decision Tree operates entirely on numeric comparisons, categorical columns were encoded using a dictionary built during `fit()` and applied during `transform()`:

```python
def fit_preprocess(self, dataset):
    non_numeric_cols = dataset.select_dtypes(exclude='number').columns.tolist()
    for column in non_numeric_cols:
        self.preprocessing[column] = {}
        for index, value in enumerate(dataset[column].unique()):
            self.preprocessing[column][value] = index + 1

def transform(self, dataset):
    for column in dataset.select_dtypes(exclude='number').columns:
        if column in self.preprocessing:
            dataset[column] = dataset[column].map(self.preprocessing[column])
            dataset[column] = dataset[column].fillna(dataset[column].mean())
    return dataset
```

The purpose of using this structure was to ensure similar encoding could be done to unseen categorical dataset when the model is tested. Unseen categories at prediction time are filled with the column mean — a safe fallback for ordinal-encoded features.

#### Training Loop

```python
N_FEATURES = round((n_features) ** 0.5)   # √p subspace sampling
N_TREES = 100
```

For each of the 100 trees:
1. A **bootstrap sample** is drawn (same size as training set, with replacement)
2. A `DecisionTree` is trained on the bootstrap sample
3. The tree is added to the forest

```python
def bootstrap(dataset):
    sample = dataset.sample(n=dataset.shape[0], replace=True)
    return sample.drop(columns="percentile_bin"), sample["percentile_bin"]
```

#### Tree Building — `_build_tree`

Stopping conditions checked at each node:

```python
if depth > MAX_DEPTH:             → LeafNode(majority class)
elif sample_y.nunique() == 1:     → LeafNode(majority class)   # pure node
elif sample_y.count() < MIN_SAMPLES: → LeafNode(majority class)
```

Split search — for each randomly selected feature subset, and every unique value as a candidate split threshold:

```python
initial_gini = gini_impurity(sample_y)
for column in sample_columns:           # N_FEATURES columns, randomly selected
    for value in sample_X[column].unique():
        mask = sample_X[column] <= value
        ig = information_gain(sample_y[mask], sample_y[~mask], initial_gini)
        if ig > max_info_gain:
            max_info_gain = ig
            node.feature = column
            node.split_point = value

if max_info_gain <= MIN_INFO_GAIN_REQ:
    return LeafNode(majority class)     # no useful split found
```

The Gini and information gain functions implement the formulae from Section 4.1 exactly:

```python
def gini_impurity(data_subset):
    total = data_subset.count()
    return 1 - sum((data_subset.value_counts().get(v, 0) / total) ** 2 for v in data_subset.unique())

def information_gain(left, right, initial_gini):
    n = left.count() + right.count()
    weighted = (left.count()/n) * gini_impurity(left) + \
               (right.count()/n) * gini_impurity(right)
    return initial_gini - weighted
```

#### Prediction

Each tree votes on the class for a given input row via recursive traversal:

```python
def rec_pred(node, test_X):
    if isinstance(node, LeafNode):
        return node.predicted_class
    if test_X[node.feature].iloc[0] <= node.split_point:
        return rec_pred(node.left, test_X)
    else:
        return rec_pred(node.right, test_X)
```

The forest aggregates by **majority vote** across all 100 trees:

```python
prediction = pd.Series([tree.predict(row) for tree in self.trees]).mode()[0]
```

#### Train/Test Split

An 80/20 split was used:

```python
test_data  = dev_prices.sample(frac=0.2)
train_data = dev_prices.drop(test_data.index)
```

#### Hyperparameters Used

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `N_TREES` | 100 | Sufficient ensemble size with acceptable runtime |
| `MAX_DEPTH` | 10 | Limits overfitting while allowing deep enough splits |
| `MIN_SAMPLES` | 5 | Prevents splits on tiny, noisy leaf populations |
| `N_FEATURES` | $\lfloor\sqrt{p}\rfloor$ | Standard choice for classification |
| `MIN_INFO_GAIN_REQ` | 0.01 | Prunes splits that add negligible predictive value |

---

### 5.8 Results — Three Scenarios

| Scenario | Bins | Expected Difficulty | Notes |
|----------|------|-------------------|-------|
| 1 (5-percentile) | 20 | Hardest | Adjacent bins nearly indistinguishable in feature space |
| 2 (10-percentile) | 10 | Medium | Reasonable class separation |
| 3 (20-percentile) | 5 | Easiest | Wide bins; model has clear signal per class |

The accuracy metric used was simple classification accuracy — proportion of test rows where `predicted bin == actual bin`. As expected, Scenario 3 (5 classes) produced the highest accuracy, and Scenario 1 (20 classes) the lowest, due to the difficulty of distinguishing fine-grained price boundaries from physical/spec features alone.

---

### 5.9 Design Observations

**Singleton pattern on the `RandomForest` class** — the `_instance` check in `__new__` ensures only one forest object exists.

**Ordinal encoding vs one-hot** — the custom encoder assigns integers to categories, implying an ordering that may not exist (e.g., `make: Epson=1, Canon=2` implies Epson < Canon, which is meaningless). This can mislead tree splits that treat the encoded value as a continuous threshold. A one-hot encoding would be strictly more correct for nominal categories, at the cost of dimensionality.

**Unique-value threshold splitting** — the split search tries every unique value as a threshold, which is correct but $O(n \cdot p)$ per node. Production implementations (e.g. scikit-learn's ExtraTreesClassifier) use random thresholds for speed.

---

## 6. Anomaly Detection — Deep Isolation Forest (DIF)
---

### 6.1 What is Anomaly Detection?

Anomaly detection is the task of assigning an **abnormality score** to each data object in a dataset — no labels required. The goal is to identify data points that are "different" from the majority. Applications of anomaly detection includes fraud detection, fault diagnosis, and electricity theft detection.

The abnormality scoring function is defined as:

$$f: \mathcal{D} \mapsto \mathbb{R}^N$$

where $\mathcal{D} = \{o_1, \ldots, o_N\}$ is the dataset and each output is a real-valued anomaly score. Higher score = more anomalous.

---

### 6.2 The Foundation — Standard Isolation Forest (iForest)

Before understanding DIF, we try to understand what it improves upon.

iForest builds an ensemble of **Isolation Trees (iTrees)**. The intuition is that anomalies are *rare* and *different* and hence require fewer random cuts to isolate than normal points, which are surrounded by similar neighbours. Therefore, they are quickly isolated when passed through an iTree.

**iTree construction:**
- A random subsample of $n$ data objects initialises the root node
- At each node, a random feature $j$ is selected, and a random split threshold $\eta$ is selected from the range of values the feature takes on.
- The node splits into two children: $\{o \mid o^{(j)} \leq \eta\}$ and $\{o \mid o^{(j)} > \eta\}$
- Recursion continues until each leaf contains one point or maximum depth is reached

**Anomaly scoring:**

The path length from root to leaf node is the isolation difficulty. Shorter path = easier to isolate = more anomalous.

$$\mathcal{F}_{iForest}(o \mid \mathcal{T}) = 2^{-\frac{\mathbb{E}_{\tau \in \mathcal{T}}|p(o|\tau)|}{C(n)}}$$

where $|p(o|\tau)|$ is the path length in tree $\tau$, and $C(n)$ is a normalising factor.

**Strengths of iForest:**
- Linear time complexity $O(N)$
- No assumptions about data distribution
- Scales to large datasets

---

### 6.3 The Two Core Problems with iForest

#### Problem 1 — Hard Anomalies (False Negatives)

iForest only considers **one feature per split** — it is axis-parallel. This means it can only draw horizontal or vertical cuts in the feature space. When anomalies can only be identified by looking at the **combination of multiple features** (e.g., anomalies surrounded by a ring of normal points), iForest cannot isolate them effectively — **false negatives**.

#### Problem 2 — Ghost Regions / Algorithmic Bias

iForest's axis-parallel partitions introduce artefact regions in empty parts of the data space that receive unexpectedly **low anomaly scores**. These are rectangular areas centred around clusters of normal data — regions that contain no samples but are incorrectly treated as "normal" because the axis-parallel cuts happen to produce shallow trees there.

**Extensions like EIF (Extended Isolation Forest)** use random hyperplanes (grandiented cuts with random slopes and intercepts) to partially address the ghost region problem, but they are still *linear partitions* — they cannot handle truly non-linear anomaly boundaries.

---

### 6.4 DIF — The Key Idea

DIF's central insight: **if you cannot separate anomalies with linear cuts in the original space, project the data into new spaces where you can.**

Instead of devising more complex splitting criteria, DIF uses **randomly initialised neural networks** to map data into new representation spaces, then applies the same simple axis-parallel isolation as iForest in those projected spaces. A linear cut in the projected (non-linear) space is equivalent to a **non-linear partition** back in the original data space.

$$\text{Linear cut in projected space} \equiv \text{Non-linear cut in original space}$$

Crucially, the networks are **not trained** — they are only randomly initialised. This is a deliberate design choice (discussed in Section 6.6).

---

### 6.5 The DIF Algorithm

#### Step 1 — Random Representation Ensemble

DIF constructs $r$ representation spaces using $r$ randomly initialised neural networks:

$$\mathcal{G}(\mathcal{D}) = \left\{ \mathcal{X}_u \subset \mathbb{R}^d \mid \mathcal{X}_u = \phi_u(\mathcal{D}; \theta_u) \right\}_{u=1}^{r}$$

where $\phi_u$ is a neural network with randomly initialised weights $\theta_u$ that maps each data object to a $d$-dimensional vector. Each of the $r$ representations is a completely different transformation of the same data because each neural network is randomly initialized.

#### Step 2 — Build iTrees in Projected Spaces

For each of the $r$ representations, $t$ isolation trees are built using standard method — but now operating on the transformed dataset $\mathcal{X}_u$. Total trees: $T = r \times t$.

#### Step 3 — Anomaly Scoring (DEAS)

Standard iForest only uses path length. DIF introduces the **Deviation-Enhanced Anomaly Scoring (DEAS)** function, which incorporates an additional measure of *how far* each data point is from each split threshold it encounters during traversal.

**Averaged deviation degree** for data object $o$ in tree $\tau_i$:

$$g(x_u | \tau_i) = \frac{1}{|p(x_u|\tau_i)|} \sum_{k \in p(x_u|\tau_i)} |x_u^{(j_k)} - \eta_k|$$

A small deviation means the data point was very close to the split threshold — indicating it was in a **dense region** that was difficult to partition cleanly. Large deviations indicate sparse, easily-isolated regions.

**Combined DEAS scoring function:**

$$\mathcal{F}_{DEAS}(o \mid \mathcal{T}) = 2^{-\frac{\mathbb{E}_{\tau_i \in \mathcal{T}}|p(x_u|\tau_i)|}{C(T)}} \times \mathbb{E}_{\tau_i \in \mathcal{T}}\left[g(x_u|\tau_i)\right]$$

The first term is the normal iForest depth-based score from above. The second term multiplies it with the average deviation — a point that is consistently close to split boundaries (small $g$) gets its anomaly score suppressed; a point consistently far from split boundaries (large $g$) gets it amplified.

---

### 6.6 CERE — Computation-Efficient Representation Ensemble

Producing $r$ separate neural networks sequentially would be expensive. DIF solves this with **CERE**, which computes all $r$ representations simultaneously in one forward pass.

The weight matrix $W_i$ for the $i$-th ensemble member is derived from a single shared base matrix $W_0$ and two small random vectors $p_i \in \mathbb{R}^m$ and $q_i \in \mathbb{R}^n$:

$$W_i = W_0 \circ (p_i q_i^\top)$$

where $\circ$ is the Hadamard (element-wise) product. The entire ensemble forward pass then becomes a single batched matrix operation:

$$\begin{bmatrix} XW_1 \\ XW_2 \\ \vdots \\ XW_r \end{bmatrix} = \left( \begin{bmatrix} X \\ X \\ \vdots \\ X \end{bmatrix} \circ \begin{bmatrix} P_1 \\ P_2 \\ \vdots \\ P_r \end{bmatrix} \right) W_0 \circ \begin{bmatrix} Q_1 \\ Q_2 \\ \vdots \\ Q_r \end{bmatrix}$$

This reduces the cost of generating $r$ representations to approximately the cost of a single network's forward pass, since all ensemble members are computed in parallel.

**Time complexity:** $O(ND \cdot r \times t)$ — linear in data size $N$, dimensionality $D$, and ensemble size. This inherits iForest's scalability.

---

### 6.7 Why Random (Not Trained) Representations?

This is the most counterintuitive design choice in DIF, yet is key to the models success.

Trained representations have two critical weaknesses in this context:

1. **Loss functions are not universal.** A loss that works well for one dataset's anomaly structure may be a poor fit for another. There is no single supervised representation learning objective that generalises across diverse anomaly types.

2. **Optimisation kills diversity.** Isolation-based scoring works because the *ensemble* of diverse trees covers many different views of the data. If all representations are optimised toward the same objective, they converge and the benefit of ensembling is lost.

Randomly initialised networks, by contrast, produce **diverse** projections. Non-linear activation functions (e.g., ReLU, Tanh) in these randomly initialised networks fold and bend the data space in different directions — even without training, they create genuinely diverse views. Hard anomalies that are inseparable in the original space will be exposed as easy-to-isolate in at least some projected spaces. Because anomaly scoring is an average over all trees, the anomaly will stand out wherever it is successfully isolated.

---

### 6.8 Summary — What Makes DIF Better

| Limitation | iForest | EIF | DIF |
|-----------|---------|-----|-----|
| Linear cuts only | axis-parallel | oblique linear | non-linear via NN projection |
| Ghost region bias | ✗ severe | ✗ partial | ✓ eliminated |
| Hard anomaly detection | ✗ | ✗ partial | ✓ |
| High-dimensional data | ✓ | ✗ OOM on extreme dims | ✓ (projects to small $d$) |
| Scalability | $O(N)$ | $O(N)$ | $O(N)$ — via CERE |

The fundamental advance DIF makes is separating *representation* from *isolation* — using neural networks not as learned detectors, but as **random space transformers** that give the simple axis-parallel isolation mechanism the freedom to operate effectively in any direction in the original data space.

---

## 7. Optimised Deep Isolation Forest (ODIF)

ODIF is a direct optimisation of DIF. The anomaly detection logic, scoring function (DEAS), and representation scheme (CERE) are **entirely unchanged** — only the training phase is restructured to require less computational power and less memory.

---

### 7.1 The Problem with DIF's Training Phase

In standard DIF, the full training dataset of $N$ samples is transformed into $r$ representations via CERE before any isolation trees are built. The isolation forests then subsample from those already-computed representations. This means:

- All $N$ samples are transformed — even those never used in any tree
- Memory scales with the full dataset: $O(Nrb)$ for representations, where $b$ is the representation dimensionality

This is wasteful. Each isolation tree only uses $n$ samples (default: 256), and there are $r \times t$ trees total. The rest of the transformed data is computed and stored for nothing.

---

### 7.2 The ODIF Optimisation — Sample First, Transform Second

ODIF flips the order: **select samples before transforming them**.

```
Step 1 — Pre-sample:  draw all needed indices upfront (t × n total samples)
Step 2 — Deduplicate: extract unique indices — each sample transformed only once
Step 3 — Transform:   apply CERE only to this reduced subset
Step 4 — Build trees: assign pre-sampled (already-transformed) samples to each tree
(no re-sampling at tree construction time)
```

**Complexity comparison:**

| | Time complexity | Memory (representations) |
|--|----------------|--------------------------|
| DIF | $O(Nar + rt\log n)$ | $O(Nrb)$ |
| ODIF | $O(tnar + rt\log n)$ | $O(tnrb)$ |

Where $N$ is total training samples and $tn$ is the much smaller number of samples actually needed. For large datasets where $N \gg tn$, this reduction is substantial.

---

### 7.3 Results Summary

Tested across 18 real-world datasets:

**Detection accuracy (PR AUC):** ODIF and DIF produce statistically indistinguishable results ($p = 0.52$) — the optimisation preserves detection quality exactly. Both significantly outperform iForest, ECOD, and SGAE ($p < 0.001$).

**Training speed:**

| Stage | ODIF vs DIF |
|-------|------------|
| CPU | ~1.5× faster on average; several-fold on large datasets |
| GPU | ~150× faster on average (~20 ms vs ~2,600 ms) |


**Memory:**

| Resource | Reduction |
|----------|-----------|
| RAM | ~18% lower on average |
| VRAM | ~55% lower on average |

---


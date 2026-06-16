# Creating representations in Deep Isolated Forest manually by hand
Purpose: To deeply understand the underlying mechanism and algorithm used in the CERE step.  

## Step 1: Generate a random dataset
Because this exercise is being done by hand and aims to just create some representations and to visualise the output, we limit the dataset to have 3 features each and have 9 datapoints to remain computationally simple. Hence our dataset can be represented by a 9 x 3 matrix. Each element in the dataset is limited to integers between -5 and 5 inclusive, once again with the purpose to remain computationally simple.

I have carefully chosen this dataset such that the point to be isolated (anomaly) is in the middle of the normal points (non-anomalies), however requires a circular split to be able to isolate correctly. This allows us to see if the representations really do transform the dataset, allowing me to isolate the anomlalous point using stright lines. 

```python
[[ 0  -5   5]
 [-2  -2   3]
 [-5   0   4]
 [-3   3  -2]
 [ 0   5  -3]
 [ 2   3  -1]
 [ 5   0   0]
 [ 3  -2  -5]
 [ 0   0   0]]  # Anomalous point to be isolated
```
Hence, this is our dataset for this exercise.

## Step 2: Create Representations
I have decided to have 3 representations to see the effects of transformations using randomly generated neural network.

All randomly generated values for w, p, q and b are taken to be in the range [-2, 2] to keep the numbers somewhat small and easy to compute and visualise.

Representation 1:  
* 1 hidden layer
* h = ReLU(W1 * X + b1)  --> Hidden layer 1
* o1 = W2 * h + b2        --> output layer

---
Representation 2:  
* 2 hidden layer
* h1 = ReLU(W1 * X  + b1)  --> Hidden layer 1
* h2 = ReLU(W1 * h1 + b2)  --> Hidden layer 1
* o2 = W3 * h2 + b3 --> output layer

---
Representation 3:  
* 2 hidden layer
* h1 = ReLU(W1 * X  + b1)  --> Hidden layer 1
* h2 = ReLU(W1 * h1 + b2)  --> Hidden layer 1
* o3 = W3 * h2 + b3 --> output layer




## Step 3: Create the Ws and bs for each representation
### **Representation 1**
* 1 hidden layer
* h = ReLU(W1 * X + b1)  --> Hidden layer 1
* o1 = W2 * h + b2        --> output layer

NOTE: Refer to handwritten notes for exact mathematical calculations

$
W1 =
\begin{bmatrix}
-1 &  2 &  0 \\
-2 &  2 &  4 \\
-4 &  0 & -4
\end{bmatrix}
$

$
b1 = \begin{bmatrix}
1 \\
-2 \\
0
\end{bmatrix}
$

$
W2 =
\begin{bmatrix}
 0 &  1 & -1 \\
 8 &  0 &  2 \\
 2 & -1 &  2
\end{bmatrix}
$

$
b2 = \begin{bmatrix}
2 \\
1 \\
-1
\end{bmatrix}
$

Passing the initial dataset through representation 1, we get:  
$
O1 =
\begin{bmatrix}
 10 &  12 &  22 & -16 & -10 &  2 &  2 & -6 & 2 \\
 1  &   1 &  57 & 121 & 113 & 41 &  1 & 17 & 9 \\
-9  &  11 & -5  &  57 &  45 &  9 & -1 & 15 & 1
\end{bmatrix}
$

---
### **Representation 2**
* 2 hidden layer
* h1 = ReLU(W1 * X  + b1)  --> Hidden layer 1
* h2 = ReLU(W1 * h1 + b2)  --> Hidden layer 1
* o2 = W3 * h2 + b3 --> output layer

NOTE: Refer to handwritten notes for exact mathematical calculations

$
W1 =
\begin{bmatrix}
 2 &  2 &  0 \\
 2 &  1 &  4 \\
 2 &  0 & -2
\end{bmatrix}
$

$
b1 = \begin{bmatrix}
 0 \\
 1 \\
-1
\end{bmatrix}
$

$
W2 =
\begin{bmatrix}
 0 &  2 & -1 \\
-2 &  0 & -1 \\
 0 &  0 &  0
\end{bmatrix}
$

$
b2 = \begin{bmatrix}
 1 \\
-1 \\
 2
\end{bmatrix}
$

$
W3 =
\begin{bmatrix}
 1 &  2 &  0 \\
 0 &  0 &  0 \\
-1 &  0 &  0
\end{bmatrix}
$

$
b3 = \begin{bmatrix}
 2 \\
 1 \\
-1
\end{bmatrix}
$

Passing the initial dataset through representation 1, we get:  
$
O2 =
\begin{bmatrix}
 35 &  17 &  17 &  3 &  2 &  6 &  16 & 22 &  6 \\
 1  &   1 &   1 &  1 &  1 &  1 &   1 &  1 &  1 \\
-34 & -16 & -16 & -2 & -1 & -5 & -15 & -1 & -5
\end{bmatrix}
$

---
### **Representation 3**
* 2 hidden layer
* h1 = ReLU(W1 * X  + b1)  --> Hidden layer 1
* h2 = ReLU(W1 * h1 + b2)  --> Hidden layer 1
* o3 = W3 * h2 + b3 --> output layer

NOTE: Refer to handwritten notes for exact mathematical calculations

$
W1 =
\begin{bmatrix}
-2 &  0 &  0 \\
 4 &  0 & -4 \\
 4 &  0 &  2
\end{bmatrix}
$

$
b1 = \begin{bmatrix}
 1 \\
-1 \\
-1
\end{bmatrix}
$

$
W2 =
\begin{bmatrix}
 0 &  2 &  2 \\
 2 &  0 &  1 \\
-2 & -2 & -4
\end{bmatrix}
$

$
b2 = \begin{bmatrix}
 2 \\
 2 \\
 1
\end{bmatrix}
$

$
W3 =
\begin{bmatrix}
 1 & -2 &  0 \\
-2 &  2 & -2 \\
 1 &  0 &  0
\end{bmatrix}
$

$
b3 = \begin{bmatrix}
 2 \\
-1 \\
-1
\end{bmatrix}
$

Passing the initial dataset through representation 1, we get:  
$
O3 =
\begin{bmatrix}
 -4 & -20 & -44 & -28 &  18 &  22 &   38 &   62 & -4 \\
-15 &  19 &  43 &  27 & -41 & -55 & -115 & -127 &  3 \\
 19 &   1 &   1 &   1 &  23 &  33 &   77 &   65 &  1
\end{bmatrix}
$


## Step 4: Visualising the original dataset against the transformed datasets
### **Representation 1**  
Initial Dataset:  
$
Oi =
\begin{bmatrix}
 0 & -2 & -5 & -3 &  0 &  2 &  5 &  3 &  0 \\
-5 & -2 &  0 &  3 &  5 &  3 &  0 & -2 &  0 \\
 5 &  3 &  4 & -2 & -3 & -1 &  0 & -5 &  0
\end{bmatrix}
$

After passing through representation 1:  
$
O1 =
\begin{bmatrix}
 10 &  12 &  22 & -16 & -10 &  2 &  2 & -6 & 2 \\
 1  &   1 &  57 & 121 & 113 & 41 &  1 & 17 & 9 \\
-9  &  11 & -5  &  57 &  45 &  9 & -1 & 15 & 1
\end{bmatrix}
$


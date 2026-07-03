#creating tensors

import torch

x = torch.tensor(7) #0D tensor, just an int
y = torch.tensor([1,2,3]) #1D tensor, a list
z = torch.tensor([[1,2,3], [4,5,6], [7,8,9]]) #2D tensor, a matrix
o = torch.tensor([
    [ #page 0
        [1,2,3],
        [4,5,6]
    ],
    [ #page 1
        [4,5,6],
        [7,8,9]
    ],
    [ #page 2
        [7,8,9],
        [1,2,3]
    ]
]) #3D tensor, like a book with matrices in each page

#print(z.shape)

'''
Structural rules:

# 1. One data type

Tensors are designed for GPU calculations, so the data type should be the same and it should be numerical. Almost everytime it will be float32.

# 2. Same shapes

Every rows or matrices should share the same exact shapes (lenghts). If Row 0 is 3 units long, then the row 1 should be also 3 units long. To check sizes, use tensor.shape.

# Math is Element-Wise

To change something in python lists (like adding +10 to each element), you need to use for loops. But tensors are optimized for it, so you could just do this:
    x = torch.tensor([1,2,3])
    print (x+10)
the output will be tensor([11,12,13]) because PyTorch did the calculations in memory
'''


# Shape: [2, 3, 2] -> [Page, Row, Column]
# Alternatively:      [Batch, Sequence, Features]
ai_data = torch.tensor([
  [ # Page 0 (Sentence 1)
    [1.1, 1.2], # Word 0
    [2.1, 2.2], # Word 1
    [3.1, 3.2]  # Word 2
  ],
  [ # Page 1 (Sentence 2)
    [4.1, 4.2], # Word 0
    [5.1, 5.2], # Word 1
    [6.1, 6.2]  # Word 2
  ]
])

#print(ai_data.shape)
#print(ai_data[1, 2, :])
#print(ai_data[:, 0, -1])

'''
This is called Slicing and Indexing. Use : for every and -1 for the last.
'''

# Your raw audio recording data (12 elements total)
raw_audio = torch.tensor([0.1, -0.2, 0.5, 0.9, -0.1, 0.0, 0.4, 0.3, -0.6, 0.8, 0.2, -0.1])

D2_matrix = raw_audio.view(4, -1)
#print(D2_matrix.shape)

D3_matrix = raw_audio.view(2, 3, 2)
#print(D3_matrix.shape)

'''
This is called reshaping. If something gives an output of a 6-item list and my nn accepts only 2 by 3 matrices, Iwill just use .view() to reshape the data into the desired shape. 

Rule: total number of elements stays exactly the same:
    you can reshape 6 into 2*3
    you can reshape 6 into 2*3*1
    but you cant do 6 into 4*2 because its 8. -> error

Use -1 for a automatic available length detection. Lets say, if I have to have 3 rows, but I dont know the amount of coloumns I can have, I use data.view(3, -1). 
'''


# Additional functions

'''
1. Element-Wise Math (The Simple Stuff)

if divide, multiply, add, or substract a single number (scalar) from a tensor, PyTorch does it for every value inside of the tensor.
'''

r = torch.tensor([1,2,3])
#print(r*3)

'''
2. Matrix multiplication

This is the core function and purpose of PyTorch and tensors. This is what LLMs do over 90% of the times.

It is not multiplying cell-by-cell. Instead, it takes the rows of one matrix and dots them with the coloumns of the other matrix. In PyTorch, we use @ for it.
'''

o = torch.tensor([
    [1,2],
    [3,4]
])

t = torch.tensor([
    [5,6],
    [7,8]
])

#print(t @ o)

'''
3. Aggregations

Used for getting an output after the claculations.

.sum() for summing all items in a tensor
.mean() for getting a mean of all items in a tensor
.min() / .max() for getting a min/max value

Use dim=0 / dim=1 for aggregation of only rows or coloumns
'''

m = torch.tensor([
    [1,2,3],
    [4,5,6]
])

#print(m.sum(dim=0)) #sum of coloumns
#print(m.sum(dim=1)) #sum of rows

'''
Important rules for multiplication:

1. The number of columns of the first matrix should equal to the number or rows in the second matrix:
    2x3 and 3x4

2. Order Matters Interactively

In basic math: A*B = B*A
But in matrix multiplications: A@B!=B@A

If A is 4x3 and B is 3x2:
    A@B is 3x3 and the output is 2x2 -> legal
    B@A is 2x4 -> illegal

3. The Identity Matrix is the Number "1"

Identity Matrix "I" is a squared matrix filled with 0s and 1s with in a diagonal.

Multiplying I to any matrix gives the same matrix. Its like 6*1=6

4. Associative & Distributive Laws Apply

Grouping - if you need to multiply 3 matrices you can first multiply 1st with second or 2nd with 3rd, the output wont change:
    (A@B)@C = A@(B@C)

Distribution - if you multiply a matrix to a sum of two other matrices, you could distribute them, so that the 1st matrix will be multiplyed to each matrix in the sum:
    A@(B+C) = A@B + A@C
'''


# Matrix A: Input Data (2 words, 3 features each)
# Shape: [2, 3]
X = torch.tensor([
    [1, 0, 2],  # Word 0
    [0, 2, 1]   # Word 1
], dtype=torch.float32)

# Matrix B: Model Weights (3 features in, 2 features out)
# Shape: [3, 2]
W = torch.tensor([
    [2, 1],
    [0, 3],
    [1, 2]
], dtype=torch.float32)

# matrices can be multiplied because 2x3 and 3x2

output = X @ W #X is input and W is weights
#print(output.shape)
#print(output[0,1])

# gradient is a vector that points in the direction of increasing error, telling us exactly how to adjust weights to minimize mistakes.


#I = torch.eye(3)
#print(I)

hello = torch.eye(4)
#print(hello)

# eye is used for creation of Identity Matrices

# Context window is how many info can a model hold in its brain. Weights are how smartly can it use that info.

'''
To make an LLM smarter, we often need to adjust weights.

1. The Loss function

Find out how wrong the model is. If model predicts cars price to be 20K, but the actual price is 21K, the error is small. But if 30K, and the actual is 40K, the error i big.

2. Finding the direction of adjustment

PyTorch using calculus (Backpropagation) tracks the error back backward through made matrix multiplications. It looks at every single weight parameter and asks a calculus question: "If I increase this specific weight by a tiny fraction, will the Loss number go up or down?"

Gradient helps us to choose a direction for needed adjustments:
    Gradient +5: we need to decrease a particular weight knob
    Gradient -2: we need to increase it.

The gradient value itself tells the model which direction to turn the knob.

3. Controlling the Step Size (The Learning Rate)

Now we know the direction(sign of the gradient) and the urgency to change(the absolute value of the gradient), so we need to adjust.

To contral the weight parameters, engineers use a multiplier called Learning Rate (a tiny fraction like 0.01 or 0.001)

    New Weight = Current Weight - (Learning rate * Gradient)

An Example in Plain English
Imagine a single weight knob is currently set to 3.0.  
The model makes a guess and gets it wrong.
The calculus engine calculates the gradient for this weight and gets +10.0 (meaning it's way too high).
We multiply that gradient by our small learning rate (0.01), which gives us 0.1.
The model updates the weight: 3.0 - 0.1 = 2.9.
'''


# Easy task

# Input representing 3 data samples, each with 4 features
X = torch.rand(3, 3)

# Weight matrix
W = torch.rand(3, 2)

# This line crashes!
#output = X @ W
#print(output)



#Hard task

raw_audio = torch.tensor([
    0.2, -0.4, 0.6, 0.1, -0.1, 0.0,  # Minute 0 data
    0.5, 0.9, -0.3, 0.2, 0.1, -0.2,  # Minute 1 data
    0.8, -0.7, 0.4, 0.3, 0.0, 0.1,   # Minute 2 data
    -0.5, 0.6, 0.2, -0.1, 0.9, 0.4   # Minute 3 data
])

cleaned = raw_audio.view(-1,6)
#print(cleaned)
#print(cleaned.mean(dim=1))

'''
Calculating gradient using backwards calculus. Autograd is the tracking engine that stores calculations in memory and uses calculus to undo some of them.

1. Flag a tensor

use requires_grad=True

eg:
    x = torch.tensor([1,2,3], requires_grad=True)

2. Computation Graph

When you do something with the tensor you flagged, the autograd holds the calculations in memory, so it will be easier to trace back. It creates a hidden dynamic map called a Computational Graph.
'''
x = torch.tensor(3.0, requires_grad=True)
m = torch.tensor(4.0)
y = x * m
loss = y**2

#x ---(*m)---> y ---(**2)----> loss

print(loss) #will show tensor(144., grad_fn=<PowBackward0>), which serves as a note that says the last action on this tensor was taking it into power.

loss.backward()
#print(x.grad)

'''
Backwards calculations:

    It looks at the output as (x * m)^2, and it takes the derivative of the loss with respect to the initial value (x) using the chain rule.

    derivative = 2 * x * m * m = 2 * y * m

    so the gradient = 2 * 3.0 * 4.0 * 4.0 = 96.0

    the gradient itself is stored in x.grad

Rules of the Autograd:

    1. Autograd stores only the final output x.grad. it doesnt store any intermediate grad values like y.grad

    2. Drop the value after needed adjustments.

    use .grad.zero_() to drop the gradient value to zero. When you used a gradient value and adjusted, drop the value to zero, so next time you use it, it wont get overwritten and will be like a brand new.
'''



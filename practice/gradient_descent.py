from ftplib import all_errors
import numpy as np
from plot_prepare import *
import matplotlib.pyplot as plt

true_w = 2  # weights or slope of line
true_b = 1  # intercept
N = 100

np.random.seed(42)
# y = wx + b + noise
noise = 0.1 * np.random.randn(N, 1)

x = np.random.rand(N, 1)
y = true_b + true_w * x +  noise 

# Manual train test validation split

idx = np.arange(N)
np.random.shuffle(idx)

train_idx = idx[:int(N*0.8)]  # first 80 records from the shuffles indices
val_idx = idx[int(N*0.8):]

x_train,y_train = x[train_idx],y[train_idx]  # take the values from x and y, that correspnd to the randomized indices
x_val,y_val = x[val_idx], y[val_idx]


# figure1(x_train,y_train,x_val,y_val); plt.show()

###################
#  we knew the values of b and w , because we set them, duhh !!
#  Let's randomly initialize them

np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)

print(b,w)

# Let's compute the model's prediction values based on this "deduced (random) w,b values"

yhat = w* x_train + b

# figure2(x_train, y_train,b, w); plt.show()

# let's find the loss

error = (yhat - y_train)
loss = (error **2).mean()
print(loss)

# Visualizing the loss surface
b_range = np.linspace(true_b-3, true_b + 3, N+1)
w_range = np.linspace(true_w-3, true_w + 3, N+1)

bs, ws = (np.meshgrid(b_range, w_range)) # bs is a collection of possible b and w values respectively

# pick an x value and let's calculate for all range of possible b and w values



# we want to multiply the same x value by every
# entry in the ws matrix. 
dummy_x = x_train[0]
dummy_yhat = bs + ws * dummy_x

# This operation resulted in a grid of
# predictions for that single data point. Now we need to do this
# for every one of our 80 data points in the training set
all_predictions = np.apply_along_axis(
    func1d=lambda x: bs + ws * x,
    axis=1,
    arr=x_train
)

print(all_predictions.shape)  # shape (80,101,101)

# let's restructre the y labels

all_labels = y_train.reshape(-1,1,1) # shape (80,1,1)

all_errors = all_predictions - all_labels  # shape (80, 101,101)


all_losses = (all_errors**2).mean(axis=0) # axis 0 refers to the each of the 80 instances or 1st dimension of (80,101,101)

figure4(x_train,x_train,b,w, bs,ws, all_losses); plt.show()






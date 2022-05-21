import numpy as np
from chapter0 import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler



true_b = 1 # intercept
true_w = 2 # weight or slope
N = 100

# Data Generation
np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (.1 * np.random.randn(N, 1))

# y = wx + b + noise
y = true_b + true_w * x + epsilon

# Manual train test validation split

# Shuffles the indices
idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
# Uses the remaining indices for validation
val_idx = idx[int(N*.8):]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

figure1(x_train,y_train,x_val,y_val); plt.show()

###################
#  we knew the values of b and w , because we set them, duhh !!
#  Let's randomly initialize them

# Step 0 - Initializes parameters "b" and "w" randomly
np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)

print(b, w)

# Let's compute the model's prediction values based on this "deduced (random) w,b values"
# Step 1 - Computes our model's predicted output - forward pass
yhat = b + w * x_train

figure2(x_train, y_train,b, w); plt.show()

# let's find the loss

# Step 2 - Computing the loss
# We are using ALL data points, so this is BATCH gradient
# descent. How wrong is our model? That's the error!
error = (yhat - y_train)

# It is a regression, so it computes mean squared error (MSE)
loss = (error ** 2).mean()
print(loss)

# Reminder:
# true_b = 1
# true_w = 2

# we have to split the ranges in 100 evenly spaced intervals each
b_range = np.linspace(true_b - 3, true_b + 3, 101)
w_range = np.linspace(true_w - 3, true_w + 3, 101)
# meshgrid is a handy function that generates a grid of b and w
# values for all combinations
bs, ws = np.meshgrid(b_range, w_range)
bs.shape, ws.shape

# pick an x value and let's calculate for all range of possible b and w values



# we want to multiply the same x value by every
# entry in the ws matrix. 
# 
# dummy_x = x_train[0]
# dummy_yhat = bs + ws * dummy_x
# print(dummy_yhat.shape)

# This above operation resulted in a grid of
# predictions for that single data point. Now we need to do this
# for every one of our 80 data points in the training set
all_predictions = np.apply_along_axis(
    func1d=lambda x: bs + (ws * x),
    axis=1,
    arr=x_train
)

# print(all_predictions.shape)  # shape (80,101,101)

# let's restructre the y labels

all_labels = y_train.reshape(-1,1,1) # shape (80,1,1)

all_errors = all_predictions - all_labels  # shape (80, 101,101)
all_losses = (all_errors**2).mean(axis=0) # axis 0 refers to the each of the 80 instances or 1st dimension of (80,101,101)

figure4(x_train,y_train,b,w, bs,ws, all_losses); plt.show()

# at constant b
figure5(x_train,y_train, b, w, bs, ws, all_losses); plt.show()

# at constant w
figure6(x_train,y_train,b,w, bs,ws, all_losses); plt.show()

# finding the gradients for b and w

img = mpimg.imread('images\gradient_descent_grad_calc.png')
plt.imshow(img); plt.axis('off'); plt.grid(b=None); plt.show()

# Step 3 - Computes gradients for both "b" and "w" parameters
b_grad = 2 * error.mean()
w_grad = 2 * (x_train * error).mean()
print(b_grad, w_grad)


figure7(b, w, bs, ws, all_losses); plt.show()
'''
From the "Cross-Sections" section, we already know that to
minimize the loss, both b and w needed to be increased. So,
keeping the spirit of using gradients, let’s increase each
parameter a little bit (always keeping the other one fixed!). By
the way, in this example, a little bit equals 0.12 (for convenience
sake, so it results in a nicer plot)
'''
figure8(b, w, bs, ws, all_losses); plt.show()

img = mpimg.imread('images\gradient_descent_man_grad_calc.png')
plt.imshow(img); plt.axis('off'); plt.grid(b=None); plt.show()

# Sets learning rate - this is "eta" ~ the "n" like Greek letter
lr = 0.1
# print(b, w)

# Step 4 - Updates parameters using gradients and the 
# learning rate
b = b - lr * b_grad
w = w - lr * w_grad

# print(b, w)

figure9(x_train, y_train, b, w); plt.show()


# Manual Learning Rate
manual_grad_b = -2.90
manual_grad_w = -1.79

np.random.seed(42)
b_initial = np.random.randn(1)
w_initial = np.random.randn(1)

# Low learning rate
# Learning rate - greek letter "eta" that looks like an "n"
lr = .2

figure10(b_initial, w_initial, bs, ws, all_losses, manual_grad_b, manual_grad_w, lr); plt.show()

# High learning rate
# Learning rate - greek letter "eta" that looks like an "n"
lr = .8

figure10(b_initial, w_initial, bs, ws, all_losses, manual_grad_b, manual_grad_w, lr); plt.show()

# Very High Learning rate - greek letter "eta" that looks like an "n"
lr = 1.1

figure10(b_initial, w_initial, bs, ws, all_losses, manual_grad_b, manual_grad_w, lr); plt.show()


"""
BAD Feature
"""

true_b = 1
true_w = 2
N = 100

# Data Generation
np.random.seed(42)

# We divide w by 10
bad_w = true_w / 10
# And multiply x by 10
bad_x = np.random.rand(N, 1) * 10

# So, the net effect on y is zero - it is still
# the same as before
y = true_b + bad_w * bad_x + (.1 * np.random.randn(N, 1))

# Generates train and validation sets
# It uses the same train_idx and val_idx as before,
# but it applies to bad_x
bad_x_train, y_train = bad_x[train_idx], y[train_idx]
bad_x_val, y_val = bad_x[val_idx], y[val_idx]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(x_train, y_train)
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_ylim([0, 3.1])
ax[0].set_title('Train - Original')
ax[1].scatter(bad_x_train, y_train, c='k')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_ylim([0, 3.1])
ax[1].set_title('Train - "Bad"')
fig.tight_layout(); plt.show()

# The ranges CHANGED because we are centering at the new minimum, using "bad" data
bad_b_range = np.linspace(-2, 4, 101)
bad_w_range = np.linspace(-2.8, 3.2, 101)
bad_bs, bad_ws = np.meshgrid(bad_b_range, bad_w_range)

figure14(x_train, y_train, b_initial, w_initial, bad_bs, bad_ws, bad_x_train); plt.show()

# how does this compare with gradients of previous setting

figure15(x_train, y_train, b_initial, w_initial, bad_bs, bad_ws, bad_x_train); plt.show()



scaler = StandardScaler(with_mean=True, with_std=True)
# We use the TRAIN set ONLY to fit the scaler
scaler.fit(x_train)

# Now we can use the already fit scaler to TRANSFORM
# both TRAIN and VALIDATION sets
scaled_x_train = scaler.transform(x_train)
scaled_x_val = scaler.transform(x_val)

fig, ax = plt.subplots(1, 3, figsize=(15, 6))
ax[0].scatter(x_train, y_train, c='b')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_ylim([0, 3.1])
ax[0].set_title('Train - Original')
ax[1].scatter(bad_x_train, y_train, c='k')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_ylim([0, 3.1])
ax[1].set_title('Train - "Bad"')
ax[1].label_outer()
ax[2].scatter(scaled_x_train, y_train, c='g')
ax[2].set_xlabel('x')
ax[2].set_ylabel('y')
ax[2].set_ylim([0, 3.1])
ax[2].set_title('Train - Scaled')
ax[2].label_outer()

fig.tight_layout(); plt.show()


# The ranges CHANGED AGAIN because we are centering at the new minimum, using "scaled" data
scaled_b_range = np.linspace(-1, 5, 101)
scaled_w_range = np.linspace(-2.4, 3.6, 101)
scaled_bs, scaled_ws = np.meshgrid(scaled_b_range, scaled_w_range)

figure17(x_train, y_train, scaled_bs, scaled_ws, bad_x_train, scaled_x_train); plt.show()

figure18(x_train, y_train)
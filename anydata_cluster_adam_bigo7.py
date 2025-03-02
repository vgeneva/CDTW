
# Import


import numpy as np
#import matplotlib.pyplot as plt
np.random.seed();

#from matplotlib import pyplot as plt
import pandas as pd
from numpy import linalg as LA
#from scipy.integrate import solve_ivp  # solve ODE
#from matplotlib.animation import PillowWriter    # writing the .gif
import time
import tensorflow as tf
import random

from scipy.interpolate import interp1d
import tensorflow_probability as tfp
import argparse
import sys
import os


"""# Length of signal $x(0)$"""

def i_traj(n, a = 0, b = 1):
  """ Args
    n: length of signal
    a: start of interval, default 0
    b: end of interval, default 1
  """
  h_x = (b-a)/n #step size
  k_x = (round((b-a)/h_x)) #int makes it round down. round, lets it round
  x_ini = tf.linspace(start = a, stop = b, num = n) #creates a tensor of n points
  x = tf.reshape(x_ini, (n,1)) #reshape into a column vector
  return tf.cast(x, dtype = tf.float64)     #return as a float64

"""## Select x(0) $\in \mathbb{R}^n$, where $n = 256$.

Since our trajectories have 256 initial trajectories, we choose 256

Define $x(0)$ vector. I will start with 256.
"""

# input n for i_traj function to get x(0) 


try:
   signal_length = input("Input signal length, i.e. 256: ")
   signal_length = int(signal_length)
   x0 = i_traj(signal_length)
except ValueError:
    print("Invalid input. Please enter an integer.")
    sys.exit(1)



#x0 = i_traj(256)



"""# Bring in CSVs

## Data with n length signals.
Train Matrices are (n, t), where t = class time quanity of classes

Test Matrices are (n, m), where m = quantity of a class
"""

# define a function that brings in the data
# 


def load_matrices():
    """
    Prompts the user for CSV file names (without '_matrix.csv') and loads them as TensorFlow tensors.
    Assumes all CSV files are stored in '/home/vhaney/EKG/filtered_data/'.
    """
    #base_path = "/home/vhaney/EKG/filtered_data/nonresample/"
    base_path = "/home/vhaney/EKG/current_filtered_data/"

    while True:
        # Prompt user for the tensor name (without '_matrix.csv')
        tensor_name = input("Enter the dataset name (e.g., 'Beef_train' or 'Beef1_test') or press Enter to finish: ").strip()
        
        if not tensor_name:  # Stop if the user presses Enter
            print("Data input complete. Loading finished.\n")
            break
        
        # Construct the tensor name by appending '_matrix' to the user input
        tensor_name_with_matrix = tensor_name + '_matrix'
        
        # Construct full file path
        csv_file = os.path.join(base_path, f"{tensor_name_with_matrix}.csv") # assuming the CSV files are named like 'Beef_train_matrix.csv'
        
        # Try loading the file
        try:
            df = pd.read_csv(csv_file, header=None, skiprows=1)  # Assuming no header in CSV
            
            # Convert DataFrame to a TensorFlow tensor and store in globals()
            globals()[tensor_name_with_matrix] = tf.convert_to_tensor(df.values, dtype=tf.float64)
            
            print(f"Loaded {tensor_name_with_matrix} from {csv_file} with shape {globals()[tensor_name_with_matrix].shape}\n")
        
        except Exception as e:
            print(f"Error loading {csv_file}: {e}\n")

# Run the function
load_matrices()



#print(Beef1_test_matrix.shape)
#print(Beef1_test_matrix)
#print(Beef_train_matrix.shape)
#print(Beef_train_matrix)


"""

N_train_matrix = tf.convert_to_tensor(pd.read_csv('N_train_matrix.csv'), dtype = tf.float64)
N_test_matrix = tf.convert_to_tensor(pd.read_csv('N_test_matrix.csv'), dtype = tf.float64)

V_train_matrix = tf.convert_to_tensor(pd.read_csv('V_train_matrix.csv'), dtype = tf.float64)
V_test_matrix = tf.convert_to_tensor(pd.read_csv('V_test_matrix.csv'), dtype = tf.float64)

A_train_matrix = tf.convert_to_tensor(pd.read_csv('A_train_matrix.csv'), dtype = tf.float64)
A_test_matrix = tf.convert_to_tensor(pd.read_csv('A_test_matrix.csv'), dtype = tf.float64)

R_train_matrix = tf.convert_to_tensor(pd.read_csv('R_train_matrix.csv'), dtype = tf.float64)
R_test_matrix = tf.convert_to_tensor(pd.read_csv('R_test_matrix.csv'), dtype = tf.float64)

L_train_matrix = tf.convert_to_tensor(pd.read_csv('L_train_matrix.csv'), dtype = tf.float64)
L_test_matrix = tf.convert_to_tensor(pd.read_csv('L_test_matrix.csv'), dtype = tf.float64)

E_train_matrix = tf.convert_to_tensor(pd.read_csv('E_train_matrix.csv'), dtype = tf.float64)
E_test_matrix = tf.convert_to_tensor(pd.read_csv('E_test_matrix.csv'), dtype = tf.float64)

J_train_matrix = tf.convert_to_tensor(pd.read_csv('J_train_matrix.csv'), dtype = tf.float64)
J_test_matrix = tf.convert_to_tensor(pd.read_csv('J_test_matrix.csv'), dtype = tf.float64)

g_train_matrix = tf.convert_to_tensor(pd.read_csv('jj_train_matrix.csv'), dtype = tf.float64)
g_test_matrix = tf.convert_to_tensor(pd.read_csv('jj_test_matrix.csv'), dtype = tf.float64)

#jj_train_matrix = tf.convert_to_tensor(pd.read_csv('jj_train_matrix.csv'), dtype = tf.float64)
#jj_test_matrix = tf.convert_to_tensor(pd.read_csv('jj_test_matrix.csv'), dtype = tf.float64)

# bring in all_train_matrix as a tensor 
all_train_matrix = tf.convert_to_tensor(pd.read_csv('all_train_matrix.csv'), dtype = tf.float64)
"""


"""# Kernel Functions"""

# A collection of vectorized kernel functions
# x, and y are assumed to be Matrices
# if you want to evaluate on a single point
# reshape the vectors from d to d x 1, this
# will turn them into matrices. All the kernel
# functions work with vectors or matrices with the
# exception of k5v which requires matrices as inputs.

# Linear kernel, notice, if x, and y
# are vectors, this will return a scalar
# if x, and y are matrices, this will
# return a matrix and this is an efficient
# and vectorized implementation of the linear
# kernel function.
def k1v(x,y):
    # np.dot computes the vector dot
    # product if x, and y are vectors
    # or it computes the matrix product
    # of x and y, if x or y are matrices
    return (np.dot(x,y.T))

# Affine kernel. Again, if x and y are
# vectors, this will return a scalar but
# if x, and y are matrices this is a vectorized
# version of the affine kernel.
def k2v(x,y):
    return (np.dot(x,y.T)+1)

# Quadratic kernel. Again, if x and y are
# vectors, this will return a scalar but
# if x, and y are matrices this is a vectorized
# version of the Quadratic kernel.
def k3v(x,y):
    # To raise a number by a power python
    # uses ** instead of ^, ^ is reserved
    # for bit operations. (and?)
    return ((1+np.dot(x,y.T))**2)

# dth order kernel, (here d=4)
d = 4
def k4v(x,y):
    return ((1+np.dot(x,y.T))**d)

# Gaussian kernel. This is the only kernel function
# in this set of example kernels that is tricky to
# efficiently vectorize. This is because to evaluate
# this kernel function, we have to evaluate (x-y)^T(x-y)
# whereas in all the other examples, we evaluate x^Ty
# Expanding (x-y)^T(x-y) gives us x^Tx - 2x^Ty + y^Ty
# So k(x_i,y_j) = e^(-(x_i-y_j)^T(x_i-y_j)/(2*sigma^2))
# So k(x_i,x_j) = f(x_i^Tx_i+y_j^Ty_j-2x_i^Ty_j)
# we get x_i^Ty_j using np.dot(X,Y) where X and Y are the
# X and Y data matrices respectively. We get x_i^Tx_i using
# np.sum(X*X,axis=1).reshape(d,1). This function requires
# matrices and will fail if x, or y are vectors. It is
# vectorized and efficient. For any kernel function that uses
# k(x_i,x_j) = f((x_i-x_j)^T(x_i-x_j)) use the line that
# computes XminusYSquared to compute the matrix whose i,jth
# entry is (x_i-x_j)^T(x_i-x_j) in a vectorized and efficient
# manner. Then apply f(XminusYsquared).

# sigma can be picked for the specific problem at hand
sigma = 2**0;
def k5v(x,y):
    numxPoints = np.shape(x)[0];
    numyPoints = np.shape(y)[0];
    XminusYSquared =np.sum(x*x,axis=1).reshape([numxPoints,1])-2*np.dot(x,y.T)+np.sum(y*y,axis=1).reshape([1,numyPoints]);
    return np.exp(-XminusYSquared/(2*sigma**2))


def gauss_0(x,y): # zero boundary conditions
    numxPoints = np.shape(x)[0];
    numyPoints = np.shape(y)[0];
    XminusYSquared =np.sum(x*x,axis=1).reshape([numxPoints,1])-2*np.dot(x,y.T)+np.sum(y*y,axis=1).reshape([1,numyPoints]);
    return np.exp(-XminusYSquared/(2*sigma**2))



def k7v(x,y):
    return np.exp(-1/(2*sigma**2) * (x-y)**2)

"""# Variables for Random Fouirer Features

\begin{align}
    \gamma(x) = \frac{1}{\sqrt{2p}} \begin{bmatrix}
        \cos(z_1x) \\
        \vdots\\
        \cos(z_{p/2}x)\\
        \sin(z_1x)\\
        \vdots\\
        \sin(z_{p/2}x)
    \end{bmatrix}
\end{align}
"""

num_features = 50

feature_coeff = 1 / np.sqrt(num_features)

"""## Z's for guassian random Fourier features for $\gamma$

"""

np.random.seed(3) # set the seed for reproducibility
# creating fourier samples (create a vector of random numbers to be the scalar inputs, which are x)
# in the feature_map, we have cosines and sines and we have num_features cosines and
# num_features sines for the fourier features. Our input is num_features, but our vector col will be
# 2 * num_features.

# we need to make our standard deviation 1/sqrt(\sigma^2). Here , when I selet
# fourier, sample, its with sigma = 1, so this is 1\sqrt (1).

#these are the randomly picked z's for the fourier-features
fourier_sample_gamma = np.random.normal(scale = 1, size = (num_features,1))
fourier_sample_gamma = tf.constant(fourier_sample_gamma, dtype = tf.float64) #make into a tensor

#print(fourier_sample_gamma.shape)

"""# Random Fourier Features

## Function for $\gamma(x)$.

First we leverage Michael's code to create a fourier features map we denote as $\gamma(x)$. Our inputs for this function are: an $x$ value, fourier_sample (which are scaling the inputs into the $\sin$ and $\cos$ components of the vector), and a feature coefficient which in our case is $1 / \sqrt{\text{number of features}}$. \begin{align}
    \gamma(x) = \frac{1}{\sqrt{2p}} \begin{bmatrix}
        \cos(z_1x) \\
        \vdots\\
        \cos(z_{p/2}x)\\
        \sin(z_1x)\\
        \vdots\\
        \sin(z_{p/2}x)
    \end{bmatrix}
\end{align}
"""

#feature map: Computes a feature map for a Gaussian kernel Fourier features approximation.
#Input:
#x - (n,d) matrix of n d-vectors to perform feature map on
#fourier_sample - (d,p) matrix of p samples from a d-dimensional standard normal distribution
#feature_coeff - 1/sqrt(p).  This could be computed from fourier_sample.
#Output:
#M - (2p+1,n) matrix of n feature vectors
#The +1 comes from the fact that this feature vector incorporates a constant term
# in regards to bandwidth, we are sampling from a normal distribution but using the guassian kernel.
# so we need to make our standard deviation 1/sqrt(\sigma^2). Here , when i selet
# fourier, sample, its with sigma = 1, so this is 1\sqrt (1).
def feature_map(x,fourier_sample,feature_coeff,bandwidth = .1):  #original 0.1#bandwith is the sigma

    x = tf.transpose(x) #make x (n,d) into a row tensor (if d=1)
    # fourierx: fourier_sample is an outer product which creates a
    # matrix which its rows are scaled x row vectors by each fourier_sample
    # (p, n)
    fourierx = tf.linalg.matmul(fourier_sample,x)
    #print(f' fourier x: {fourierx.shape}')
    # then take cosine, stiill be (p,n)
    cos_fourierx = feature_coeff*tf.cos(fourierx/bandwidth)
    #print(f' cos_fourierx: {cos_fourierx.shape}')
    sin_fourierx = feature_coeff*tf.sin(fourierx/bandwidth)
    #print(f' sin_fourierx: {sin_fourierx.shape}')
    M = tf.concat([cos_fourierx,sin_fourierx],axis=0)
    #print(f' M: {M.shape}')
    return M

"""# feature map matrix style"""

def feature_map_m(x,fourier_sample,feature_coeff,bandwidth = .1):  #original 0.1#bandwith is the sigma

    '''
    feature map provides evaluation for every x component in a matrix.
    if x \in (n,1), the out put will be (1,2*p,n) since there are p features (num_features)

    Args:
    x: a tensor (n, m). (256,m) n-dimensions, m, how columns many are there
    fourier_sample: a tensor (p, 1). 2*p is how many features we have in total

    featuer_coeff: constant
    bandwidth: constant

    Returns (m, 2*p, n) tensor

    '''
    n = len(x) #how many rows in the vector
    m = len(x[0]) # how many columns are there
    x = tf.transpose(x) # transpose x to: (m,n)
    x = tf.expand_dims(x, axis = 1) # makes 1 dimension in rows (m,1,n)
    #print(f'x new shape should be ({m}, 1, {n}): {x.shape}')

    p = num_features
    fourierx  = tf.expand_dims(fourier_sample, axis = 0) # add's the first indexed dimension, i.e. debth 'm'
    fourierx= tf.tile(fourierx, [m, 1, 1]) # the 1's keep what it has in that dimension (m, p, 1)
    #print(f' fourier tiling shape should be ({m}, {p}, 1): {fourierx.shape}')    # (m, p, 1)
    fourierx = tf.linalg.matmul(fourier_sample,x) # This will ben (m, )
    #print(f' fourierx shape should be ({m}, {p}, {n}): {fourierx.shape}')

    # then take cosine,
    cos_fourierx = feature_coeff*tf.cos(fourierx/bandwidth)
    #print(f' cos_fourierx: {cos_fourierx.shape}')
    sin_fourierx = feature_coeff*tf.sin(fourierx/bandwidth)
    #print(f' sin_fourierx: {sin_fourierx.shape}')
    M = tf.concat([cos_fourierx,sin_fourierx],axis=1)
    #print(f' Output should be: ({m}, {2*p}, {n}): {M.shape}')
    return M

"""## k9v, Kernel function for Guassian Random Fourier Features, $k(x,y) = \gamma(x)^T\gamma(y)$"""

# normal guassian
def k9v(x,y,
        fourier_sample = fourier_sample_gamma,
        feature_coeff = feature_coeff,
        feature_map = feature_map):

    return tf.linalg.matmul(tf.transpose(feature_map(x, fourier_sample, feature_coeff))
                                                       , feature_map(y, fourier_sample
                                                       , feature_coeff))

"""## K9v Guassian explicit in matrix form"""

# normal guassian matrix function
def k9v_m(x,y,
          fourier_sample = fourier_sample_gamma,
          feature_coeff = feature_coeff,
          feature_map = feature_map_m):
  '''
  Args:
  x \in (n, m)
  y \in (n ,m)
  fourier_sample \in (p,1)
  feature_coeff = scaler
  feature_map: default feature_map_m, matrix valued feature map, (m, 2*p, n)
  '''


  fmx = feature_map(x, fourier_sample, feature_coeff)
  fmy = feature_map(y, fourier_sample, feature_coeff)
  return tf.linalg.matmul(tf.transpose(fmx, perm = [0 , 2, 1]), fmy)

"""## k8v, Now create projection kernel function for $\gamma_0(x) = \Lambda^{1/2} U^T \gamma(x)$, $k_0(x,y) = \gamma_0(x)^T \gamma_0(y)$.

This is the explicit kernel function, but with ensuring we start and end at the same place for phi_0 and phi_1.

Using $k_0(x,y) = \gamma(x)^T A \gamma(y)$, where $A$ is a Hermitian matrix described below and in the notes of Dr. Bruno.

When we generate our $\alpha$ function, we ensured that the $\alpha(0) = \alpha(1) = 0$.  We did this by following Dr. Bruno's notes on A RKHS constrained to functions verifying $\alpha(0) = 0$ and $\alpha(1) = 0$. We first start by considering an explicit kernel with random fourier features: $k(x,y)  = \gamma(x)^T \gamma(y)$, k9v above

First make tensor flow arrays for $x_1 = [1], x_0 =[0], x_{01} =[0 ,1 ]$.
"""

x_01 = np.array([0.,1.])
# reshape x_1 into a column vector
x_01 = np.reshape(x_01, (len(x_01),1))
# make numpy array into a tensorflow array
x_01 = tf.constant(x_01, dtype = tf.float64)
#print(x_01.dtype)

x_0 = np.array([0.])
x_0 = np.reshape(x_0, (len(x_0),1))
# make numpy array into a tensorflow array
x_0 = tf.constant(x_0, dtype = tf.float64)
#print(x_0.dtype)

x_1 = np.array([1.])
x_1 = np.reshape(x_1, (len(x_1),1))
# make numpy array into a tensorflow array
x_1 = tf.constant(x_1, dtype = tf.float64)
#print(x_1.dtype)

"""Now, let's create a gram matrix with our end points, $x = [0,1]$. Here we make  
G =\begin{bmatrix} k(0,0) & k(0,1) \\
                      k(0,1) & k(1,1)
      \end{bmatrix}
"""

G = k9v(x_01,x_01)
#print(G)
G_inv = tf.linalg.inv(G)
#print(G_inv)

"""$(\gamma(0),\gamma(1))$, with $\gamma(0), \gamma(1) \in \mathbb{R}^p$."""

g0g1 = feature_map(x_01, fourier_sample = fourier_sample_gamma, feature_coeff = feature_coeff)
#print(g0g1.shape)

"""$(\gamma(0), \gamma(1)) G^{-1} (\gamma(0), \gamma(1))^T$"""

gGg = tf.linalg.matmul(tf.linalg.matmul(g0g1, G_inv), tf.transpose(g0g1))
#print(gGg.dtype)
#print(gGg.shape)

I = tf.eye(2 * num_features, dtype=tf.float64)

"""$A = I - (\gamma(0), \gamma(1)) G^{-1} (\gamma(0), \gamma(1))^T$"""

A_g = tf.subtract(I,gGg)
#print(A_g.shape)
#print(A_g)

# A Hermitian matrix is a square matrix that is equal to its own complex conjugate transpose.
#if np.allclose(A_g, np.conj(A_g).T):
#    print("A is Hermitian")
#else:
#    print("A is not Hermitian")

# Use eigh
eig, U = LA.eigh(A_g)

# Define the condition for removing small values from
# eigenvalues. These are basically zero
condition = eig < 1e-14 #had to change to 1e-3 due to float 64 to 32

# Use boolean indexing to select the elements that satisfy the condition
# this is telling me what meets the condition
remove = tf.boolean_mask(eig, condition)
# Obtain indexes where this condition is met
ind_to_remove = np.where(condition)

# new eig vector with no zeros
new_eig = np.delete(eig, ind_to_remove)
#new_eig = np.where(condition, eig, 0)

# new U matrix removing the eigenvectors corresponding to
# eigenvalues that were removed
new_U = np.delete(U, ind_to_remove, axis=1)


#print(remove)
#print(ind_to_remove)
#print(new_eig)
#print(new_eig.shape)
#print(new_U.shape)
#print(new_U[:,0].shape)

new_L = tf.linalg.diag(tf.sqrt(new_eig))
#print(new_L.shape)

"""## $Λ^{1/2}U^T =  LU $

will be the same, so I don't need to recalculate every time we go through the equation.
"""

LU = tf.linalg.matmul(new_L, tf.transpose(new_U))
#print(LU.shape)

"""$\gamma_0(x) = Λ^{1/2}U^T \psi(x)$"""

# new feature map to make alpha and learning alpha
def feature_map_0(x, Lambda_unitary = LU, fourier_sample = fourier_sample_gamma,
                  feature_coeff = feature_coeff, bandwidth = .1):#, feature_map = feature_map,
  #print(x.shape)
  p = feature_map(x, fourier_sample, feature_coeff, bandwidth)
  return tf.linalg.matmul(Lambda_unitary, p)

"""## Matrix projectied feature map matrix"""

#LU.shape
LU_m = tf.expand_dims(LU, axis = 0)
#print(LU_m.shape)
LU_m = tf.tile(LU_m, [10, 1, 1])
#print(LU_m.shape)

# new feature map to make alpha and learning alpha
def feature_map_m_0(x, Lambda_unitary = LU, fourier_sample = fourier_sample_gamma,
                  feature_coeff = feature_coeff, bandwidth = .1):#, feature_map = feature_map,
  #print(x.shape)
  m = len(x[0]) #how many columns are there of n-dim vectors
  #print(m)
  LU_m = tf.expand_dims(LU, axis = 0) #add depth for 'm'
  #print(LU_m.shape)
  LU_m = tf.tile(LU_m, [m, 1,1]) # make m copies of LU_m (m, (2*p-2), 2*p)
  LU_m = tf.cast(LU_m, dtype = tf.float64)
  #print(LU_m.dtype)
  p = feature_map_m(x, fourier_sample, feature_coeff, bandwidth) #(m, 2*p, n) matrix
  return tf.linalg.matmul(LU_m, p) #should be (m,2*p, n)

"""## New kernel function for $\gamma_0(x) = \Lambda^{1/2} U^T \gamma(x)$ such that $k_0(x,y) = \gamma_0(x)^T \gamma_0(y)$.

This is the explicit kernel function, but with ensuring we start and end at the same place as phi_0 and phi_1.
"""

# Remeber, LU will be fixed the entire time, will not change since it is about
# gamma(x) at the boundaries only.
def k8v(x,y, Lambda_unitary = LU, fourier_sample = fourier_sample_gamma,
                  feature_coeff = feature_coeff):#, feature_map = feature_map):

    return tf.linalg.matmul(tf.transpose(feature_map_0(x, Lambda_unitary
                                                       , fourier_sample
                                                       , feature_coeff ))#, feature_map))
                                                       , feature_map_0(y, Lambda_unitary
                                                                       , fourier_sample
                                                                       , feature_coeff ))#, feature_map))

"""## NEW Kernel function for feature_map_matrix_0"""

#explict gausssain using projection function
def k8v_m(x,y
          , Lambda_unitary = LU
          , fourier_sample = fourier_sample_gamma
          , feature_coeff = feature_coeff
          , feature_map = feature_map_m_0):
  '''
  Args:
  x \in (n, m).
  y \in (n ,m)
  fourier_sample \in (p,1)
  feature_coeff = scaler
  LU: lambda_unitary matrix (2*p-2, 2*p), projection matrix
  feature_map: default feature_map_m, matrix valued feature map, (m, 2*p, n)

  return: (m, n,n)
  '''


  fmx = feature_map(x, Lambda_unitary, fourier_sample, feature_coeff)
  #print(fmx.shape) #(m, (2*p-2),n)
  fmy = feature_map(y, Lambda_unitary, fourier_sample, feature_coeff)
  #print(fmy.shape) #(m, (2*p-2),n)
  return tf.linalg.matmul(tf.transpose(fmx, perm = [0 , 2, 1]), fmy)

"""# alpha function using projected feature map

i.e. $\alpha(0) = 0= \alpha(1)$. $\beta$ must be length $(2 *$ numfeatures) - 2 for dimensions.

$\alpha(x) = \beta^T \gamma_0(x)$

## alpha function that gives (1, n)
"""

#beta_true is shorter by 2 dimensions
def alpha(t, x, b,
           Lambda_unitary = LU,
           fourier_sample = fourier_sample_gamma,
           feature_coeff = feature_coeff):
  return tf.linalg.matmul(tf.transpose(b), feature_map_0(x))

"""## alpha function input X as a matrix of vectors and Beta as matrix of betas, gives (m,n) matrix. NO for loop

"""

def alpha_m(t, x, b,
            Lambda_unitary = LU,
            fourier_sample = fourier_sample_gamma,
            feature_coeff = feature_coeff):
  '''
  Args:
  x \in (n,m): m n-dim vectors input. (n,m)
  b \in ((2*p-2, m))
  def alpha_test_m(x, b = beta_t, fourier_sample = fourier_sample_gamma, feature_coeff = feature_coeff):

  Returns M: (m,n) there are m functions provided for each n-dimensional vector
  I want it to return a (m,n) matrix (m trajectories, n -dimensional)
  '''
  m = len(x[0]) #columns of x
  n = len(x) #rows of x
  b = tf.transpose(b) #(m, (2*p-2))
  #print(f'b transposed: {b.shape}')
  b = tf.expand_dims(b, axis = 1) # #makes 1 dimension in rows (m,1,(2*p-2))
  #b = tf.tile(b, [m, 1, 1]) # makes dimension (m, d, (2*p-2))
  #print(f'b added m axis and tiled beta (m,d,2p): {b.shape}')
  #print(f'print feature_map shape: {feature_map_m(x,fourier_sample, feature_coeff).shape}')
  M = tf.linalg.matmul(b, feature_map_m_0(x, Lambda_unitary, fourier_sample, feature_coeff))
  M_reshape = tf.squeeze(M, axis= 1) # if axis is left alone, tf.squeeze will
                                     # get rid of dimension1. so if x i (n,1),
                                    # it would get rid of the first and second dimesnions
                                    # so it's important to have axis = 1 in order to rid of
                                    # the middle '1' dimesion only.
                                    # i want a (m,n) matrix
  #print(f'M shape should be ({m}, 1, {n}): {M.shape}')
  return tf.cast(M_reshape, dtype = tf.float64) #returns (m,n)

# prompt: make a beta tensor (2*numfeatuers) - 2 dimensions
# Assuming 'num_features' is defined as in the provided code.
#beta_a = tf.Variable(tf.random.normal(shape=(2 * num_features - 2, 1), dtype=tf.float64))

#beta_a.shape

#beta_M = tf.Variable(tf.random.normal(shape=(2 * num_features - 2, 10), dtype=tf.float64))
#beta_M_np = beta_M.numpy()

#beta_M2 = tf.Variable(tf.random.normal(shape=(2 * num_features - 2, 20), dtype=tf.float64))
#beta_M2_np = beta_M.numpy()

#beta_M3 = tf.Variable(tf.random.normal(shape=(2 * num_features - 2, 30), dtype=tf.float64))
#beta_M3_np = beta_M.numpy()

# prompt: tile beta_a to be 98,2


#beta_a_tiled = tf.tile(beta_a, [1, 2])
#beta_a_tiled.shape

#alpha(1, x_0, beta_a)

#x_01.shape

#alpha_m(1, tf.transpose(x_01), beta_a_tiled)

#x0_alpha = alpha(1, x0, beta_a)
#print(x0_alpha.shape)

#print(x0_alpha)

#X0_M = tf.tile(x0, (len(x0[0]), len(x_01)))
#print(X0_M.shape)
#print(X0_M[0:5, 0:5])

#X0_M2 = alpha_m(1, X0_M, beta_a_tiled)

#print(X0_M2.shape)

#print(x0.shape)
#print(x_0)
#print(x_01)

#X_M = (tf.random.normal(shape=(256, 10), dtype=tf.float64))
#X_M_np = X_M.numpy()

#print(beta_M.shape)
#print(X_M.shape)

#A_1 = alpha_m(1, X_M, beta_M)
#print(A_1.shape)
#print(A_1[:, -10:])

#test_alpha = np.zeros((10,256))
#for i in range(10):
#    test_alpha[i,:] = alpha(1, X_M_np[:,i].reshape(256,1), beta_M_np[:,i].reshape(98,1))

#print(test_alpha[:, -10:])

#print(np.linalg.norm(test_alpha - A_1.numpy())/np.linalg.norm(test_alpha))

"""## Make time intervals for ODE


"""

# time intervals for my ODE integrator
num = 20
t_list = np.linspace(0, 1.0, num = num)
#print(t_list)

"""## Vectorized Euler Tensor flow

"""

#THIS IS WHAT I USE
def s_alpha_tf_m(x, b, a = alpha_m, num = num): #alpha_matrix is for loop
                                                      #alpha_m is vectorized
    """
    TensorFlow version of the `s_alpha` function.

    Args:
    - x: Input tensor (n,).
    - b: Parameter for the `a_func`.
    - num: Number of columns in the output tensor.
    - a_func: A function that computes values based on alpha, x, and b.

    - alpha_matrix(t, x, b, fourier_sample, feature_coeff)


    Returns:
    - A tensor of shape (n, d). n-dimensional vector,
      d how many vectors for t = 1

    - X is a list for each d vectors for each time t,
      not including 0. There are 'num' time steps.
    """
    # Initialize the output tensor with zeros
    #columns =[]
    #columns.append(x)
    #print(x.dtype)

    # Assign the first column and keeping the rest zeros
    #sa = tf.keras.layers.Concatenate(axis=1)((x0,sa[:, 1:]))

    # Iteratively compute the remaining columns

    #a = alpha_matrix(1, x, b)
    #print(tf.transpose(a).shape)

    #print(x.shape)
    X = []
    for i in range(1, num):

      x += (1 / (num - 1)) * tf.transpose(a(1, x, b))
      X.append(x)
      #print(x.shape)
      #print(x[0:5,:])
      #columns.append(x)

    #print(columns)

    return x,X

#print(beta_M.shape)
#print(X_M.shape)

#x_v, X_V = s_alpha_tf_m(X_M, beta_M)
#x_v = s_alpha_tf_m(X_M, beta_M)

#print(x_v)

#traj = np.zeros((256,19))
#for i in range(10):
#  for j in range(19):
#    traj[:,j] = X_V[j][:,i]
#  plt.plot(traj[2,:],traj[1,:])

#X_M3 = (tf.random.normal(shape=(256, 30), dtype=tf.float64))

#print(X_M3.shape)
#print(beta_M3.shape)
#print(num)

#x_v3, X_V3 = s_alpha_tf_m(X_M3, beta_M3)

#len(X_V3)

#traj3 = np.zeros((256,(num-1)))
#for i in range(30):
#  for j in range((num-1)):
#    traj3[:,j] = X_V3[j][:,i]
#  plt.plot(traj3[2,:],traj3[1,:])

#print(s_alpha_tf_m(X_M, beta_M))

#print(tf.transpose(alpha_m(1, x0, beta_a))[0:5])
#print(x0[0:5])
#print(s_alpha_tf_m(x0, beta_a)[0:5])

#print(s_alpha_tf_m(X0_M, beta_a_tiled))

"""# Phi_1 interpolation
I need to interpolate to obtain phi_1.
"""

def linear_inter(x, xp, fp):
    """
    Use TensorFlow Probability's interpolation for 1D linear interpolation.
    Args:
        x: Tensor of query points.
        xp: List/Tensor of known x-coordinates (must be equally spaced).
        fp: List/Tensor of known y-coordinates.
    Returns:
        Tensor (n, 1) Interpolated values at query points `x`.
    """
    # Ensure xp is a regular grid
    xp = tf.convert_to_tensor(tf.squeeze(xp), dtype=tf.float64) #need 0-d
    fp = tf.convert_to_tensor(tf.squeeze(fp), dtype=tf.float64) #need 0-d

    # Ensure x is within the range of xp
    x_min, x_max = xp[0], xp[-1]
    x = tf.clip_by_value(x, x_min, x_max)

    # Use tfp interp_regular_1d_grid
    interpolated_values = tfp.math.interp_regular_1d_grid(
        x=x,
        x_ref_min=x_min,
        x_ref_max=x_max,
        y_ref=fp,
        axis=0
    )
    return interpolated_values

"""## New phi_1 vectorized interpolation function"""

# This is a function that takes in matrices. x_targ: matrix of target x's to be
# interpolated. x_orig: the original x's and y_vals, the original y_vals.
#

def int_vectorized(inputs):
  """
  Args:
  x_targ: x_targ is a transposed matrix. The original matrix has a certain amount of columns
          that need interpolation.  This matrix must be transposed so that the columns
          are rows.
  x_orig: x_orig is a transposed matrix. The original matrix has a certain amount of
          columns that are the original x values that correspond to the y_vals.  This
          matrix must be transposed so that the columns are rows.
  y_vals: y_vals is a tranposed matrix. The original matrix has a certain amount of
          columns that are the originnal y values that correspond to the x_orig pre
          transposed matirx. This matrix must be tranposed so that the columns are
          rows.
  This function is used along with
        tf.vectorized_map(int_vectorized, (x_targ, x_orig, y_vals))

        For the objective function, x_targ will be the x_traj_m, x_orig
        will be X_M, and y_vals will be BIG_PHI_ONE, but it must be transposed

        tf.transpose(tf.vectorized_map(int_vectorized, (x_targ, x_orig, y_vals)))
  """
  x_targ, x_orig, y_vals = inputs
  return tfp.math.interp_regular_1d_grid(x_targ, x_orig[0], x_orig[-1], y_vals)

"""# Objective function

$$ J_i(\beta) = \psi_i(\beta^T \gamma(\cdot), x_0) + \lambda \beta^T \beta$$
Where
$$\psi_i(\beta^T \gamma(\cdot), x_0) = \left( \phi_1(x_i(1)) - \phi_0(x_i(0))\right)^2$$
And
$$ \psi(\beta^T \gamma(\cdot),x_0) = \frac{1}{n} \sum_{i=1}^n \psi_i(\beta^T \gamma(\cdot),x_0) $$
So
\begin{align}
J(\beta)
& = \frac{1}{n} \sum_{i=1}^n  \left( \phi_1(x_i(1)) - \phi_0(x_i(0))\right)^2 +\lambda \beta^T \beta
\end{align}
"""

#len(N_train_matrix[0])

"""## Try objective in a vectorized way
  Args:

  x: initial trajectories at time = 0

  b: beta matrix (2*p-2,m), where m is len(phi_0) * len(phi_1)

  Within function Computes (x_t1: estimated x @ time 1)

\begin{align}
J(\beta)
& = \frac{1}{n} \sum_{i=1}^n  \left( \phi_1(x_i(1)) - \phi_0(x_i(0))\right)^2 +\lambda \beta^T \beta
\end{align}
"""

@tf.function #helps keep variables and not recalc everytime
#def J_obj_m(b_m, x = x0, LAM = .00001, phi0fixed = N_test_matrix, phi1fixed = N_train_matrix):
def J_obj_m(b_m, x, LAM, phi0fixed, phi1fixed):
  '''
  Args:
  (x_t1: estimated x @ time 1)
  x: initial trajectories at time = 0
  b_m: beta matrix (2*p-2,m)

  Returns:
  (len_phi_1) * (len_phi_0) non-dimensional vector of objective values for each
  comparison of test with training data
  '''

  # change x0 to a matrix so it can start in euler
  # x0 is a vector (256,1), x0 matrix is (256, 10 * 1000), for example
  # since we will be comparing 10 train to each vector test
  # phi1fixed is always (256,10): TRAN
  # phi0fixed is always (256,0:bat)
  """ For debuddiing
  print(f'phi1fixed shape: {phi1fixed.shape}')
  print(f'phi0fixed shape: {phi0fixed.shape}')
  print(f' len phi1fixed: should be 10 always: {len(phi1fixed[0])}')
  print(f' len phi0fixed[0]:  {len(phi0fixed[0])}')
  print(f'')
  """
  X_M = tf.tile(x, (len(x[0]), len(phi0fixed[0])* len(phi1fixed[0])))
  #print(f' X_M lenL {len(X_M[0])}')
  #print(X_M[0:5, 0:5])
  #print('X_M shape', X_M.shape)

  # Eulers tensfor flow function to obtain trajectories
  # this will be t @ 1 for all trajectories
  # b_m will change
  x_traj_m, _ = s_alpha_tf_m(X_M, b_m, alpha_m) #vectorized way of doing the euler
  #print(f' x_traj from eulers shape: {x_traj_m.shape}')
  #print(f' x_traj from euler: {x_traj_m[0:5, 0:5]}')

  # I want phi_one (256,10) in first 10, then another copy of (256,10), and so on
  # with length of how many columns are in phi0fixed provided
  # this is the only place I use phi1fixed (TRAIN) data to input in the
  # phi_1 interpolation function
  BIG_PHI_ONE = tf.tile(phi1fixed, [1, len(phi0fixed[0])])
  #BIG_PHI_ONE = tf.concat([tf.tile(phi1fixed[:, v:v+1], (1, len(phi0fixed[0]))) for v in range(phi1fixed.shape[1])], axis=1)
  """ For debugging
  print(f'BIG_PHI_ONE shape: {BIG_PHI_ONE.shape}')
  print(f'BIG_PHI_ONE: {BIG_PHI_ONE}')
  print(f'BIG_PHI_ONE : {BIG_PHI_ONE[:,0:10]}')
  print(f'BIG_PHI_ONE : {BIG_PHI_ONE[:,10:20]}')

  for w in range(len(BIG_PHI_ONE[0])):
    print(f' BIG_PHI_ONE norm: {np.linalg.norm(BIG_PHI_ONE[:,w])}')
  # now i have all my first x_traj for ALL my combo's
  #
  """
  #mat = []

  # compare each column of test to all train
  # then i could minimize values for sets of 10
  # since this is nearest neighbor
  #print(f' len phi1fixed: should be 10 always: {len(phi1fixed[0])}')
  #print(f' len phi0fixed[0]:  {len(phi0fixed[0])}')
  #print(f' len phi1fixed.shape[1] {phi1fixed.shape[1]}')
  """ OLD
  for j in range(len(phi0fixed[0])): # go through how many phi_0 (test) batch going in.
    print(f'j is the test fixed: {j}')
    for i in range(len(phi1fixed[0])): # for fixed j, compare that to each train (phi1)
      print(f'i is the train not fixed: {i}')
   #THIS only looks at the first 2 of x_traj, i want all of them.  This has nothing
   #to do with phi1, its the ESTMATED phi1 using x_traj and phi_1 inter func
      p = linear_inter(x_traj_m[:,j:j +1] # i want to fix each element in th 10
                                    , x
                                    , tf.transpose(phi1fixed[:,i:(i +1)]))

      mat.append(p)
  """
  """
  #********************* #THIS is original interpolation
  mat = []
  # Interpolation function, takes each x_traj_m column
  # and interpolates to each train (in BIG_PHI_ONE)
  # recall that I tiled phi1fixed to be the same size as x_traj_m
  # in order to compare to each column given in phi1fixed (this will vary)
  for i in range(len(X_M[0])):
    #print(i)
    p = linear_inter(x_traj_m[:,i:i+1]
                     , x
                     , tf.transpose(BIG_PHI_ONE[:,i:(i +1)]))
    mat.append(p)
    #print(len(p))

  # This is the contatenated BIG_PHI_est of all the interpolations
  BIG_PHI_est = tf.keras.layers.Concatenate(axis=1)(mat)
  BIG_PHI_est = tf.cast(BIG_PHI_est , dtype = tf.float64)
  #print('BIG_PHI_est', BIG_PHI_est.shape)
  #print(BIG_PHI_est[0:5, 2])
  #*****************************
  """

  #****************************** This is vectorized interpolation
  X_M_T = tf.transpose(X_M)
  #print('shape of X_M_T', X_M_T.shape)
  x_traj_m_T = tf.transpose(x_traj_m)
  #print('shape of x_traj_m_T', x_traj_m_T.shape)
  BIG_PHI_ONE_T = tf.transpose(BIG_PHI_ONE)
  #print('shape of BIG_PHI_ONE_T', BIG_PHI_ONE_T.shape)


  BIG_PHI_int = tf.transpose(tf.vectorized_map(int_vectorized
                                               , (x_traj_m_T
                                                  , X_M_T
                                                  , BIG_PHI_ONE_T)))

  BIG_PHI_int = tf.cast(BIG_PHI_int , dtype = tf.float64)
  #print('BIG_PHI_int shape: ', BIG_PHI_int.shape)

  #BIG_PHI_est_2 = tf.transpose(BIG_PHI_int)
  #print('BIG_PHI_est_2 shape', BIG_PHI_est_2.shape)
  #print(BIG_PHI_est_2[0:5, 2]


  #print('norm', np.linalg.norm(BIG_PHI_est - BIG_PHI_int)/np.linalg.norm(BIG_PHI_est))


  #*********************************

  """ For debugging:
  print(f' len(phi1fixed[0]) * len(phi0fixed[0]): {len(phi1fixed[0]) * len(phi0fixed[0])}')
  for k in range(len(phi1fixed[0]) * len(phi0fixed[0])):
    print(f' Big_phi_est norm: {np.linalg.norm(BIG_PHI_est[:,k])}')
  BIG_PHI_est = tf.cast(BIG_PHI_est , dtype = tf.float64)
  print(f'BIG_PHI_est shape: {BIG_PHI_est.shape}')
  """


  """
  Not correct.
  # bat is repeate, variable from above, could be 10, 2 or 5
  # len(phi1fixed) should always = 10
  # Big_Phi_est give me bat (i.e. 10, 5, 2) interposed with phi1filed[:,0:1],
  # then bat compared with phi1fixed[:, 1:2], until all 10 are compared with
  # so I need to have BIG_PHI_NOT be tiled in that same way.  The for bat of BIG_PHI_not
  # will be  phi1filed[:,0:1], then the next bat phi1fixed[:, 1:2],
  # so when we subtract from BIT_PHI_EST, we are subtracting the correct
  # comparison
  #print(f'first phi1fixed: {phi1fixed[0:5, 0:5]}')
  #'''
  #tile_repeats = len(phi1fixed[0]) * len(phi0fixed[0])
  #print(f'tile_repeast: {tile_repeats}')
  """


  # I want to compare BIG_PHI_NOT to each column of BIG_PHI_est (now BIG PHI_int)
  # should take phi0fixed[:, 0:bat] and make the first 10 columns the first column of
  # phi0fixed, and the next 10 should be the second column of phi0fixed and so on.
  BIG_PHI_NOT = tf.concat([tf.tile(phi0fixed[:, r:r+1], (1, len(phi1fixed[0]))) for r in range(phi0fixed.shape[1])], axis=1)
  """ For debugging
  print('These norms should be the same for each 10 columns')
  for s in range(len(BIG_PHI_NOT[0])):
    print(f' Big_phi_not norm: {np.linalg.norm(BIG_PHI_NOT[:,s])}')
  print(f'BIG_PHI_NOT : {BIG_PHI_NOT[:,0:10]}')
  print(f'BIG_PHI_NOT : {BIG_PHI_NOT[:,10:20]}')
  print(f'BIG+PHI_NOT: {BIG_PHI_NOT}')
  print(f'BIIG_PHI_NOT shape: {BIG_PHI_NOT.shape}')
  print(f' BIG_PHI_NOT len {len(BIG_PHI_NOT[0])}')
  """
  #BIG_PHI_NOT = tf.concat([tf.tile(phi0fixed[:, r:r+1], (1, len(phi1fixed[0]))) for r in range(phi0fixed.shape[1])], axis=1)



  #'''
  #print(f'bat: {bat}')
  #print(f'len(phi1fixed) (train) should be ten: {len(phi1fixed)}')
  #print(f'BIG_PHI_NOT shape: {BIG_PHI_NOT.shape}')
  #print(f'The first bat should be the same{BIG_PHI_NOT[0:5,0:2*bat]}')

  """
  not tiling correctly
  # tile the phi0 data to compare to how many phi1 test data were batched in.  i.e. 10 at a time,
  # so compare each test to each train. BIG_PHI_est comes out fixing each test
  # and comparing to each train. So here big_phi_not is tiled for the 256, 10 matrix batch times
  BIG_PHI_NOT = tf.tile(phi0fixed, [1,len(phi1fixed[0])]) #10 for the whole phione matrix
  #BIG_PHI_NOT = tf.tile(phi0fixed, [1,10])

  BIG_PHI_NOT = tf.cast(BIG_PHI_NOT , dtype = tf.float64)
  for l in range(tile_repeats):
    print(f' Big_phi_not norm: {np.linalg.norm(BIG_PHI_NOT[:,l])}')
  print(f'BIG_PHI_NOT : {BIG_PHI_NOT}')
  print(f'BIIG_PHI_NOT shape: {BIG_PHI_NOT.shape}')



  for t in range(tile_repeats):
    print(np.linalg.norm(BIG_PHI_NOT[:,t] - BIG_PHI_NOT_test[:,t])/np.linalg.norm(BIG_PHI_NOT[:,t]))
  """


  # taking the difference and squaring in regards to comparing the test to each train repsectively
  # in order of the train data (256, 10)
  #************************* original
  #phi_diff = tf.math.squared_difference(BIG_PHI_est,BIG_PHI_NOT)
  #************************
  #**************************
  phi_diff = tf.math.squared_difference(BIG_PHI_int,BIG_PHI_NOT)
  #****************


  #print(f'phi_diff shape: {phi_diff.shape}')
  # this give a tensor of (10 * len(phinot[0]))
  summ = tf.reduce_sum(phi_diff, axis = 0)
  #print(f'summ of phi_diff: {summ.shape}')



  n = (len(X_M)) #* len(X_M[0]))
  n = tf.cast(n, dtype=tf.float64)
  beta_term = tf.reduce_sum((b_m * b_m), axis = 0)
  #print("**********************************************")
  #print("**********************************************")

  #print(f'beta_term shape: {beta_term.shape}')
  #print(f' summ shape: {summ.shape}')
  #print(f'summ: {summ}')
  #print(f'beta_term: {beta_term}')
  #obj = ((1/(n)) * summ) + (LAM * beta_term)
  obj = (summ / n) + (LAM * beta_term)
  #print(obj)
  #need to add lam * (sum) (B haddamard B)

  #return obj
  # RETURN: calculate the vectorized objective and beta gradient separately
  return obj, (summ / n), (LAM * beta_term), x_traj_m

#with tf.GradientTape() as tape:
#  tape.watch(beta_M)
#  obj, _, _ = J_obj_m(b_m=beta_M3
#      , x = x0
#      , LAM = .0001
#      , phi0fixed = N_test_matrix[:,0:3]
#      , phi1fixed = N_train_matrix)

#gradient = tape.gradient(obj, beta_M3)

#gradient.shape

#gradient[0:7, 0:6]

#for i in range(30):
#  print(f' norm grad: {np.linalg.norm(gradient[:,i])}')
#  #print(f' norm beta: {np.linalg.norm(beta_M[:,i])}')

"""# Gradient Descent

## initialize beta
"""

# prompt: generate a vector close to zero but not zero using random selectiion mean, sd
np.random.seed(3)
# Set mean and standard deviation close to zero
mean = 0.001
sd = 0.001

# Generate a vector of random numbers with the specified mean and standard deviation
#b_ini = np.random.normal(mean, sd, ((num_features*2)-2,1))  # Replace 10 with the desired vector length
#b_ini = tf.zeros(shape = (len(sx0)-2,1), dtype = tf.float64)
b_ini = tf.random.normal(shape=((2 * num_features) - 2, 1),
                         mean=mean,
                         stddev=sd,
                         dtype=tf.float64)

b_ini = tf.constant(b_ini, dtype=tf.float64)

#print(b_ini
#print(b_ini.shape)

"""# Gradient Descent with BIG obj function

## Train and Test datasets for run
"""

try:
   test_label = input("Input test_label (phinot), i.e. Beef1, N: ")
   tensor_test = test_label + '_test_matrix'
   # check if variable exists in the global space and is a tf tensor
   if tensor_test in globals() and isinstance(globals()[tensor_test], tf.Tensor):
     phinot_m = globals()[tensor_test]
     print(f'TEST Matrix: {tensor_test} with shape: {phinot_m.shape}.')
   else:
     # throw exception
     raise KeyError(f"Error: {tensor_test} is not a valid tensor name or not a tensor. "
               "Enter valid test_label: Beef, N, A, V, R, L, E, J, g (j).")

   #print(f'TEST Matrix: {tensor_test} with shape: {phinot_m.shape}.')
   #train_label = input("Input train_label (phione), i.e. N, A, V, R, L, E, J, g (j): ")
   train_label = input("Type Beef, all R, V, L, g (jj), E, N, J , A: ")
   tensor_train = train_label + '_train_matrix'
   # check if variable exists in the global space and is a tf tensor
   if tensor_train in globals() and isinstance(globals()[tensor_train], tf.Tensor): #globals() is a dictionary
     phione = globals()[tensor_train]
     print(f'TRAIN Matrix: {tensor_train} with shape: {phione.shape}.')
   else:
     # throw exception
     raise KeyError(f"Error: {tensor_train} is not a valid tensor name or not a tensor."
            "Enter valid train_label: N, A, V, R, L, E, J, g (j).")


   print('!!!In order to conduct all experiments, len(TEST_matrix[0]) / chunck must be an integer!!!')
   block = input("Input the block size, i.e. 1, 2, 6, 50: ")
   try:
     block = int(block)
     check = len(phinot_m[0]) / block

     if check.is_integer():
       print(f'Block of TEST matrix will be: {block}')
     else:
       raise KeyError( f'{len(phinot_m[0]) / block} is not an integer. '
           'Choose chunck where len(TEST_matrix[0]) / chunck is an integer.')
   except ValueError:
     print("Invalid input. Please enter an block value that works.")
except Exception as oops:
    print(oops)
    sys.exit(1)

#print(block)
#print(phinot_m.shape)
#print(phione.shape)

#bat = 50 # , 50
#phinot_m = N_test_matrix #J, E , N,
phinot = phinot_m[:,0:block] #The first 10 compared with all 10 of train = 50 comparisons
                         # this is How i'm or
#phione = N_train_matrix #N, A, V, R, L, E, J, j
#phione = N_train_matrix

b_ini_m = tf.tile(b_ini, [len(b_ini[0]),len(phinot[0]) * len(phione[0])])
b_ini_m = tf.cast(b_ini_m, dtype = tf.float64)
print(b_ini_m.shape)

x_trajectories =[]
obj_values = []
squared_dist = []
beta_grad_dist = []
beta_list = []
start_time = time.time()
lr = 1/(2**9)
print(f'learning rate {lr}')
#BETA = tf.Variable(b_ini_m, trainable = True)
#BETA = tf.Variable(b_ini_m)
#BETA = b_ini_m
iter = 10
for chunck in range(int(len(phinot_m[0])/len(phinot[0]))):#how many chuncks  
  BETA = tf.Variable(b_ini_m)
  #BETA.assign(tf.Variable(b_ini_m, trainable = True))
  #print(f"Epoch {epoch}, BETA reset to initial value: {BETA.numpy()[0:5,0:5]}")
  print(f'Chunck: {chunck} of {int(len(phinot_m[0])/len(phinot[0]))}')
  phinot = phinot_m[:,chunck*len(phinot[0]):(chunck+1)*len(phinot[0])]
  #print(phinot.shape)
  #print(epoch*len(phinot[0]))
  #print((epoch+1)*len(phinot[0]))

  #iter = 6 #

  #beta_list.append(BETA)

  #print(f'learning rate {lr}')
  LAMBDA = 1e2
  #phinot= N_test_matrix
  #phione = N_train_matrix


  #optimizer = tf.keras.optimizers.SGD(learning_rate = lr)
  optimizer = tf.keras.optimizers.Adam(learning_rate = lr)




  start_time_iter = time.time()

  for iteration in range(iter): #iter not batch

    with tf.GradientTape() as tape:
      tape.watch(BETA)
      #obj = J_obj_m(b_m=BETA
      obj, _, _, _ = J_obj_m(b_m=BETA
          , x = x0
          , LAM = LAMBDA
          , phi0fixed = phinot
          , phi1fixed = phione)

    grad = tape.gradient(obj, BETA)
    """ #For debugging:
    #if grad is None:N
    #  print("grad are none")
    #else:
    #  print("grad computed")
    """
    """ #For debugging:
    #watched_vars = tape.watched_variables()
    #print(f"Watched variable: {watched_vars:}")
    #if BETA in watched_vars:
    #  print("BETA is being watched")
    #else:
    #  print("BETA is not being watched."  )
    """
    #assign_sub updates BETA in place without creating a new tensor.
    #Subtracts value from the current value of the variable.
    #BETA.assign_sub(lr*grad)
    #BETA = BETA - lr * grad
    optimizer.apply_gradients([(grad, BETA)])

    """ #For debugging:
    #print(f'grad: {grad.shape}')
    #print(f'grad: {grad[0:5,0:3]}')
    #print(f' beta update: {BETA[0:5,0:5]}')
    """

    obj_new, sq_dist, b_grad, x_traj = J_obj_m(BETA, x = x0, LAM = LAMBDA, phi0fixed = phinot, phi1fixed = phione)
    #obj_new, _, _ = J_obj_m(BETA, x = x0, LAM = LAMBDA, phi0fixed = phinot, phi1fixed = phione)
    #_, sq_dist, _ = J_obj_m(BETA, x = x0, LAM = LAMBDA, phi0fixed = phinot, phi1fixed = phione)
    #sq_dist = J_obj_sqdist(BETA, x = x0, phi0fixed = phinot, phi1fixed = phione)
    #b_grad = J_obj_betagrad(BETA, LAM = LAMBDA)
    #print(f'batch: {batch}, new objective: {obj_new[0]}')
    #print('**************************************************************')
    #print('**************************************************************')
    print(f'Iteration: {iteration}, new objective: {obj_new[0:6]}')
    #obj_values.append(np.squeeze(obj_new)) #obj is a tensor, remove it by squeeze
    #beta_list.append(BETA)
    #print(f"Gradient at batch {batch}: {grad.numpy()[0:5,0:5]}")
    #print(f"Epoch {epoch}, BETA value after 6 iters: {BETA.numpy()[0:5,0:5]}")
  #print(obj_new)
  obj_values.append(obj_new)
  squared_dist.append(sq_dist)
  beta_grad_dist.append(b_grad)
  x_trajectories.append(x_traj)
  #print(obj_new)

  end_time_iter = time.time()
  elapsed_time_iter = end_time_iter - start_time_iter
  print(f"Elapsed time per iter: {elapsed_time_iter} sec");


end_time = time.time()
elapsed_time = round((end_time - start_time) / 60, 2)
print(f"Elapsed time total: {elapsed_time} min");
#print(obj_values)
obj_values = tf.concat(obj_values, axis=0)
squared_dist = tf.concat(squared_dist, axis=0)
beta_grad_dist = tf.concat(beta_grad_dist, axis=0)
x_trajectories = tf.concat(x_trajectories, axis=1) 

"""# copy files"""

base_path = "/home/vhaney/EKG/current_filtered_data/"

# file for objective function
file_obj = f"{test_label}{train_label}_dist_obj.csv"
file_path_1 = os.path.join(base_path, file_obj)

df_obj = f"df_{test_label}{train_label}_dist_obj"
globals()[df_obj] = pd.DataFrame(obj_values)
globals()[df_obj].to_csv(file_path_1, index=False, header=False)
#print(df_obj)
print(f'{test_label}{train_label}_dist_obj saved to {file_path_1}')

# file for squared distance
file_sd = f"{test_label}{train_label}_dist_sd.csv"
file_path_2 = os.path.join(base_path, file_sd)

df_sd = f"df_{test_label}{train_label}_dist_sd"
globals()[df_sd] = pd.DataFrame(squared_dist)
globals()[df_sd].to_csv(file_path_2, index=False, header=False)
print(f'{test_label}{train_label}_dist_sd saved to {file_path_2}')

# file for beta norm
file_b = f"{test_label}{train_label}_dist_b.csv"
file_path_3 = os.path.join(base_path, file_b)

df_b = f"df_{test_label}{train_label}_dist_b"
globals()[df_b] = pd.DataFrame(beta_grad_dist)
globals()[df_b].to_csv(file_path_3, index=False, header = False)
print(f'{test_label}{train_label}_dist_b saved to {file_path_3}')

# file for elapsed time
file_elapsed_time = f"{test_label}{train_label}_elapsed_time.txt"
file_path_4 = os.path.join(base_path, file_elapsed_time)
with open(file_path_4, 'w') as f:
    f.write(f"{elapsed_time}")
print(f"Elapsed time saved in {file_path_4}")

# file for x trajectories
file_traj = f"{test_label}{train_label}_x_traj.csv"
file_path_5 = os.path.join(base_path, file_traj)

df_traj = f"df_{test_label}{train_label}_x_traj"
globals()[df_traj] = pd.DataFrame(x_trajectories)
globals()[df_traj].to_csv(file_path_5, index=False, header=False)
print(f'{test_label}{train_label}_x_traj saved to {file_path_5}')



""" #Origial way of saving to file where data is 
df_obj = f"df_{test_label}{train_label}_dist_obj"
globals()[df_obj] = pd.DataFrame(obj_values)
globals()[df_obj].to_csv(f'{test_label}{train_label}_dist_obj_4.csv', index=False, header=False)
#print(df_obj)
print(f'{test_label}{train_label}_dist_obj_4')



df_sd = f"df_{test_label}{train_label}_dist_sd"

globals()[df_sd] = pd.DataFrame(squared_dist)
globals()[df_sd].to_csv(f'{test_label}{train_label}_dist_sd_4.csv', index=False, header=False)
print(f'{test_label}{train_label}_dist_sd_4')

df_b = f"df_{test_label}{train_label}_dist_b"
globals()[df_b] = pd.DataFrame(beta_grad_dist)
globals()[df_b].to_csv(f'{test_label}{train_label}_dist_b_4.csv', index=False, header = False)
print(f'{test_label}{train_label}_dist_b_4')

# Save elapsed time separately
elapsed_time_filename = f'{test_label}{train_label}_elapsed_time.txt'
with open(elapsed_time_filename, 'w') as f:
    f.write(f"{elapsed_time}")

print(f"Elapsed time saved in {elapsed_time_filename}")

df_traj = f"df_{test_label}{train_label}_x_traj"
globals()[df_traj] = pd.DataFrame(x_trajectories)
globals()[df_traj].to_csv(f'{test_label}{train_label}_x_traj_4.csv', index=False, header=False)
print(f'{test_label}{train_label}_x_traj_4')
"""




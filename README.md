This is an algorithm that is a continuous for of dynamic time warping.

In order to use, one can create a shell/text file to conduct jobs on GPUs.  If you would like to conducte one
job, follow these instructions:

1. Requires allocation to a GPU.
2. Activate virtual environemnt for using tensorflow. Ensure tensorflow is pip installed.  There are a list of imports that are required that are in the notebook.
3. One needs to change base_path for the loading data function AND for saving files.
4. There is an order to inputing data:
  Depending on the data:
    1. Input length of initial trajectory. (Signal length)
    2. Testing data.  (Beef1_test, N_test, label of testing data).
    3. Training data. (Beet_train, all_train, all training data). (EKG train data is called all_train_matrix, for now).
    4. 'n' is to skip to next step.
    5. Test label for file naming convention (Beef1, N).
    6. Train label for file naming convention. (Beef, all).
    7. The number of how many testing data to do training. If testing data is larger than 10, usually the train data needs to be cut into chunks.

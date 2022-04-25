# 1. Background
利用GCN对fMRI的动态影像的数据进行编码，接上Readout Layer实现回归任务，Loss Function 采用Huber Loss.

# 2. Install & Run
1. Make sure that your virtual env has the following packages:
   1. PyTorch (GPU Ver recommended)
   2. Numpy
   3. Pandas
   4. DGL (0.7.2 ver)
      ```shell
      pip install dgl-cu113==0.7.2 -f https://data.dgl.ai/wheels/repo.html
      ```
   5. Tensorboard
   6. Scikit-Learn
   
2. download the project from the repository

3. preprocess data into one DGL graph for further training and testing

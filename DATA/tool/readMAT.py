r'''
matlab 存储的mat类型的数据有两种格式v7和v7.3 不同的格式用不同的python库读取

-v7格式
scipy.io库

    import scipy.io as sio 
    matfn = '/home/weiliu/workspace/python/matlab/mat4py.mat'
    data = sio.loadmat(matfn)

-v7.3格式 
h5py库

    import h5py
    path='C:/data.mat'                    #需要读取的mat文件路径
    feature=h5py.File(path)               #读取mat文件
    data = feature['feature_data'][:]     #读取mat文件中所有数据存储到array中

'''
import scipy.io as scio
import numpy as np
def readFromMat(file_name):
    mat_name = file_name
    dict_data = scio.loadmat(mat_name)
    #print(dict_data)
    return dict_data['X'], dict_data['y']

file = '../data/mnist'
file_name = file + '.mat'
X, Y = readFromMat(file_name)
Y = Y.astype(float)
Y[Y==0] = -1
print(X.shape)
print(Y.shape)
np.savez(file, X=X, Y=Y)
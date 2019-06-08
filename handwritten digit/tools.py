from matplotlib import pyplot
import numpy as np


# 手写数字展示函数，可略过
def displaydata(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data stored in X in a nice grid.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = pyplot.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')
    pyplot.show()


# sigmoid函数
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# 神经网络预测模型，是一个三层神经网络（两层具有权重节点），各层节点数分别为  400（20*20像素）、25、10
def getresult(theta1, theta2, data):  # for 400*25 25*10 two layers' neural network
    if data.ndim == 1:
        data = data[None]

    data_size = data.shape[0]
    num_labels = 10
    result = np.zeros(data_size)

    data = np.concatenate([np.ones((data_size, 1)), data], axis=1)
    out1 = sigmoid(np.dot(data, theta1.T))
    out1 = np.concatenate([np.ones((data_size, 1)), out1], axis=1)
    out2 = sigmoid(np.dot(out1, theta2.T))
    result = np.argmax(out2, axis=1)

    return result

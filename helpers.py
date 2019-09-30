import os
import pickle
import matplotlib.pyplot as plt

def store(data, data_filename, verbose=False):
    """
    Pickler method

    Parameters
    ----------
    data: Any serializable object
    data_filename: String
    verbose: boolean (deafult=False)
    """
    with open(data_filename, 'wb') as data_file:
        pickle.dump(data, data_file)
    if verbose:
        print('Data stored in {}'.format(str(data_filename)))

def restore(data_filename, verbose=False):
    """
    Unpickler method
    
    Paramters
    ---------
    data_filename: String

    Returns
    -------
    data object
    """
    data = None
    with open(data_filename, 'rb') as data_file:
        data = pickle.load(data_file)
    return data

def custom_scatter_plot(x, y):
    """
    Parameters
    ----------
    x,y: iterable of floats of same length
    """
    plt.plot(x,y,color='k',marker='.',linestyle='none',alpha=0.1)
    plt.grid()
    plt.xlabel('Departure')
    plt.ylabel('Arrival')
    plt.show()
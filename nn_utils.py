import pandas as pd
import numpy as np
import scipy.io as sio
from pylab import find, newaxis

from sklearn.model_selection import train_test_split

def seq_to_array(seq, max_length=None):
    """
    Given a string, this returns a 2d np array: axis 1 = sequence,
    axis 2 = character (A, T, C, G)
    """
    new_array = np.zeros((len(seq), 4))
    if max_length is not None:
        new_array = np.zeros((max_length, 4))
    for i, s in enumerate(seq):
        if s=='A' or s=='a':
            new_array[i][0] = 1
        elif s=='T' or s=='t':
            new_array[i][1] = 1
        elif s=='C' or s=='c':
            new_array[i][2] = 1
        elif s=='G' or s=='g':
            new_array[i][3] = 1
    return new_array

def array_to_seq(array):
    """
    Given an np array as created by seq_to_array, this reverses the operation
    and returns a string sequence.
    """
    # TODO

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def load_a5_data(flank=0):
    """
    Loads the A5 data.

    The data consists of X_train, X_test, Y_train, Y_test.

    X: the first 25 units of each row are the first degenerate region,
    the second 25 units are the second degenerate region.

    Y: sparse array that contains the splice probability at each site in
    the sequence (around 300?)
    """
    data = sio.loadmat('../data/Reads.mat')
    # A5SS_data: numpy array containing a row for each sequence and a column for
    # each base pair position, where each value contains the proportion of
    # splicing events that occur at that particulaar position.
    A5SS_data = data['A5SS']
    A5SS_reads = np.array(A5SS_data.sum(1)).flatten()
    A5SS_data = np.array(A5SS_data.todense())
    # Get minigenes with reads
    A5SS_nn = find(A5SS_data.sum(axis=1))
    A5SS_reads = A5SS_reads[A5SS_nn]
    A5SS_data = A5SS_data[A5SS_nn]
    # A5SS_data is now normalized so that each row sums to 1
    A5SS_data = A5SS_data/A5SS_data.sum(axis=1)[:,newaxis]
    # A5SS_seqs: all sequences
    A5SS_seqs = pd.read_csv('../data/A5SS_Seqs.csv',index_col=0).Seq[A5SS_nn]
    # A5SS_r1: first degenerate region
    A5SS_r1 = A5SS_seqs.str.slice(7-flank, 32+flank)
    # A5SS_r2: second degenerate region
    A5SS_r2 = A5SS_seqs.str.slice(50-flank, 75+flank)
    A5SS_r = A5SS_r1 + A5SS_r2
    A5SS_r_array = A5SS_r.map(seq_to_array)
    # set random_state to 0 to have repeatable trials
    X_train, X_test, Y_train, Y_test = \
                train_test_split(A5SS_r_array, \
                A5SS_data, test_size = 0.2, random_state=0)
    return (X_train, X_test, Y_train, Y_test)

def load_a3_data(flank=0):
    """
    Loads the A3 data...
    """
    data = sio.loadmat('../data/Reads.mat')
    # A5SS_data: numpy array containing a row for each sequence and a column for
    # each base pair position, where each value contains the proportion of
    # splicing events that occur at that particulaar position.
    A3SS_data = data['A3SS']
    A3SS_reads = np.array(A3SS_data.sum(1)).flatten()
    #A3SS_data = np.array(A3SS_data.todense())
    # Get minigenes with at least 8 reads
    A3SS_nn = find(A3SS_data.sum(axis=1)>=8)
    A3SS_reads = A3SS_reads[A3SS_nn]
    A3SS_data = A3SS_data[A3SS_nn]
    A3SS_data = np.array(A3SS_data.todense(), dtype=float)
    # A5SS_data is now normalized so that each row sums to 1
    A3SS_data = A3SS_data/A3SS_data.sum(axis=1)[:,newaxis]
    # A5SS_seqs: all sequences
    A3SS_seqs = pd.read_csv('../data/A3SS_Seqs.csv',index_col=0).Seq[A3SS_nn]
    # A5SS_r1: first degenerate region
    A3SS_r1 = A3SS_seqs.str.slice(0, 25)
    # A5SS_r2: second degenerate region
    A3SS_r2 = A3SS_seqs.str.slice(50-flank, 75+flank)
    A3SS_r = A3SS_r1 + A3SS_r2
    A3SS_r_array = A3SS_r.map(seq_to_array)
    # set random_state to 0 to have repeatable trials
    X_train, X_test, Y_train, Y_test = \
                train_test_split(A3SS_r_array, \
                A3SS_data, test_size = 0.2, random_state=0)
    return (X_train, X_test, Y_train, Y_test)

def load_cell_type_data(flank=0, cell_type='HELA'):
    """
    Loads the A5 data for multiple cell types

    The different cell types: HEPG2, MCF7, CHO, HELA, LNCAP, HEK
    , which is passed as a string

    The data consists of X_train, X_test, Y_train, Y_test.
    """
    data = sio.loadmat('../data/Alt_5SS_Usage_All_Cells.mat')
    A5SS_data = data[cell_type]
    A5SS_reads = np.array(A5SS_data.sum(1)).flatten()
    A5SS_data = np.array(A5SS_data.todense())
    # Get minigenes with reads
    A5SS_nn = find(A5SS_data.sum(axis=1))
    A5SS_reads = A5SS_reads[A5SS_nn]
    A5SS_data = A5SS_data[A5SS_nn]
    # A5SS_data is now normalized so that each row sums to 1
    A5SS_data = A5SS_data/A5SS_data.sum(axis=1)[:,newaxis]
    # A5SS_seqs: all sequences
    A5SS_seqs = pd.read_csv('../data/A5SS_Seqs.csv',index_col=0).Seq[A5SS_nn]
    # A5SS_r1: first degenerate region
    A5SS_r1 = A5SS_seqs.str.slice(7-flank, 32+flank)
    # A5SS_r2: second degenerate region
    A5SS_r2 = A5SS_seqs.str.slice(50-flank, 75+flank)
    A5SS_r = A5SS_r1 + A5SS_r2
    A5SS_r_array = A5SS_r.map(seq_to_array)
    # set random_state to 0 to have repeatable trials
    X_train, X_test, Y_train, Y_test = \
                train_test_split(A5SS_r_array, \
                A5SS_data, test_size = 0.2, random_state=0)
    return (X_train, X_test, Y_train, Y_test)



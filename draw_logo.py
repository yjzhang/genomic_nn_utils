# code to draw a sequence logo...

import matplotlib.pyplot as plt
from matplotlib import transforms
import matplotlib.patheffects

import numpy as np
from keras.models import Model
from keras import backend as K

def entropy(row):
    return -sum(x*np.log2(x+1e-10) for x in row)

def get_max_filters(model, X_test, layer=None):
    """
    Returns the maximum position of each convolutional filter in the first
    layer for each input.
    """
    f0 = K.function([model.layers[0].input], [model.layers[0].output])
    if layer is None:
        for l in model.layers:
            if 'conv1d' in l.name:
                f0 = K.function([l.input], [l.output])
                break
    else:
        f0 = K.function([layer.input], [layer.output])
    # f1 = K.function([model.layers[1].input], [model.layers[1].output])
    # conv_out is a 3d array - dim is (input) x (position) x (filter)
    # layer_out is a 2d array - dim is (# inputs) x (# filters)
    conv_out = f0([X_test])[0]
    # just get the maximum output for each seq
    layer_out = np.max(conv_out, 1)
    n_inputs, n_filters = layer_out.shape
    subseqs = []
    for i in range(n_filters):
        conv_out_i = conv_out[:,:,i]
        max_pos = np.argmax(conv_out_i, 1)
        subseqs.append(max_pos)
    return subseqs, layer_out
    # positive regulators: positive coefficient in dense layer

def get_avg_filters(model, X_test, layer=None):
    """
    model: a Keras nn model
    X_test: a numpy array of encoded sequences

    Returns the average of the maximum activation subsequence for every 1st
    layer filter.
    """
    x, layer_outs = get_max_filters(model, X_test, layer)
    avg_filters = []
    n_inputs, n_filters = layer_outs.shape
    # i is the filter, k is the input
    for i in range(n_filters):
        nonzeros = np.nonzero(layer_outs[:,i]>0)[0]
        avg_filter1 = sum(X_test[k,x[i][k]:(x[i][k]+6),:] for k in nonzeros)
        eps=1e-8
        if len(nonzeros) == 0:
            avg_filters.append(np.zeros((6,4)) + 0.25)
        else:
            avg_filters.append(avg_filter1/(len(nonzeros)+eps))
    # TODO: sometimes, the results give zero... in that case we probably
    # want completely flat motifs.
    return avg_filters

def get_influence_functional(model, X_test, depth=0, layer_id=0):
    """
    Gets the influence of every filter of a functional model.
    """
    inputs = model.input_layers
    layer_nodes = model.nodes_by_depth[depth]
    m1 = Model(inputs=inputs, outputs=layer_nodes)
    m2 = Model(inputs=layer_nodes, outputs=model.output)

def get_influence(model, X_test, set_zero=False, layer_id=0):
    """
    Gets the influence of each filter.

    Influence is defined as the L2 norm of the difference between the output
    across all inputs when the given filter is nullified (output set to its
    mean? or to zero?)

    This can be done manually...
    """
    # TODO: what if there are multiple convolutional layers???
    f0 = K.function([model.layers[layer_id].input], [model.layers[layer_id].output])
    f1 = K.function([model.layers[layer_id+1].input], [model.layers[-1].output])
    conv_out = f0([X_test])[0]
    final_out = f1([conv_out])[0]
    # conv_out is a 3d array - dim is (input) x (position) x (filter)
    n_inputs, n_positions, n_filters = conv_out.shape
    influences = []
    for i in range(n_filters):
        # TODO
        # try both zeroing the filter and setting the value to the mean.
        filter_mean = np.mean(conv_out[:,:,i])
        conv_out_old = conv_out[:,:,i].copy()
        if set_zero:
            conv_out[:,:,i] = 0.0
        else:
            conv_out[:,:,i] = filter_mean
        mean_out = f1([conv_out])[0]
        conv_out[:,:,i] = conv_out_old
        influences.append(np.sqrt(((mean_out - final_out)**2).sum()))
    return influences

# TODO: find splice enhancer vs splice repressor using influence

# ATCG
# data format: same as stuff produced by get_avg_filters
def calc_logo(data):
    """
    Data format: table of relative frequencies - first dimension is length,
    second dimension is character. Each entry is the frequency of that
    character.

    The output is the frequency * entropy for each position.
    """
    hmax = 2.0
    num_chars = data.shape[1]
    length = data.shape[0]
    logo = np.array(data)
    for i in range(length):
        ent = hmax - entropy(data[i,:])
        logo[i,:] *= ent
    return logo

def draw_logo(data, fixed_scale=True, ax=None):
    logo_data = calc_logo(data)
    if ax is None:
        ax = plt.gca()
    fig = ax.figure
    colors = ['r', 'g', 'b', 'y']
    bars = []
    width = 0.8
    ind = np.arange(logo_data.shape[0])
    cumulative_heights = np.zeros(logo_data[:,0].shape)
    for i, c in enumerate(colors):
        nuc = logo_data[:,i]
        if i == 0:
            p = ax.bar(ind, nuc, width, color=c)
        else:
            p = ax.bar(ind, nuc, width, color=c, bottom = cumulative_heights)
        cumulative_heights += nuc
        bars.append(p)
    if fixed_scale:
        ax.set_ylim(0, 2)
    ax.legend((b[0] for b in bars), ['A', 'T', 'C', 'G'])
    ax.set_ylabel('Information (bits)')
    return bars

# based on: https://github.com/saketkc/notebooks/blob/master/python/Sequence Logo Python -- Any font.ipynb
class Scale(matplotlib.patheffects.RendererBase):
    def __init__(self, sx, sy=None):
        self._sx = sx
        self._sy = sy

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        affine = affine.identity().scale(self._sx, self._sy)+affine
        renderer.draw_path(gc, tpath, affine, rgbFace)

def draw_logo_2(data, fixed_scale=True, ax=None):
    """
    Draws a logo with letters instead of bars.
    """
    logo_data = calc_logo(data)
    if ax is None:
        ax = plt.gca()
    fig = ax.figure
    ax.set_xlim(0, logo_data.shape[0]+0.5)
    ax.set_ylim(0, 2)
    ax.set_ylabel('Information (bits)')
    #fig, ax = plt.subplots(figsize=(logo_data.shape[0], 2.5))
    colors = ['r', 'g', 'b', 'y']
    bases = ['A', 'T', 'C', 'G']
    trans_offset = transforms.offset_copy(ax.transData,
            fig=fig,
            x=1,
            y=0,
            units='dots')
    scale = 1.5
    for i in range(logo_data.shape[0]):
        scores = logo_data[i,:]
        for b, c, s in zip(bases, colors, scores):
            txt = ax.text(i+1, 0, b, transform=trans_offset, fontsize=100, ha='center', color=c)
            txt.set_path_effects([Scale(0.8, s*scale)])
            yshift = s*scale
            trans_offset = transforms.offset_copy(trans_offset,
                    fig=fig,
                    y=yshift)
        trans_offset = transforms.offset_copy(ax.transData,
                fig=fig,
                y=0,
                units='points')
    return fig, ax

def information_content(motif):
    """
    Returns the information content of a given filter/motif.
    Just the maximum possible entropy minus the actual entropy.
    """
    l = len(motif)
    return l*2.0 - sum(entropy(motif[j,:] for j in range(l)))

def all_info_content(motifs):
    info = []
    for i in range(motifs.shape[0]):
        info.append(information_content(motifs[i,:,:]))
    return info

def meme_output(motif):
    """
    Returns a string that represents the motif in MEME format.
    """
    # permutation = A C G T
    permutation = [0,2,3,1]
    motif1 = motif[:,permutation]
    s1 = ''
    for i in range(motif1.shape[0]):
        s1 += '\t'.join(map(str, motif1[i,:]))
        s1 += '\n'
    return s1

def meme_output_file(motifs, filename):
    """
    Writes the motifs in MEME format to the given filename.
    """
    with open(filename, 'w') as f:
        f.write("""MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.303 C 0.183 G 0.209 T 0.306

""")
        # permutation = A C G T
        permutation = [0,2,3,1]
        for j in range(len(motifs)):
            motif1 = motifs[j][:,permutation]
            f.write('MOTIF {0}\n'.format(j))
            f.write('letter-probability matrix: alength= 4 w= 6 nsites= 6 E= 1e-010\n')
            for i in range(motif1.shape[0]):
                f.write(' \t'.join(map(str, motif1[i,:])))
                f.write('\n')
            f.write('\n')

def kld(dist1, dist2):
    """
    KL Divergence for two discrete prob distributions (1-d lists or vectors)
    """
    return sum(x*np.log2(x/y) for x,y in zip(dist1, dist2))

def symmetric_kld(motif1, motif2):
    """
    Symmetric K-L divergence between motif1 and motif2.

    does this by calculating the sum of the KL divergences between each of the positions.
    """
    positions, chars = motif1.shape
    return sum(kld(motif1[i,:], motif2[i,:]) + kld(motif2[i,:], motif1[i,:]) for i in range(positions))

def motif_set_comparison(motifs1, motifs2):
    """
    computes distance matrix between two sets of motifs
    """
    distances = np.zeros((motifs1.shape[0], motifs2.shape[0]))
    eps = 1e-8
    motifs1 += eps
    motifs2 += eps
    for i in range(motifs1.shape[0]):
        for j in range(motifs2.shape[0]):
            distances[i,j] = symmetric_kld(motifs1[i,:,:], motifs2[j,:,:])
    return distances

# TODO: matching motifs... given a distance matrix created by motif_set_comparision, find the motifs that are most similar in the two sets.
def motif_set_matches(motifs1, motifs2):
    """
    Returns a list of pairs of indices, indicating the matches between motifs.

    Each motif in motifs1 is matched to one motif in motifs2, but there may be multiple motifs in motifs1 matched to the same motif in motifs2.
    """
    distances = motif_set_comparison(motifs1, motifs2)
    matches = []
    for i in range(motifs1.shape[0]):
        min_dist = distances[i,:].argmin()
        matches.append((i, min_dist, distances[i, min_dist]))
    return matches

# TODO: better heuristics for matching???

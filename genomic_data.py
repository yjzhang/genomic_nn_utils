# tools for working with genomic data
import sqlite3

from keras.preprocessing import sequence
from keras import backend as K
from keras.models import Model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import nn_utils





expit = lambda x: 1./(1.+np.exp(-x))
logit = lambda x: np.log(x)-np.log(1-x)

def get_delta_psis(model, wt_seqs, mut_seqs, wt_psis):
    """
    Returns a list of dPSIs for a list of sequences.
    """
    # get_output takes the output of the dense layers before
    # it passes through the activation.
    model2 = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    # TODO: do not take into account the bias term
    outputs = []
    bias = model.layers[-2].get_weights()[1][0]
    for w, m, psi in zip(wt_seqs, mut_seqs, wt_psis):
        score_wt = model2.predict([w])[0][0] - bias
        score_mut = model2.predict([m])[0][0] - bias
        mut_psi = expit(score_mut - score_wt + logit(psi))
        outputs.append((mut_psi - psi))
    return np.array(outputs)

def get_score(model1, input_data):
    get_3rd_layer_output_1 = K.function([model1.layers[0].input],
        [model1.layers[-2].get_output(train=False)])
    return get_3rd_layer_output_1([input_data])[0]
    #return (get_3rd_layer_output_2([input_data])[0] + get_3rd_layer_output_1([input_data])[0])/2

def get_delta_psi(model1, wt_seq, mut_seq, wt_psi):
    score_wt = get_score(model1, wt_seq)
    score_mut = get_score(model1, mut_seq)
    mut_psi = expit(score_mut - score_wt + logit(wt_psi))
    return mut_psi - wt_psi

def get_dpsi_simple(model, wt_seqs, mut_seqs, wt_psi):
    vals = []
    for wt_seq, mut_seq in zip(wt_seqs, mut_seqs):
        vals.append((model.predict(mut_seq) - model.predict(wt_seq))[0][0])
    return np.array(vals)

def reverse_complement(seq):
    complements = {'A':'T', 'T':'A', 'G':'C', 'C':'G', 'N':'N'}
    new_seq = [complements[bp] for bp in reversed(seq)]
    return ''.join(new_seq)

def lookup_A5(chrom, pos):
    """
    Given a position, finds all events that contain an alternative exon that includes the given position.
    Returns a list of events - list of tuples 
    (event_id, gene_id, alt_exon_start, alt_exon_end, chrom, strand, baseline_psi)
    """
    conn = sqlite3.connect('../../alternative_splicing_snp_prediction/data/alt_5_gtex.db')
    cur = conn.cursor()
    # leaving a
    cur.execute("SELECT * FROM alt_5 WHERE chrom=? AND alt_exon_start < ? AND alt_exon_end > ?;",
                (chrom, pos+3, pos-6))
    results = cur.fetchall()
    return results

def lookup_SE(chrom, pos):
    """
    Given a position, finds all events that contain an alternative exon that includes the given position.
    Returns a list of events - list of tuples 
    (event_id, gene_id, alt_exon_start, alt_exon_end, chrom, strand, baseline_psi)
    """
    conn = sqlite3.connect('./../alternative_splicing_snp_prediction/data/skipped_exon_gtex.db')
    cur = conn.cursor()
    # leaving a
    cur.execute("SELECT * FROM skipped_exon WHERE chrom=? AND alt_exon_start < ? AND alt_exon_end > ?;",
                (chrom, pos+3, pos-6))
    results = cur.fetchall()
    return results

def lookup_sequence_db(event_id, cur):
    """
    Given an event id (alt_5 or skipped exon), finds the alternative exon
    corresponding to that event.
    """
    cur.execute('SELECT seq FROM exons WHERE event_id=?', (event_id,))
    results = cur.fetchone()
    return results[0]

def extract_se_data():
    conn = sqlite3.connect('../../alternative_splicing_snp_prediction/data/skipped_exon_gtex.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM skipped_exon;')
    data_out = []
    conn2 = sqlite3.connect('../../alternative_splicing_snp_prediction/data/alt_exons.db')
    cur2 = conn2.cursor()
    for e in cur.fetchall():
        event_id = e[0]
        gene = e[1]
        start = e[2]
        end = e[3]
        strand = e[5]
        seq = lookup_sequence_db(event_id, cur2)
        if strand==0:
            seq = reverse_complement(seq)
        wt_psi = e[6]
        data_out.append((event_id, seq, wt_psi))
    return data_out

def construct_data_set():
    """
    Constructs training/test data sets from genomic skipped exon data.

    Returns:
        G_train: array of training data
        G_test
        P_train: array of splicing ratios
        P_test
    """
    data = extract_se_data()
    genomic_exons = [c[1] for c in data]
    genomic_exons_encoded = [nn_utils.seq_to_array(s) for s in genomic_exons]
    actual = [c[2] for c in data]
    new_actual = []
    new_g = []
    max_len = 200
    min_len = 100
    for a, g in zip(actual, genomic_exons_encoded):
        if a!=None and a>0 and a<1 and len(g)<=max_len and len(g) >= min_len:
            new_actual.append(a)
            new_g.append(g)
    new_g2 = map(lambda x: np.rollaxis(sequence.pad_sequences(np.rollaxis(x, 1), max_len), 1), new_g)
    new_g2 = np.dstack(new_g2)
    new_g2 = np.rollaxis(new_g2, 2)
    G_train, G_test, P_train, P_test = train_test_split(new_g2, new_actual, test_size=0.2, random_state=0)
    return G_train, G_test, P_train, P_test


def load_hal_test_data(count_threshold=1, filter_effect=True, use_hetero=True):
    """
    Load data from Rosenberg 2015

    Returns:
        wt_seqs_encoded
        mut_seqs_encoded
        wt_psi
        hal_pred_dpsi
        actual_dpsi
    """
    table1 = pd.read_table('../data/mmc2.tsv')
    # TODO: select only mutations that don't occur in a splice site
    wt_seqs = table1.WT_SEQ.str.slice(198, -194)
    mut_seqs = table1.MUT_SEQ.str.slice(198, -194)
    is_alt_exon = table1.ALT_EXON_MUT
    wt_counts = table1.WT_COUNTS >= 10
    large_effect = table1.LARGE_HETERO_EFFECT
    if not use_hetero:
        large_effect = table1.LARGE_HOMO_EFFECT
    if not filter_effect:
        large_effect = [True]*len(wt_counts)
    hetero_counts = table1.HETERO_COUNTS >= count_threshold
    if not use_hetero:
        hetero_counts = table1.HOMO_COUNTS >= count_threshold
    wt_seqs_encoded = [nn_utils.seq_to_array(x) for x in wt_seqs]
    mut_seqs_encoded = [nn_utils.seq_to_array(x) for x in mut_seqs]
    wt_seqs_encoded = map(lambda x: x.reshape((1, x.shape[0], x.shape[1])), wt_seqs_encoded)
    mut_seqs_encoded = map(lambda x: x.reshape((1, x.shape[0], x.shape[1])), mut_seqs_encoded)
    wt_psi = []
    hal_pred = []
    actual_dpsi = []
    indices = []
    i = 0
    psis = zip(table1.HETERO_DPSI_PRED, table1.WT_PSI, table1.HETERO_DPSI)
    if not use_hetero:
        psis = zip(table1.HOMO_DPSI_PRED, table1.WT_PSI, table1.HOMO_DPSI)
    for h, w, a in psis:
        if h!=None and h>=-1 and h<=1 and a>=-1 and a<=1 and is_alt_exon[i] and hetero_counts[i] and wt_counts[i] and large_effect[i]:
            hal_pred.append(h)
            wt_psi.append(w)
            actual_dpsi.append(a)
            indices.append(i)
        i += 1
    wt_seqs_encoded = [wt_seqs_encoded[i] for i in indices]
    mut_seqs_encoded = [mut_seqs_encoded[i] for i in indices]
    return wt_seqs_encoded, mut_seqs_encoded, np.array(wt_psi), np.array(hal_pred), np.array(actual_dpsi)

# TODO: extract skipped exon events...

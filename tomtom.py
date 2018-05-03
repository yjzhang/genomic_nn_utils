"""
Tools for running tomtom and analyzing the output...
"""
import os
import subprocess

import numpy as np
import pandas as pd

import draw_logo

def tomtom(motifs, motif_filename, output_dir, db_file):
    """
    Runs tomtom with the given motif set, outputting to the provided
    output_dir.
    """
    draw_logo.meme_output_file(motifs, motif_filename)
    retcode = subprocess.call(['tomtom', '-o', output_dir, motif_filename, db_file])
    return retcode

def tomtom_analysis(motifs, motif_filename, output_dir, db_file):
    """
    Finds significant matches???
    """
    results = pd.read_table(os.path.join(output_dir, 'tomtom.txt'))
    return results

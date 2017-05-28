import tensorflow as tf
from ntm_cell import NTMCell
from ntm import NTM

import tasks.copy as tc

input_dim=10
output_dim=10

sess = tf.InteractiveSession()

scope='NTM-copy'
cell = NTMCell(input_dim=input_dim, output_dim=output_dim)
ntm = NTM(cell, sess, 1, 10, test_max_length=10, forward_only=True, scope=scope)

ntm.load('checkpoint', 'copy')


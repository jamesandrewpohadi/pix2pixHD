# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function
from data_load import load_data
import tensorflow as tf
from scipy.io.wavfile import write
import os
import numpy as np
from test_one import infer


def synthesize(pre):
    if not os.path.exists(hp.sampledir):
        os.mkdir(hp.sampledir)
    infer(pre+1,name)

if __name__ == '__main__':
    synthesize()
    print("Done")


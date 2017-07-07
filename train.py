#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2017 Intel Nervana 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Trains Character-level CNN for Text Classification.

Reference:
    "Character-level Convolutional Networks for Text Classification
    https://arxiv.org/pdf/1509.01626.pdf"
Usage:
    python character/train.py -e 20
"""

from neon.backends import gen_backend
from neon.callbacks.callbacks import Callbacks
from neon.data import MNIST
from neon.initializers import Gaussian
from neon.layers import Conv, Pooling, Dropout, Linear, GeneralizedCost
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Misclassification, CrossEntropyMulti, MeanSquared
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.data import PTB
#from neon.data import AGs
# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

time_steps = 150

# setup backend
#be = gen_backend(**extract_valid_args(args, gen_backend))
be = gen_backend(backend='gpu', batch_size=128)

# download penn treebank
dataset = PTB(time_steps, path=args.data_dir)
train_set = dataset.train_iter
valid_set = dataset.valid_iter

# setup weight initialization function
linit = Gaussian(loc=0.0, scale=0.02)
sinit = Gaussian(loc=0.0, scale=0.05)
# setup layers 

relu = Rectlin()

# Large CNN

llayers = [

    Conv((7, 7, 1024), init=linit, activation=relu),
    Pooling((3, max)),
    Conv((7, 7, 1024), init=linit, activation=relu),
    Pooling((3, max)),
    Conv((3, 3, 1024), init=linit, activation=relu),
    Conv((3, 3, 1024), init=linit, activation=relu),
    Conv((3, 3, 1024), init=linit, activation=relu),
    Conv((3, 3, 1024), init=linit, activation=relu),
    Pooling((3, max)),
    Linear(nout=2048, init=linit),
    Dropout(keep=.5),
    Linear(nout=2048, init=linit),
    Dropout(keep=.5),
    Linear(nout=4, init=linit)
]

# Small CNN
sconv = dict(init=sinit, batch_norm=False, activation=relu)
slayers = [

    Conv((7, 7, 256), init=sinit, activation=relu),
    Pooling(3, max),
    Conv((7, 7, 256), init=sinit, activation=relu),
    Pooling(3, max),
    Conv((3, 3, 256), init=sinit, activation=relu),
    Conv((3, 3, 256), init=sinit, activation=relu),
    Conv((3, 3, 256), init=sinit, activation=relu),
    Conv((3, 3, 256), init=sinit, activation=relu),
    Pooling(3, max),
    Linear(nout=1024, init=sinit),
    Dropout(keep=.5),
    Linear(nout=1024, init=sinit),
    Dropout(keep=.5),
    Linear(nout=4, init=sinit)
]

# setup cost function
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

# setup optimizer
optimizer = GradientDescentMomentum(0.01, momentum_coef=0.9, stochastic_round=args.rounding)


# initialize model object
charlcnn = Model(layers=llayers)
charscnn = Model(layers=slayers)

# configure callbacks
callbacks = Callbacks(charscnn, eval_set=valid_set, **args.callback_args)

# run fit for small CNN
charscnn.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
print('Misclassification error = %.1f%%' % (charscnn.eval(valid_set, metric=Misclassification())*100))                         
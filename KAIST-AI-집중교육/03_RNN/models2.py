from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

from utils import _linear
from utils import RNNCell

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class BasicRNNCell(RNNCell):
  """The most basic RNN cell.

  Args:
    num_units: int, The number of units in the LSTM cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
  """

  def __init__(self, num_units, activation=None, reuse=None):
    super(BasicRNNCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  #self: self class, inputs: first param, state:default null(or 0) 
  def call(self, inputs, state):
    ### your code here ###
    print(self)
    print(inputs)
    print(state)

    #_linear:
    output = self._activation(_linear([inputs, state], self._num_units, True))

    #####################
	#hydden layer and hydden layer
    return output, output


class GRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(GRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    print(self)
    print(inputs)
    print(state)
    """Gated recurrent unit (GRU) with nunits cells."""
    ### your code here ###
    with tf.variable_scope("r"):
        r = tf.sigmoid(_linear([inputs, state], self._num_units, True))
    with tf.variable_scope("z"):
        z = tf.sigmoid(_linear([inputs, state], self._num_units, True))

    c = self._activation(_linear([inputs, r * state], self._num_units, True))
    new_h = (1-z)

    #####################
    return new_h, new_h


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order.

  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype


class BasicLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
  that follows.
  """

  def __init__(self, num_units, forget_bias=1.0,
               state_is_tuple=True, activation=None, reuse=None):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(BasicLSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    print(inputs)
    print(inputs)
    print(state)
    """Long short-term memory cell (LSTM)."""
    #h:before hydden layer, c:before cell
    c, h = state

    ### your code here ###
    with tf.variable_scope("i"):
        i = tf.sigmoid(_linear([inputs, h], self._num_units, True))
    with tf.variable_scope("f"):
        f = tf.sigmoid(_linear([inputs, h], self._num_units, True))
    with tf.variable_scope("o"):
        o = tf.sigmoid(_linear([inputs, h], self._num_units, True))
    with tf.variable_scope("g"):
        g = self._activation(_linear([inputs, h], self._num_units, True))
        #g = tf.sigmoid(_linear([inputs, h], self._num_units, True))

    new_c = f * c + i *g
    new_h = o * self._activation(new_c)

    ######################
    new_state = LSTMStateTuple(new_c, new_h)
    
    #tensorflow method needs 2 vector: hydden layer and next state(cell)
    return new_h, new_state
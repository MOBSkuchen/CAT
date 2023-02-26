"""Utility functions and classes. """
import heapq

import numpy as np
import tensorflow as tf


class CosineDecayLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
    """Defines cosine decay learning rate."""

    def __init__(
            self, learning_rate, decay_steps, alpha, warmup_steps, warmup_lr):
        """Constructor.
    Args:
      learning_rate: float scalar, the base learning rate.
      decay_steps: int scalar, num of steps to decay over.
      alpha: float scalar, minimum learning rate value as a fraction of
        learning rate.
      warmup_steps: int scalar, the num of warm-up steps.
      warmup_lr: float scalar, learning rate for warm-up steps.
    """
        super(CosineDecayLearningRateSchedule, self).__init__()
        self._learning_rate = learning_rate
        self._decay_steps = decay_steps
        self._alpha = alpha
        self._warmup_steps = warmup_steps
        self._warmup_lr = warmup_lr

    def __call__(self, global_step):
        """Computes learning rate.
    Args:
      global_step: int scalar tensor, the current global step.
    Returns:
      learning_rate: float scalar tensor, the learning rate as a function of
        the input `global_step`.
    """
        global_step = tf.cast(global_step, 'float32')

        cosine_decay = 0.5 * (1 + tf.cos(np.pi * tf.minimum(global_step
                                                            - self._warmup_steps,
                                                            self._decay_steps) / self._decay_steps))
        decayed = (1 - self._alpha) * cosine_decay + self._alpha
        decayed_learning_rate = self._learning_rate * decayed

        decayed_learning_rate = tf.where(global_step < self._warmup_steps,
                                         self._warmup_lr,
                                         decayed_learning_rate)

        return decayed_learning_rate


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule."""

    def __init__(self, learning_rate, hidden_size, warmup_steps):
        """Constructor.
    Args:
      learning_rate: float scalar, the base learning rate.
      hidden_size: int scalar, the hidden size of continuous representation.
      warmup_steps: int scalar, the num of warm-up steps
    """
        super(LearningRateSchedule, self).__init__()
        self._learning_rate = learning_rate
        self._hidden_size = hidden_size
        self._warmup_steps = tf.cast(warmup_steps, 'float32')

    def __call__(self, global_step):
        """Computes learning rate with linear warmup and rsqrt decay.
    Args:
      global_step: int scalar tensor, the current global step.
    Returns:
      learning_rate: float scalar tensor, the learning rate as a function of
        the input `global_step`.
    """
        global_step = tf.cast(global_step, 'float32')
        learning_rate = self._learning_rate
        learning_rate *= (self._hidden_size ** -0.5)
        # linear warmup
        learning_rate *= tf.minimum(1.0, global_step / self._warmup_steps)
        # rsqrt decay
        learning_rate /= tf.sqrt(tf.maximum(global_step, self._warmup_steps))
        return learning_rate


def save_attention_weights(filename, data):
    """Saves attention weights data to *.npy file.
  Args:
    filename: string scalar, filename.
    data: a list or tuple or dict of numpy arrays, the attention weights and
      token ids of input and translated sequence.
  """
    np.save(filename, data)


def dict_to_example(dictionary):
    """Convert dict to protobuf example message.
  Args:
    dictionary: a dict mapping string to list of integers
  Returns:
    a protobuf example message.
  """
    features = {}
    for k, v in dictionary.items():
        features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    return tf.train.Example(features=tf.train.Features(feature=features))


def nucleus_sampling(scores, threshold=0.95):
    """Sample from the head of the probability distribution that contains the
  vast majority of probability mass. See https://arxiv.org/abs/1904.09751
  for details. The distribution is truncated to the  and re-normalized.
  Args:
    scores: numpy array of shape [vocab_size], the probability distribution (
      sum to one) of all possible next-tokens over the vocabulary.
    threshold: float scalar, the minimum value of the sum of probability mass
      that the head of the distribution must exceed.
  Returns:
    next_token_id: int scalar, the sampled id of the next token.
  """
    ids = np.argsort(-scores)
    cumsum = [0.] + np.cumsum(scores[ids]).tolist()
    # search space is any value >= low and <= high
    low, high = 0, len(cumsum) - 2

    while low <= high:
        mid = (low + high) // 2
        sum1 = cumsum[mid]
        sum2 = cumsum[mid + 1]
        if sum1 < threshold and sum2 >= threshold:
            break
        elif sum2 < threshold:  # exclude indices <= mid
            low = mid + 1
        elif sum1 >= threshold:  # exclude indices >= mid
            high = mid - 1
        else:
            raise ValueError('Impossible outcome')

    probs = scores[ids[:mid + 1]] / sum2
    next_token_id = np.random.choice(ids[:mid + 1], p=probs)
    return next_token_id


def topk_sampling(scores, k=40):
    """Sample from the top-k tokens with the largest probability. The distribution
   is truncated and re-normalized.
  Args:
    scores: numpy array of shape [vocab_size], the probability distribution (
      sum to one) of all possible next-tokens over the vocabulary.
    k: int scalar, the num of next-tokens with largest probability to sample
      from.
  Returns:
    next_token_id: int scalar, the sampled id of the next token.
  """
    min_pq = list(zip(scores[:k], range(k)))
    heapq.heapify(min_pq)
    for i in np.arange(k, len(scores)):
        if scores[i] > min_pq[0][0]:
            min_pq[0] = scores[i], i
            heapq.heapify(min_pq)

    probs, ids = list(zip(*min_pq))
    probs = np.array(probs)
    probs /= probs.sum()
    next_token_id = np.random.choice(ids, p=probs)
    return next_token_id


def rel_shift(inputs):
    """Shift the matrix in the input tensor, so that the query position matches
  correctly with the key position for computing attention scores.
  Given input tensor `x` of shape [batch_size, num_heads, q_seq_len, r_seq_len],
  each slice `x[i, j]` is a matrix of shape [q_seq_len, r_seq_len] (Note that
  generally `r_seq_len` >= `q_seq_len`
  the matrix `x[i, j]` in the output will be a left-shifted version of the input
  , where the 0th, 1st, ..., and `q_seq_len - 1`-th row will be left-shifted by
  `q_seq_len - 1`, `q_seq_len - 2`, ..., and 0 positions.
  Args:
    inputs: float tensor of shape [batch_size, num_heads, q_seq_len, r_seq_len],
      the input tensor.
  Returns:
    outputs: float tensor of shape [batch_size, num_heads, q_seq_len, r_seq_len]
      , the shifted tensor.
  """
    shape = tf.shape(inputs)
    padded = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [1, 0]])
    reshaped = tf.reshape(padded, [shape[0], shape[1], shape[3] + 1, shape[2]])
    sliced = reshaped[:, :, 1:]
    outputs = tf.reshape(sliced, shape)
    return outputs


def get_padding_mask(inputs, padding_value=0):
    """Creates a binary tensor to mask out padded tokens.
  Args:
    inputs: int tensor of shape [batch_size, src_seq_len], token ids
      of source sequences.
    padding_value: int scalar, the vocabulary index of the PAD token.
  Returns:
    mask: binary tensor of shape [batch_size, 1, 1, src_seq_len], storing ones
      for padded tokens and zeros for regular tokens.
  """
    mask = tf.cast(tf.equal(inputs, padding_value), 'float32')
    mask = mask[:, tf.newaxis, tf.newaxis, :]
    return mask


def get_look_ahead_mask(seq_len):
    """Creates a tensor to mask out future tokens in the target sequences when in
  training mode.
  Given sequence length `L` of target sequence, the mask would be a L x L
  matrix (when `tf.squeeze`'ed) where upper diagonal entries are ones and all
  other entries zeros.
  0, 1, 1, ..., 1
  0, 0, 1, ..., 1

      ... ...

  0, 0, 0, ..., 0
  Args:
    seq_len: int scalar tensor, sequence length.
  Returns:
    mask: float tensor of shape [1, 1, seq_len, seq_len], the mask tensor.
  """
    mask = 1 - tf.linalg.band_part(tf.ones([seq_len, seq_len]), -1, 0)
    mask = mask[tf.newaxis, tf.newaxis, :, :]
    return mask


def get_positional_encoding(seq_len, hidden_size, reverse=False):
    """Creates a tensor that encodes positional information.
  Args:
    seq_len: int scalar tensor, sequence length.
    hidden_size: int scalar, the hidden size of continuous representation.
    reverse: bool, whether to reverse the sequence. Defaults to False.
  Returns:
    positional_encoding: float tensor of shape [seq_len, hidden_size], the
      tensor that encodes positional information.
  """
    distances = tf.cast(tf.range(seq_len), 'float32')
    hidden_size //= 2
    inverse_frequencies = 1 / (
            10000 ** (tf.cast(tf.range(hidden_size), 'float32') / (hidden_size - 1)))
    positional_encoding = tf.einsum('i,j->ij', distances, inverse_frequencies)
    positional_encoding = tf.concat([tf.sin(positional_encoding),
                                     tf.cos(positional_encoding)], axis=1)
    return positional_encoding


def compute_loss(labels, logits, smoothing, vocab_size, padding_value=0):
    """Computes average (per-token) cross entropy loss.
  1. Applies label smoothing -- all entries in the groundtruth label tensor
     get non-zero probability mass.
  2. Computes per token loss of shape [batch_size, tgt_seq_len], where padded
     positions are masked, and then the sum of per token loss is normalized by
     the total number of non-padding entries.
  Args:
    labels: int tensor of shape [batch_size, tgt_seq_len], the groundtruth
      token ids.
    logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size], the
      predicted logits of tokens over the vocabulary.
    smoothing: float scalar, the amount of label smoothing applied to the
      one-hot class labels.
    vocab_size: int scalar, num of tokens (including SOS and EOS) in the
      vocabulary.
    padding_value: int scalar, the vocabulary index of the PAD token.
  Returns:
    loss: float scalar tensor, the per-token cross entropy
  """
    # effective_vocab = vocab - {SOS_ID}
    effective_vocab_size = vocab_size - 1

    # prob mass allocated to the token that should've been predicted
    on_value = 1.0 - smoothing
    # prob mass allocated to all other tokens
    off_value = smoothing / (effective_vocab_size - 1)

    # [batch_size, tgt_seq_len, vocab_size]
    labels_one_hot = tf.one_hot(
        labels,
        depth=vocab_size,
        on_value=on_value,
        off_value=off_value)

    # compute cross entropy over all tokens in vocabulary but SOS_ID (i.e. 0)
    # because SOS_ID should never appear in the decoded sequence
    # [batch_size, tgt_seq_len]
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_one_hot[:, :, 1:], logits=logits[:, :, 1:])

    # this is the entropy when the softmax'ed logits == groundtruth labels
    # so it should be deducted from `cross_entropy` to make sure the minimum
    # possible cross entropy == 0
    normalizing_constant = -(on_value * tf.math.log(on_value) +
                             (effective_vocab_size - 1) * off_value * tf.math.log(off_value + 1e-20))
    cross_entropy -= normalizing_constant

    # mask out predictions where the labels == `padding_value`
    weights = tf.cast(tf.not_equal(labels, padding_value), 'float32')
    cross_entropy *= weights
    loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(weights)
    return loss

import tensorflow as tf
import jsonlines as jsl
import multiprocessing as mp


computed = []


def _single_load(i, op):
    p = jsl.open(i)
    for n in p.iter():
        n = n["text"]
        op.write(n + '\n')


def load(inputs, output):
    open(output, 'w').write('')
    op = open(output, 'a')
    for i in inputs:
        p = mp.Process(target=_single_load, args=(i, op))
        p.start()


def _trunc(element):
    return element["text"]


def _single_compute(inp):
    out = inp + ".tfrecord"
    dataset = tf.data.Dataset.from_generator(jsl.open(inp))
    dataset.map(_trunc)  # Get text
    writer = tf.data.experimental.TFRecordWriter(out)
    writer.write(dataset)
    computed.append(out)


def compute(input_files):
    for i in input_files:
        p = mp.Process(target=_single_compute, args=(i, ))
        p.start()

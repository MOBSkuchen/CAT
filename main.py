import tensorflow as tf
from model import TransformerModel
from api import tokenization, pile, dataset, utils, runner, config
from glob import glob

cfg = config.config

swt = cfg["swt"]
vto = cfg["vto"]
ds_pat = cfg["dataset"]
model_dir = cfg["model_cp"]


def create_vocab():
    files = glob(ds_pat)[:2]  # Only use 2 DS
    pile.load(files, vto)
    pile.compute(files)
    x: tokenization.subword.SubTokenizer = tokenization.create_subtokenizer_from_raw_text_files([vto], None, 10,
                                                                                                min_count=10000)
    x.save_to_file(swt)
    return x


def load_vocab():
    return tokenization.restore_subtokenizer_from_vocab_files(swt)


def create_model(vocab_size):
    _model = TransformerModel(vocab_size=vocab_size,
                              encoder_stack_size=cfg["esz"],
                              decoder_stack_size=cfg["dsz"],
                              hidden_size=cfg["hidden_size"],
                              num_heads=cfg["heads"],
                              filter_size=cfg["filter_size"],
                              dropout_rate=cfg["dropout_rate"])
    return _model


def train(_model, filenames):
    builder = dataset.DynamicBatchDatasetBuilder(
        cfg["max_num_tokens"], True, cfg["max_length"], cfg["num_parallel_calls"])
    train_ds = builder.build_dataset(filenames)

    # learning rate and optimizer
    optimizer = tf.keras.optimizers.Adam(
        utils.LearningRateSchedule(cfg["learning_rate"],
                                   cfg["hidden_size"],
                                   cfg["learning_rate_warmup_steps"]),
        cfg["optimizer_adam_beta1"],
        cfg["optimizer_adam_beta2"],
        epsilon=cfg["optimizer_adam_epsilon"])

    # checkpoint
    ckpt = tf.train.Checkpoint(model=_model, optimizer=optimizer)

    # build trainer and start training
    trainer = runner.SequenceTransducerTrainer(_model, cfg["label_smoothing"])
    trainer.train(
        train_ds, optimizer, ckpt, model_dir, cfg["num_steps"], cfg["save_ckpt_per_steps"])


vocab = create_vocab()
model = create_model(vocab.vocab_size)
train(model, pile.computed)

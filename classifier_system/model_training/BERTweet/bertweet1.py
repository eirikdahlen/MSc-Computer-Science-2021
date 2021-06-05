import pandas as pd
import numpy as np
import time
from matplotlib import pyplot
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os


def load_and_setup_dataset(filename: str):
    df = pd.read_csv(filename)
    return df['text'].values.tolist(), df["label"].values.tolist()


def create_test_val_dataset(X, y, train_size: float, test_size=None, random_state: int = 42):
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, test_size=test_size,
                                                      random_state=random_state, stratify=y)
    return X_train, X_val, y_train, y_val


def to_categorical_labels(y_train, y_val, y_test, binary: bool = False):
    labels_dict = {'unrelated': 0, 'proED': 1, 'prorecovery': 0 if binary else 2}

    for i in range(len(y_train)):
        y_train[i] = labels_dict[y_train[i]]
    for i in range(len(y_val)):
        y_val[i] = labels_dict[y_val[i]]
    for i in range(len(y_test)):
        y_test[i] = labels_dict[y_test[i]]
    return np.array(y_train), np.array(y_val), np.array(y_test)


def tokenize(tokenizer, X_train, X_val, X_test, truncation: bool = True, padding: bool = True):
    train_encodings = tokenizer(X_train, truncation=truncation, padding=padding)
    val_encodings = tokenizer(X_val, truncation=truncation, padding=padding)
    test_encodings = tokenizer(X_test, truncation=truncation, padding=padding)
    train_encodings = np.array(list(dict(train_encodings).values()))
    val_encodings = np.array(list(dict(val_encodings).values()))
    test_encodings = np.array(list(dict(test_encodings).values()))
    return train_encodings, val_encodings, test_encodings


def train_model(model,
                train_encodings,
                y_train,
                val_encodings,
                y_val,
                batch_size: int,
                learning_rate: float,
                epochs: int,
                checkpoint_path: str,
                save_model_weights: bool = True,
                use_class_weights: bool = False):
    if save_model_weights:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=model.compute_loss,
                  metrics=['accuracy'])

    if use_class_weights:
        class_weight = calculate_class_weights(y_train)

    history = model.fit(
        x=train_encodings[0],
        y=y_train,
        validation_data=(val_encodings[0], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cp_callback] if save_model_weights else None,
        shuffle=True,
        class_weight=class_weight if use_class_weights else None
    )
    print(history.history)
    return model, history


def load_weights(model, cp_dir: str):
    latest = tf.train.latest_checkpoint(cp_dir)
    model.load_weights(latest)
    return model


def calculate_class_weights(y_train):
    weights = {}
    y_train = y_train.tolist()
    length = len(y_train)
    weights[0] = ((1 / y_train.count(0)) * length)
    weights[1] = ((1 / y_train.count(1)) * length)
    weights[2] = ((1 / y_train.count(2)) * length / 3.5)
    return weights


def predict(model, test_data, test_labels, softmax: bool = False):
    print("Performing predictions...")
    logits = model.predict(test_data[0])["logits"]
    if softmax:
        predictions_probabilities = tf.nn.softmax(logits, axis=1)
        print(predictions_probabilities)

    classes = np.argmax(logits, axis=-1)
    score = classification_report(test_labels, classes, digits=3)
    print(score)
    return predictions_probabilities if softmax else logits


def plot_stats(history, should_show=True):
    # plot loss during training
    pyplot.figure(1)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.xlabel('Epoch')
    pyplot.legend()
    pyplot.savefig('loss_bertweet1_soft.png')
    # plot accuracy during training
    pyplot.figure(2)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='validation')
    pyplot.xlabel('Epoch')
    pyplot.legend()
    pyplot.savefig('acc_bertweet1_soft.png')
    if should_show:
        pyplot.show()


def main(args):
    start = time.time()
    use_idun = args.idun
    load_model = args.loadmodel

    training_data_path = '/cluster/home/eirida/masteroppgave/Masteroppgave/data/dataset_training.csv' if use_idun else '../../data/dataset_training.csv'
    test_data_path = '/cluster/home/eirida/masteroppgave/Masteroppgave/data/dataset_test.csv' if use_idun else '../../data/dataset_test.csv'
    X, y = load_and_setup_dataset(training_data_path)
    X_test, y_test = load_and_setup_dataset(test_data_path)
    X_train, X_val, y_train, y_val = create_test_val_dataset(X, y, train_size=0.95)
    y_train, y_val, y_test = to_categorical_labels(y_train, y_val, y_test)
    train_encodings, val_encodings, test_encodings = tokenize(
        tokenizer=AutoTokenizer.from_pretrained('vinai/bertweet-base'),
        X_train=X_train,
        X_val=X_val,
        X_test=X_test
    )

    model = TFAutoModelForSequenceClassification.from_pretrained('vinai/bertweet-base',
                                                            num_labels=3,
                                                            return_dict=True)
    checkpoint_path = "bertweet1_soft_ckpt/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if not load_model:
        trained_model, history = train_model(model=model,
                                             train_encodings=train_encodings,
                                             y_train=y_train,
                                             val_encodings=val_encodings,
                                             y_val=y_val,
                                             batch_size=16,
                                             learning_rate=1e-5,
                                             epochs=4,
                                             checkpoint_path=checkpoint_path,
                                             save_model_weights=True,
                                             use_class_weights=False)
        plot_stats(history, should_show=not use_idun)
        trained_model.summary()
    else:
        trained_model = load_weights(model=model, cp_dir=checkpoint_dir)

    predictions = predict(model=trained_model,
                          test_data=test_encodings,
                          test_labels=y_test,
                          softmax=True)

    print(f"Used {time.time() - start} seconds")


if __name__ == "__main__":
    print("Starting run...")
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    print(f"Number of GPUs Avail: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"GPU Name: {tf.test.gpu_device_name()}")
    print(f"Cuda: {tf.test.is_built_with_cuda()}")

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--idun", default=False, type=bool)
    parser.add_argument("--loadmodel", default=False, type=bool)
    args = parser.parse_args()
    main(args)

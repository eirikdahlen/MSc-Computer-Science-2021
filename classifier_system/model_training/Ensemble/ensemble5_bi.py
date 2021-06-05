import os

import numpy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
import pandas as pd
import time
from matplotlib import pyplot
import numpy as np
from transformers import BertTokenizerFast, \
    TFBertForSequenceClassification, \
    AutoTokenizer, \
    TFAutoModelForSequenceClassification, \
    DistilBertTokenizerFast, \
    TFDistilBertForSequenceClassification, \
    RobertaTokenizer,\
    TFRobertaForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import pickle
import random
import string
from argparse import ArgumentParser
from typing import List, Union


def voting_classifier(predictions: Union[List[List[List[float]]], List[List[int]]], num_labels: int,
                      voting: str = 'soft') -> List[int]:
    """
    Args:
        predictions: (if soft) list of lists of lists with predictions,
                     (if hard) list of lists with predictions
        num_labels: number of classes in dataset
        voting: hard or soft voting

    Returns: list of predictions
    """
    final_predictions = []
    for data_instance_index in range(len(predictions[0])):
        counters = [0 for _ in range(num_labels)]
        for prediction in predictions:
            if voting == 'soft':
                for i in range(len(prediction[data_instance_index])):
                    counters[i] += prediction[data_instance_index][i]
            elif voting == 'hard':
                counters[prediction[data_instance_index]] += 1
        final_predictions.append(counters.index(max(counters)))

    return final_predictions


def load_and_setup_dataset(filename: str):
    df = pd.read_csv(filename)
    posts = df['text'].values.tolist()
    bios_posts = df[['description', 'text']]
    labels = df["label"].values.tolist()
    return posts, bios_posts, labels


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def predict_classifier(model, test_data):
    predict_proba_dist = model.decision_function(test_data)
    pred_probability = []
    for eachArr in predict_proba_dist:
        pred_probability.append(softmax(eachArr).tolist())
    return pred_probability


def setup_classifier(data, model_filename, mapper_filename):
    letters = string.ascii_lowercase
    data['description'] = data['description'].fillna(''.join(random.choice(letters) for _ in range(7)))
    mapper = pickle.load(open(mapper_filename, 'rb'))
    features = mapper.transform(data)
    model = pickle.load(open(model_filename, 'rb'))
    return model, features


def to_categorical_labels(y_test, binary: bool = False):
    labels_dict = {'unrelated': 0, 'proED': 1, 'prorecovery': 0 if binary else 2}

    for i in range(len(y_test)):
        y_test[i] = labels_dict[y_test[i]]
    return np.array(y_test)


def tokenize_transformers(tokenizer, data):
    test_encodings = tokenizer(data, truncation=True, padding=True)
    return np.array(list(dict(test_encodings).values()))


def load_weights(model, cp_dir: str):
    latest = tf.train.latest_checkpoint(cp_dir)
    model.load_weights(latest)
    return model


def predict_transformers(model, test_data, softmax: bool = True):
    print("Performing predictions...")
    logits = model.predict(test_data[0])["logits"]
    if softmax:
        predictions_probabilities = tf.nn.softmax(logits, axis=1)
    else:
        classes = np.argmax(logits, axis=-1)

    return predictions_probabilities.numpy().tolist() if softmax else classes.numpy().tolist()


def get_score(test_labels, test_predictions):
    score = classification_report(test_labels, test_predictions, digits=3)
    print(score)


def define_nn(n_models: int,
              output_dim: int = 3,
              lr: float = 0.001,
              neurons1: int = 512,
              neurons2: int = 256,
              dropout: float = 0.2):
    input_dim = n_models * output_dim
    model = Sequential()
    model.add(Dense(neurons1, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(neurons2, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def flatten_predictions(predictions: List[List[List[float]]]):
    flat_preds = []
    for i in range(len(predictions[0])):
        preds = []
        for model_prediction in predictions:
            for j in range(len(model_prediction[i])):
                preds.append(model_prediction[i][j])
        flat_preds.append(preds)
    return np.array(flat_preds)


def lr_time_based_decay(epoch, lr):
    decay = 0.001 / 50
    return lr / (1 + decay * epoch)


def plot_stats(history, should_show=True):
    # plot loss during training
    pyplot.figure(1)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.xlabel('Epochs')
    pyplot.legend()
    pyplot.savefig('loss_ffnn5_bi.png')
    # plot accuracy during training
    pyplot.figure(2)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    # pyplot.plot(history.history['val_accuracy'], label='validation')
    pyplot.xlabel('Epochs')
    pyplot.legend()
    pyplot.savefig('acc_ffnn5_bi.png')
    if should_show:
        pyplot.show()


def main(args):
    start = time.time()
    use_idun = args.idun

    train_data_path = '/cluster/home/eirida/masteroppgave/Masteroppgave/data/dataset_training.csv' if use_idun else '../../data/dataset_training.csv'
    test_data_path = '/cluster/home/eirida/masteroppgave/Masteroppgave/data/dataset_test.csv' if use_idun else '../../data/dataset_test.csv'
    X_train_transformers, X_train_svm, y_train = load_and_setup_dataset(train_data_path)
    X_test_transformers, X_test_svm, y_test = load_and_setup_dataset(test_data_path)
    print(len(X_train_transformers), len(X_train_svm), len(y_train))
    print(len(X_test_transformers), len(X_test_svm), len(y_test))
    y_train = to_categorical_labels(y_train, binary=True)
    y_test = to_categorical_labels(y_test, binary=True)

    num_labels = len(set(y_train))
    svm_filepath = '/cluster/home/eirida/masteroppgave/Masteroppgave/model_training/SVM/svm-bio-tweet.pkl' if use_idun else '../SVM/svm-bio-tweet.pkl'
    mapper_filepath = '/cluster/home/eirida/masteroppgave/Masteroppgave/model_training/SVM/mapper.pkl' if use_idun else '../SVM/mapper.pkl'

    svm_clf, X_train_svm = setup_classifier(data=X_train_svm,
                                            model_filename=svm_filepath,
                                            mapper_filename=mapper_filepath)
    _, X_test_svm = setup_classifier(data=X_test_svm,
                                     model_filename=svm_filepath,
                                     mapper_filename=mapper_filepath)
    print("Load tokenizers")
    bert_large_tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
    bertweet_tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
    ernie_large_tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-2.0-large-en')
    distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    print("Tokenize...")
    train_encodings_bert = tokenize_transformers(tokenizer=bert_large_tokenizer,
                                                 data=X_train_transformers)
    train_encodings_bertweet = tokenize_transformers(tokenizer=bertweet_tokenizer,
                                                     data=X_train_transformers)
    train_encodings_ernie = tokenize_transformers(tokenizer=ernie_large_tokenizer,
                                                     data=X_train_transformers)
    train_encodings_distilbert = tokenize_transformers(tokenizer=distilbert_tokenizer,
                                                     data=X_train_transformers)
    # train_encodings_roberta = tokenize_transformers(tokenizer=roberta_tokenizer,
    #                                                 data=X_train_transformers)
    test_encodings_bert = tokenize_transformers(tokenizer=bert_large_tokenizer,
                                                data=X_test_transformers)
    test_encodings_bertweet = tokenize_transformers(tokenizer=bertweet_tokenizer,
                                                    data=X_test_transformers)
    test_encodings_ernie = tokenize_transformers(tokenizer=ernie_large_tokenizer,
                                                    data=X_test_transformers)
    test_encodings_distilbert = tokenize_transformers(tokenizer=distilbert_tokenizer,
                                                    data=X_test_transformers)
    # test_encodings_roberta = tokenize_transformers(tokenizer=roberta_tokenizer,
    #                                                data=X_test_transformers)
    print("Load models")
    bert_model = TFBertForSequenceClassification.from_pretrained('bert-large-uncased',
                                                                 num_labels=num_labels,
                                                                 return_dict=True)
    bertweet_model = TFAutoModelForSequenceClassification.from_pretrained('vinai/bertweet-base',
                                                                          num_labels=num_labels,
                                                                          return_dict=True)
    ernie_model = TFAutoModelForSequenceClassification.from_pretrained('nghuyong/ernie-2.0-large-en',
                                                                          num_labels=num_labels,
                                                                          return_dict=True)
    distilbert_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                  num_labels=num_labels,
                                                                  return_dict=True)
    # roberta_model = TFRobertaForSequenceClassification.from_pretrained('roberta-base',
    #                                                                    num_labels=num_labels,
    #                                                                    return_dict=True)
    bert_checkpoint_path = '/cluster/home/eirida/masteroppgave/Masteroppgave/bertL2_bi_ckpt/' if use_idun else 'bertL2_ckpt/cp-{epoch:04d}.ckpt'
    bertweet_checkpoint_path = '/cluster/home/eirida/masteroppgave/Masteroppgave/bertweet1_bi_ckpt/' if use_idun else 'bertweet1_ckpt/cp-{epoch:04d}.ckpt'
    ernie_checkpoint_path = '/cluster/home/eirida/masteroppgave/Masteroppgave/ernieL2_bi_ckpt/' if use_idun else 'ernieL2_ckpt/cp-{epoch:04d}.ckpt'
    distilbert_checkpoint_path = '/cluster/home/eirida/masteroppgave/Masteroppgave/distilbert2_bi_ckpt/' if use_idun else 'distilbert2_ckpt/cp-{epoch:04d}.ckpt'
    # distilbert_balanced_checkpoint_path = '/cluster/home/eirida/masteroppgave/Masteroppgave/distilbert7_ckpt/' if use_idun else 'distilbert7_ckpt/cp-{epoch:04d}.ckpt'
    # roberta_checkpoint_path = '/cluster/home/eirida/masteroppgave/Masteroppgave/roberta5_ckpt/' if use_idun else 'roberta5_ckpt/cp-{epoch:04d}.ckpt'
    bert_checkpoint_dir = os.path.dirname(bert_checkpoint_path)
    bertweet_checkpoint_dir = os.path.dirname(bertweet_checkpoint_path)
    ernie_checkpoint_dir = os.path.dirname(ernie_checkpoint_path)
    distilbert_checkpoint_dir = os.path.dirname(distilbert_checkpoint_path)
    # distilbert_balanced_checkpoint_dir = os.path.dirname(distilbert_balanced_checkpoint_path)
    # roberta_checkpoint_dir = os.path.dirname(roberta_checkpoint_path)

    print("Load weights")
    bert_trained_model = load_weights(model=bert_model, cp_dir=bert_checkpoint_dir)
    bertweet_trained_model = load_weights(model=bertweet_model, cp_dir=bertweet_checkpoint_dir)
    ernie_trained_model = load_weights(model=ernie_model, cp_dir=ernie_checkpoint_dir)
    distilbert_trained_model = load_weights(model=distilbert_model, cp_dir=distilbert_checkpoint_dir)
    # distilbert_balanced_trained_model = load_weights(model=distilbert_model, cp_dir=distilbert_balanced_checkpoint_dir)
    # roberta_trained_model = load_weights(model=roberta_model, cp_dir=roberta_checkpoint_dir)

    bert_train_predictions = predict_transformers(model=bert_trained_model, test_data=train_encodings_bert)
    bertweet_train_predictions = predict_transformers(model=bertweet_trained_model, test_data=train_encodings_bertweet)
    ernie_train_predictions = predict_transformers(model=ernie_trained_model, test_data=train_encodings_ernie)
    distilbert_train_predictions = predict_transformers(model=distilbert_trained_model, test_data=train_encodings_distilbert)
    # distilbert_balanced_train_predictions = predict_transformers(model=distilbert_balanced_trained_model,
    #                                                              test_data=train_encodings_distilbert)
    # roberta_train_predictions = predict_transformers(model=roberta_trained_model, test_data=train_encodings_roberta)
    svm_train_predictions = predict_classifier(model=svm_clf, test_data=X_train_svm)

    bert_test_predictions = predict_transformers(model=bert_trained_model, test_data=test_encodings_bert)
    bertweet_test_predictions = predict_transformers(model=bertweet_trained_model, test_data=test_encodings_bertweet)
    ernie_test_predictions = predict_transformers(model=ernie_trained_model, test_data=test_encodings_ernie)
    distilbert_test_predictions = predict_transformers(model=distilbert_trained_model, test_data=test_encodings_distilbert)
    # distilbert_balanced_test_predictions = predict_transformers(model=distilbert_balanced_trained_model,
    #                                                             test_data=test_encodings_distilbert)
    # roberta_test_predictions = predict_transformers(model=roberta_trained_model, test_data=test_encodings_roberta)
    svm_test_predictions = predict_classifier(model=svm_clf, test_data=X_test_svm)

    train_predictions = [bert_train_predictions, bertweet_train_predictions, svm_train_predictions,
                         ernie_train_predictions, distilbert_train_predictions]
    test_predictions = [bert_test_predictions, bertweet_test_predictions, svm_test_predictions, ernie_test_predictions,
                        distilbert_test_predictions]

    preds_train_nn = flatten_predictions(predictions=train_predictions)
    preds_test_nn = flatten_predictions(predictions=test_predictions)

    # preds_train_nn = numpy.load('preds_train_nn.npy')
    # preds_test_nn = numpy.load('preds_test_nn.npy')

    save_preds = True
    if save_preds:
        numpy.save(file='preds_train_nn5_bi.npy', arr=preds_train_nn)
        numpy.save(file='preds_test_nn5_bi.npy', arr=preds_test_nn)

    do_grid_search = True
    if do_grid_search:
        param_grid = {
            'n_models': [len(preds_train_nn[0]) // num_labels],
            'output_dim': [num_labels],
            'lr': [0.01, 0.001],
            'neurons1': [512, 1024],
            'neurons2': [256, 512],
            'dropout': [0.2],
            'epochs': [25, 50],
            'batch_size': [16, 32]
        }
        ffnn = KerasClassifier(build_fn=define_nn, verbose=0)
        grid = GridSearchCV(estimator=ffnn, param_grid=param_grid, cv=5, verbose=3)
        grid_result = grid.fit(preds_train_nn, y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        ffnn_preds = grid.predict(preds_test_nn)
    else:
        ffnn = define_nn(n_models=len(preds_train_nn[0]) // num_labels, output_dim=num_labels)
        initial_lr = 0.001
        epochs = 50
        batch_size = 16
        history = ffnn.fit(preds_train_nn,
                           y_train,
                           epochs=epochs,
                           batch_size=batch_size,
                           callbacks=[LearningRateScheduler(schedule=lr_time_based_decay, verbose=1)]
        )
        print(history.history)
        plot_stats(history, should_show=not use_idun)
        print(ffnn.summary())
        ffnn.save('ffnn_model_5_bi')
        # ffnn = tf.keras.models.load_model('ffnn_model_1/')
        ffnn_preds = ffnn.predict(preds_test_nn)
        ffnn_preds = np.argmax(ffnn_preds, axis=-1)

    print("Score from NN:")
    get_score(test_labels=y_test, test_predictions=ffnn_preds)
    conf_mat = confusion_matrix(y_test, ffnn_preds)
    cmd = ConfusionMatrixDisplay(conf_mat, display_labels=['Unrelated', 'Pro-ED'])
    cmd.plot(cmap='Blues')
    pyplot.savefig('cm_nn_5_bi.png')

    voting_preds = voting_classifier(predictions=test_predictions, num_labels=num_labels, voting='soft')
    print("Score from voting classifier:")
    get_score(test_labels=y_test, test_predictions=np.array(voting_preds))
    conf_mat = confusion_matrix(y_test, voting_preds)
    cmd = ConfusionMatrixDisplay(conf_mat, display_labels=['Unrelated', 'Pro-ED'])
    cmd.plot(cmap='Blues')
    pyplot.savefig('cm_vote_5_bi.png')

    print(f"Used {time.time() - start} seconds")


if __name__ == "__main__":
    print("Starting run...")
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    print(f"Number of GPUs Avail: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"GPU Name: {tf.test.gpu_device_name()}")
    print(f"Cuda: {tf.test.is_built_with_cuda()}")

    parser = ArgumentParser()
    parser.add_argument("--idun", default=False, type=bool)
    args = parser.parse_args()
    main(args)

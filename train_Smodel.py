import os
import time
import argparse
import random
import numpy as np
import tensorflow as tf
from configs.configs import Configs
from models.Smodel import Smodel
from utils.read_data import read_essays_single_score_words, read_word_vocab, read_essays_single_score, read_pos_vocab
from utils.general_utils import get_single_scaled_down_score, pad_hierarchical_text_sequences, \
    load_word_embedding_dict, build_embedd_table
from metrics.custom_metrics import KappaScore, CorrelationCoefficient
import matplotlib.pyplot as plt


def main():
    tf.config.run_functions_eagerly(True)
    model_name = 'Smodel_pos_features'

    parser = argparse.ArgumentParser(description="PAES_attributes model")
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--attribute_name', type=str, default='score', help='name of the attribute to be trained on')
    parser.add_argument('--train_mode', type=str, default='domain-adaptation')
    parser.add_argument('--with_features', type=str, default='no')
    parser.add_argument('--input_seq', type=str, default='words', help='words or pos')
    args = parser.parse_args()
    test_prompt_id = args.test_prompt_id
    attribute_name = args.attribute_name
    seed = args.seed
    train_mode = args.train_mode
    input_seq = args.input_seq
    if args.with_features == 'yes':
        with_features = True
    elif args.with_features == 'no':
        with_features = False
    else:
        print('argument error')

    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print("Test prompt id is {} of type {}".format(test_prompt_id, type(test_prompt_id)))
    print("Attribute: {}".format(attribute_name))
    print("Seed: {}".format(seed))

    configs = Configs()

    data_path = configs.DATA_PATH
    train_path = data_path + str(test_prompt_id) + '/train.pk'
    dev_path = data_path + str(test_prompt_id) + '/dev.pk'
    test_path = data_path + str(test_prompt_id) + '/test.pk'
    features_path = configs.FEATURES_PATH
    pretrained_embedding = configs.PRETRAINED_EMBEDDING
    embedding_path = configs.EMBEDDING_PATH
    embedding_dim = configs.EMBEDDING_DIM
    readability_path = configs.READABILITY_PATH
    vocab_size = 8000
    epochs = 100
    batch_size = 32

    read_configs = {
        'train_path': train_path,
        'dev_path': dev_path,
        'test_path': test_path,
        'features_path': features_path,
        'readability_path': readability_path,
        'vocab_size': vocab_size
    }

    if input_seq == 'words':
        word_vocab = read_word_vocab(read_configs)
        print('words vocab size is {}'.format(len(word_vocab)))
        source_train_data, source_dev_data, target_data = read_essays_single_score_words(read_configs, word_vocab, attribute_name)
    elif input_seq == 'pos':
        pos_vocab = read_pos_vocab(read_configs)
        print('pos vocab size is {}'.format(len(pos_vocab)))
        source_train_data, source_dev_data, target_data = read_essays_single_score(read_configs, pos_vocab, attribute_name)

    if input_seq == 'words':
        embedd_dict, embedd_dim, _ = load_word_embedding_dict(embedding_path)
        embedd_matrix = build_embedd_table(word_vocab, embedd_dict, embedd_dim, caseless=True)
        embed_table = [embedd_matrix]
    elif input_seq == 'pos':
        embed_table = None

    max_sentlen = max(source_train_data['max_sentlen'], source_dev_data['max_sentlen'], target_data['max_sentlen'])
    max_sentnum = max(source_train_data['max_sentnum'], source_dev_data['max_sentnum'], target_data['max_sentnum'])
    print('max sent length: {}'.format(max_sentlen))
    print('max sent num: {}'.format(max_sentnum))
    source_train_data['y_scaled'] = get_single_scaled_down_score(source_train_data['data_y'], source_train_data['prompt_ids'], attribute_name)
    source_dev_data['y_scaled'] = get_single_scaled_down_score(source_dev_data['data_y'], source_dev_data['prompt_ids'], attribute_name)
    target_data['y_scaled'] = get_single_scaled_down_score(target_data['data_y'], target_data['prompt_ids'], attribute_name)

    if input_seq == 'words':
        X_train = pad_hierarchical_text_sequences(source_train_data['words'], max_sentnum, max_sentlen)
        X_dev = pad_hierarchical_text_sequences(source_dev_data['words'], max_sentnum, max_sentlen)
        X_test = pad_hierarchical_text_sequences(target_data['words'], max_sentnum, max_sentlen)
    elif input_seq == 'pos':
        X_train = pad_hierarchical_text_sequences(source_train_data['pos_x'], max_sentnum, max_sentlen)
        X_dev = pad_hierarchical_text_sequences(source_dev_data['pos_x'], max_sentnum, max_sentlen)
        X_test = pad_hierarchical_text_sequences(target_data['pos_x'], max_sentnum, max_sentlen)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_dev = X_dev.reshape((X_dev.shape[0], X_dev.shape[1] * X_dev.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

    X_train_linguistic_features = np.array(source_train_data['features_x'])
    X_dev_linguistic_features = np.array(source_dev_data['features_x'])
    X_test_linguistic_features = np.array(target_data['features_x'])

    X_train_readability = np.array(source_train_data['readability_x'])
    X_dev_readability = np.array(source_dev_data['readability_x'])
    X_test_readability = np.array(target_data['readability_x'])

    Y_train = np.array(source_train_data['y_scaled'])
    Y_dev = np.array(source_dev_data['y_scaled'])
    Y_test = np.array(target_data['y_scaled'])

    id_train = np.array(source_train_data['prompt_ids'])
    id_dev = np.array(source_dev_data['prompt_ids'])
    id_test = np.array(target_data['prompt_ids'])

    print('================================')
    print('X_train: ', X_train.shape)
    print('X_train_readability: ', X_train_readability.shape)
    print('X_train_ling: ', X_train_linguistic_features.shape)
    print('Y_train: ', Y_train.shape)

    print('================================')
    print('X_dev: ', X_dev.shape)
    print('X_dev_readability: ', X_dev_readability.shape)
    print('X_dev_ling: ', X_dev_linguistic_features.shape)
    print('Y_dev: ', Y_dev.shape)

    print('================================')
    print('X_test: ', X_test.shape)
    print('X_test_readability: ', X_test_readability.shape)
    print('X_test_ling: ', X_test_linguistic_features.shape)
    print('Y_test: ', Y_test.shape)
    print('================================')

    count = 0
    X_test_rep = X_test
    Y_test_rep = Y_test
    while X_train.shape[0] != X_test_rep.shape[0]:
        X_test_rep = np.vstack([X_test_rep, X_test[count%X_test.shape[0]]])
        Y_test_rep = np.vstack([Y_test_rep, Y_test[count%Y_test.shape[0]]])

    print('================================')
    print('X_test_rep: ', X_test_rep.shape)
    print('Y_test_rep: ', Y_test_rep.shape)
    print('================================')

    assert(X_train.shape[0] == X_test_rep.shape[0])
    source_dataset = tf.data.Dataset.from_tensor_slices((X_train.astype('float32'),
                                                         Y_train.astype('float32'),
                                                         X_train_linguistic_features.astype('float32'),
                                                         X_train_readability.astype('float32'),
                                                         id_train)).shuffle(1000).batch(batch_size*2)
    da_dataset = tf.data.Dataset.from_tensor_slices((X_train.astype('float32'),
                                                     Y_train.astype('float32'),
                                                     X_train_linguistic_features.astype('float32'),
                                                     X_train_readability.astype('float32'),
                                                     X_test_rep.astype('float32'),
                                                     Y_test_rep.astype('float32'),
                                                     id_train)).shuffle(1000).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_dev.astype('float32'),
                                                       Y_dev.astype('float32'),
                                                       X_dev_linguistic_features.astype('float32'),
                                                       X_dev_readability.astype('float32'),
                                                       id_dev)).shuffle(1000).batch(batch_size*2) #Test Dataset over Target Domain
    test_dataset2 = tf.data.Dataset.from_tensor_slices((X_test.astype('float32'),
                                                        Y_test.astype('float32'),
                                                        X_test_linguistic_features.astype('float32'),
                                                        X_test_readability.astype('float32'),
                                                        id_test)).shuffle(1000).batch(batch_size*2) #Test Dataset over Target (used for training)

    if input_seq == 'words':
        model = Smodel(configs, max_sentnum, max_sentlen, len(word_vocab), embed_table, with_features)
    elif input_seq == 'pos':
        model = Smodel(configs, max_sentnum, max_sentlen, len(pos_vocab), embed_table, with_features)

    def CCE_loss_func(input_logits, target_labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=input_logits, labels=target_labels))
    
    def MSE_loss_func(input_value, target_labels):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(input_value, target_labels)

    def get_loss(l_logits, labels, d_logits=None, domain=None):
        if d_logits is None:
            return MSE_loss_func(l_logits, labels)
        else:
            return MSE_loss_func(l_logits, labels) + CCE_loss_func(d_logits, domain)

    #model_optimizer = tf.optimizers.SGD()
    model_optimizer = tf.optimizers.Adam(lr=0.0001)
    #model_optimizer = tf.optimizers.RMSprop()


    domain_labels = np.vstack([np.tile([1., 0.], [batch_size, 1]),
                               np.tile([0., 1.], [batch_size, 1])])
    domain_labels = domain_labels.astype('float32')

    epoch_acc = tf.keras.metrics.CategoricalAccuracy()
    epoch_qwk = KappaScore('quadratic')
    epoch_lwk = KappaScore('linear')
    epoch_corr = CorrelationCoefficient()
    epoch_rmse = tf.keras.metrics.RootMeanSquaredError()
    epoch_mae = tf.keras.metrics.MeanAbsoluteError()
    epoch_metrics = [epoch_qwk, epoch_lwk, epoch_corr, epoch_rmse, epoch_mae]

    source_metrics = {'qwk': [], 'lwk': [], 'corr': [], 'rmse': [], 'mae': []}  # Source Domain Accuracy while Source-only Training
    da_metrics = {'qwk': [], 'lwk': [], 'corr': [], 'rmse': [], 'mae': []}      # Source Domain Accuracy while DA-training
    test_metrics = {'qwk': [], 'lwk': [], 'corr': [], 'rmse': [], 'mae': []}    # Testing Dataset (Target Domain) Accuracy 
    test2_metrics = {'qwk': [], 'lwk': [], 'corr': [], 'rmse': [], 'mae': []}   # Target Domain (used for Training) Accuracy

    @tf.function
    def train_step_source(s_essays, s_labels, s_lin, s_read, s_ids, lamda=1.0):
        essays = s_essays
        labels = s_labels
        linguistic_features = s_lin
        readability_features = s_read
        ids = s_ids
        
        with tf.GradientTape() as tape:
            output = model(essays, linguistic_features, readability_features, train=True, source_train=True, lamda=lamda)
            
            model_loss = get_loss(output, labels)
            for i, epoch_metric in enumerate(epoch_metrics):
                if i in [0, 1]:
                    epoch_metric.update_state(output, labels, ids)
                else:
                    epoch_metric.update_state(output, labels)
            
        gradients_mdan = tape.gradient(model_loss, model.trainable_variables)
        model_optimizer.apply_gradients(zip(gradients_mdan, model.trainable_variables))
    
    @tf.function
    def train_step_da(s_essays, s_labels, s_lin, s_read, t_essays=None, t_labels=None, s_ids=None, lamda=1.0):
        essays = tf.concat([s_essays, t_essays], 0)
        labels = s_labels
        ids = s_ids
        linguistic_features = s_lin
        readability_features = s_read
        
        with tf.GradientTape() as tape:
            output = model(essays, linguistic_features, readability_features, train=True, source_train=False, lamda=lamda)
            
            try:
                model_loss = get_loss(output[0], labels, output[1], domain_labels)
                for i, epoch_metric in enumerate(epoch_metrics):
                    if i in [0, 1]:
                        epoch_metric.update_state(output[0], labels, ids)
                    else:
                        epoch_metric.update_state(output[0], labels)
            except: # ミニバッチのサイズが32でない場合
                domain_labels = np.vstack([np.tile([1., 0.], [output[0].shape[0], 1]),
                                           np.tile([0., 1.], [output[0].shape[0], 1])])
                domain_labels = domain_labels.astype('float32')
                model_loss = get_loss(output[0], labels, output[1], domain_labels)
                for i, epoch_metric in enumerate(epoch_metrics):
                    if i in [0, 1]:
                        epoch_metric.update_state(output[0], labels, ids)
                    else:
                        epoch_metric.update_state(output[0], labels)
            
        gradients_mdan = tape.gradient(model_loss, model.trainable_variables)
        model_optimizer.apply_gradients(zip(gradients_mdan, model.trainable_variables))

    @tf.function
    def test_step(t_essays, t_labels, t_lin, t_read, t_ids):
        essays = t_essays
        labels = t_labels
        ids = t_ids
        linguistic_features = t_lin
        readability_features = t_read
        
        output = model(essays, linguistic_features, readability_features, train=False, source_train=True)
        for i, epoch_metric in enumerate(epoch_metrics):
                if i in [0, 1]:
                    epoch_metric.update_state(output, labels, ids)
                else:
                    epoch_metric.update_state(output, labels)

    def train(train_mode, epochs=epochs):
    
        if train_mode == 'source':
            dataset = source_dataset
            train_func = train_step_source
            metrics_dict = source_metrics
        elif train_mode == 'domain-adaptation':
            dataset = da_dataset
            train_func = train_step_da
            metrics_dict = da_metrics
        else:
            raise ValueError("Unknown training Mode")
        
        for epoch in range(epochs):
            # p = float(epoch) / epochs
            # lamda = 2 / (1 + np.exp(-100 * p, dtype=np.float32)) - 1
            # lamda = lamda.astype('float32')

            for batch in dataset:
                train_func(*batch, lamda=0.1)
            
            print("Training: {} / {} EPOCHS".format(epoch, epochs))
            for key, epoch_metric in zip(source_metrics.keys(), epoch_metrics):
                print('[Source] {} : {:.3f}'.format(key.upper(), epoch_metric.result()))
                metrics_dict[key].append(epoch_metric.result())
            test()
            for epoch_metric in epoch_metrics:
                epoch_metric.reset_states()

    def test():
        for epoch_metric in epoch_metrics:
            epoch_metric.reset_states()
        
        #Testing Dataset (Target Domain)
        for batch in test_dataset:
            test_step(*batch)
        
        for key, epoch_metric in zip(source_metrics.keys(), epoch_metrics):
            print('[DEV] {} : {:.3f}'.format(key.upper(), epoch_metric.result()))
            test_metrics[key].append(epoch_metric.result())
            epoch_metric.reset_states()
        
        #Target Domain (used for Training)
        for batch in test_dataset2:
            test_step(*batch)
        
        for key, epoch_metric in zip(source_metrics.keys(), epoch_metrics):
            print('[Target] {} : {:.3f}'.format(key.upper(), epoch_metric.result()))
            test2_metrics[key].append(epoch_metric.result())
            epoch_metric.reset_states()
        print('-'*100)

    #Training
    train(train_mode, epochs)

    #Plot Results
    for key in da_metrics.keys():
        fig = plt.figure()
        x_axis = [i for i in range(0, epochs)]
        if train_mode == 'source':
            plt.plot(x_axis, source_metrics[key], label="source {}".format(key))
        elif train_mode == 'domain-adaptation':
            plt.plot(x_axis, da_metrics[key], label="source_da {}".format(key))
        plt.plot(x_axis, test_metrics[key], label="dev {}".format(key))
        plt.plot(x_axis, test2_metrics[key], label="target {}".format(key))
        plt.legend()
        plt.savefig('img/{}/{}{}{}.png'.format(model_name, train_mode, test_prompt_id, key))
        #plt.show()
        

if __name__ == '__main__':
    main()

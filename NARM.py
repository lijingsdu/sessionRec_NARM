'''
Build NARM model
'''

from __future__ import print_function
import pickle

from collections import OrderedDict
import sys
import time

import numpy as np
import theano
from theano import config
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import data_process

datasets = {'rsc2015': (data_process.load_data, data_process.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 42
np.random.seed(SEED)


def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng, drop_p=0.5):
    retain = 1. - drop_p
    proj = T.switch(use_noise, (state_before * trng.binomial(state_before.shape,
                                                             p=retain, n=1,
                                                             dtype=state_before.dtype)), state_before * retain)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not GRU) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    params['Wemb'] = init_weights((options['n_items'], options['dim_proj']))
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # attention
    params['W_encoder'] = init_weights((options['hidden_units'], options['hidden_units']))
    params['W_decoder'] = init_weights((options['hidden_units'], options['hidden_units']))
    params['bl_vector'] = init_weights((1, options['hidden_units']))
    # classifier
    # params['U'] = init_weights((2*options['hidden_units'], options['n_items']))
    # params['b'] = np.zeros((options['n_items'],)).astype(config.floatX)
    params['bili'] = init_weights((options['dim_proj'], 2 * options['hidden_units']))

    return params


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def init_weights(shape):
    sigma = np.sqrt(2. / shape[0])
    return numpy_floatX(np.random.randn(*shape) * sigma)


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_gru(options, params, prefix='gru'):
    """
    Init the GRU parameter:

    :see: init_params
    """
    Wxrz = np.concatenate([init_weights((options['dim_proj'], options['hidden_units'])),
                           init_weights((options['dim_proj'], options['hidden_units'])),
                           init_weights((options['dim_proj'], options['hidden_units']))], axis=1)
    params[_p(prefix, 'Wxrz')] = Wxrz

    Urz = np.concatenate([ortho_weight(options['hidden_units']),
                          ortho_weight(options['hidden_units'])], axis=1)
    params[_p(prefix, 'Urz')] = Urz

    Uh = ortho_weight(options['hidden_units'])
    params[_p(prefix, 'Uh')] = Uh

    b = np.zeros((3 * options['hidden_units'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)
    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_):
        preact = T.dot(h_, tparams[_p(prefix, 'Urz')])
        preact += x_[:, 0:2 * options['hidden_units']]

        z = T.nnet.hard_sigmoid(_slice(preact, 0, options['hidden_units']))
        r = T.nnet.hard_sigmoid(_slice(preact, 1, options['hidden_units']))
        h = T.tanh(T.dot((h_ * r), tparams[_p(prefix, 'Uh')]) + _slice(x_, 2, options['hidden_units']))

        h = (1.0 - z) * h_ + z * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    state_below = (T.dot(state_below, tparams[_p(prefix, 'Wxrz')]) +
                   tparams[_p(prefix, 'b')])

    hidden_units = options['hidden_units']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=T.alloc(numpy_floatX(0.), n_samples, hidden_units),
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval

layers = {'gru': (param_init_gru, gru_layer)}


def adam(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]

    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    """

    updates = OrderedDict()
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g  # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2   # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)               # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)               # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

        updates[m_previous] = m
        updates[v_previous] = v
        updates[theta_previous] = theta
    updates[t] = t + 1.

    return updates


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = T.matrix('x', dtype='int64')
    mask = T.matrix('mask', dtype=config.floatX)
    y = T.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    if options['use_dropout']:
        emb = dropout_layer(emb, use_noise, trng, drop_p=0.25)

    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)

    def compute_alpha(state1, state2):
        tmp = T.nnet.hard_sigmoid(T.dot(tparams['W_encoder'], state1.T) + T.dot(tparams['W_decoder'], state2.T))
        alpha = T.dot(tparams['bl_vector'], tmp)
        res = T.sum(alpha, axis=0)
        return res

    last_h = proj[-1]

    sim_matrix, _ = theano.scan(
        fn=compute_alpha,
        sequences=proj,
        non_sequences=proj[-1]
    )
    att = T.nnet.softmax(sim_matrix.T * mask.T) * mask.T
    p = att.sum(axis=1)[:, None]
    weight = att / p
    atttention_proj = (proj * weight.T[:, :, None]).sum(axis=0)

    proj = T.concatenate([atttention_proj, last_h], axis=1)

    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng, drop_p=0.5)

    ytem = T.dot(tparams['Wemb'], tparams['bili'])
    pred = T.nnet.softmax(T.dot(proj, ytem.T))
    # pred = T.nnet.softmax(T.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    # f_weight = theano.function([x, mask], weight, name='f_weight')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -T.log(pred[T.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, cost


def pred_evaluation(f_pred_prob, prepare_data, data, iterator):
    """
    Compute recall@20 and mrr@20
    f_pred_prob: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    recall = 0.0
    mrr = 0.0
    evalutation_point_count = 0
    # pred_res = []
    # att = []

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index])
        preds = f_pred_prob(x, mask)
        # weights = f_weight(x, mask)
        targets = y
        ranks = (preds.T > np.diag(preds.T[targets])).sum(axis=0) + 1
        rank_ok = (ranks <= 20)
        # pred_res += list(rank_ok)
        recall += rank_ok.sum()
        mrr += (1.0 / ranks[rank_ok]).sum()
        evalutation_point_count += len(ranks)
        # att.append(weights)

    recall = numpy_floatX(recall) / evalutation_point_count
    mrr = numpy_floatX(mrr) / evalutation_point_count
    eval_score = (recall, mrr)

    # ff = open('/storage/lijing/mydataset/res_attention_correct.pkl', 'wb')
    # pickle.dump(pred_res, ff)
    # ff.close()
    # ff2 = open('/storage/lijing/mydataset/attention_weights.pkl', 'wb')
    # pickle.dump(att, ff2)
    # ff2.close()

    return eval_score


def train_gru(
    dim_proj=50,  # word embeding dimension
    hidden_units=100,  # GRU number of hidden units.
    patience=100,  # Number of epoch to wait before early stop if no progress
    max_epochs=30,  # The maximum number of epoch to run
    dispFreq=100,  # Display to stdout the training progress every N updates
    lrate=0.001,  # Learning rate
    n_items=37484,  # Vocabulary size
    encoder='gru',  # TODO: can be removed must be gru.
    saveto='gru_model.npz',  # The best model will be saved there
    is_valid=True,  # Compute the validation error after this number of update.
    is_save=False,  # Save the parameters after every saveFreq updates
    batch_size=512,  # The batch size during training.
    valid_batch_size=512,  # The batch size used for validation/test set.
    dataset='rsc2015',

    # Parameter for extra option
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    load_data, prepare_data = get_dataset(dataset)

    print('Loading data')
    train, valid, test = load_data()

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('gru_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask,
     y, f_pred_prob, cost) = build_model(tparams, model_options)

    all_params = list(tparams.values())

    updates = adam(cost, all_params, lrate)

    train_function = theano.function(inputs=[x, mask, y], outputs=cost, updates=updates)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    history_vali = []
    best_p = None
    bad_count = 0

    uidx = 0  # the number of update done
    estop = False  # early stop

    try:
        for eidx in range(max_epochs):
            start_time = time.time()
            n_samples = 0
            epoch_loss = []

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                loss = train_function(x, mask, y)
                epoch_loss.append(loss)

                if np.isnan(loss) or np.isinf(loss):
                    print('bad loss detected: ', loss)
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Loss ', np.mean(epoch_loss))

            if saveto and is_save:
                print('Saving...')

                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                np.savez(saveto, history_errs=history_errs, **params)
                print('Saving done')

            if is_valid:
                use_noise.set_value(0.)

                valid_evaluation = pred_evaluation(f_pred_prob, prepare_data, valid, kf_valid)
                test_evaluation = pred_evaluation(f_pred_prob, prepare_data, test, kf_test)
                history_errs.append([valid_evaluation, test_evaluation])

                if best_p is None or valid_evaluation[0] >= np.array(history_vali).max():

                    best_p = unzip(tparams)
                    print('Best perfomance updated!')
                    bad_count = 0

                print('Valid Recall@20:', valid_evaluation[0], '   Valid Mrr@20:', valid_evaluation[1],
                      '\nTest Recall@20', test_evaluation[0], '   Test Mrr@20:', test_evaluation[1])

                if len(history_vali) > 10 and valid_evaluation[0] <= np.array(history_vali).max():
                    bad_count += 1
                    print('===========================>Bad counter: ' + str(bad_count))
                    print('current validation recall: ' + str(valid_evaluation[0]) +
                          '      history max recall:' + str(np.array(history_vali).max()))
                    if bad_count > patience:
                        print('Early Stop!')
                        estop = True

                history_vali.append(valid_evaluation[0])

            end_time = time.time()
            print('Seen %d samples' % n_samples)
            print(('This epoch took %.1fs' % (end_time - start_time)), file=sys.stderr)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    valid_evaluation = pred_evaluation(f_pred_prob, prepare_data, valid, kf_valid)
    test_evaluation = pred_evaluation(f_pred_prob,  prepare_data, test, kf_test)

    print('=================Best performance=================')
    print('Valid Recall@20:', valid_evaluation[0], '   Valid Mrr@20:', valid_evaluation[1],
          '\nTest Recall@20', test_evaluation[0], '   Test Mrr@20:', test_evaluation[1])
    print('==================================================')
    if saveto and is_save:
        np.savez('Best_performance', valid_evaluation=valid_evaluation, test_evaluation=test_evaluation, history_errs=history_errs,
                 **best_p)

    return valid_evaluation, test_evaluation


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    eval_valid, eval_test = train_gru(max_epochs=30, test_size=-1)

import os, sys, json, argparse
import numpy as np
import tensorflow as tf
import lstm_inference_dp, random
from data_reader import Data
import dp_optimizer
import sanitizer
import utils
import accountant

from rdp_accountant import compute_rdp
from rdp_accountant import get_privacy_spent


with open('config.json') as config_file:
    config = json.load(config_file)
DATA_FD = config["data_folder"]
RESULT_FD = config["result_folder"]
model_folder = config["model_folder"]
max_train_epochs = config['max_train_epochs']
batch_size = config['batch_size']
dataType = config['data_type']


tf.flags.DEFINE_boolean('dpsgd', False, 'If True, train with DP-SGD. If False, '
                        'train with vanilla SGD.')
tf.flags.DEFINE_float('learning_rate', .001, 'Learning rate for training')
tf.flags.DEFINE_float('target_delta', 1e-5, 'fixed delta')
tf.flags.DEFINE_float('noise_multiplier', 0.5, #0.001,
                      'Ratio of the standard deviation to the clipping norm')
tf.flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
tf.flags.DEFINE_integer('epochs', 100, 'Number of epochs')

FLAGS = tf.flags.FLAGS




def create_dirs(dirs):
    for dd in dirs:
        if not os.path.exists(dd):
            os.mkdir(dd)
            os.chdir(dd)

subfd = 'non_dp'
if FLAGS.dpsgd:
    subfd = 'dp_clip%s_delta%s_sigma%s_clip%s'%(FLAGS.l2_norm_clip, str(FLAGS.target_delta), str(FLAGS.noise_multiplier), str(FLAGS.l2_norm_clip))
save_dir = os.path.join(model_folder, dataType, subfd)

create_dirs([model_folder, dataType, subfd])


DATA = Data( bTrainAdv=0, dataType=dataType)
num_examples = len(DATA.x_train)
print('before, num_examples', num_examples)
microbatches = num_examples//batch_size
num_examples = batch_size * microbatches
print('after, num_examples', num_examples)

print('data.y_vali.shape', np.asarray(DATA.y_valid).shape)

def print_loss_and_accuracy(global_loss,accuracy):
    print(' - Current Model has a loss of:           %s' % global_loss)
    print(' - The Accuracy on the validation set is: %s' % accuracy)
    print('--------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------')


def validation(loss, eval_correct):
    # validation  (using test dataset)  	
    feed_dict = {str(data_placeholder.name): np.asarray(DATA.x_valid),
                 str(label_placeholder.name): np.asarray(DATA.y_valid)}

    global_loss = sess.run(loss, feed_dict=feed_dict)
    count = sess.run(eval_correct, feed_dict=feed_dict)
    accuracy = float(count) / float(len(DATA.y_valid))
    # preds = sess.run(pred, feed_dict=feed_dict)
    # print('validation probs', preds)  # actual probs
    print_loss_and_accuracy(global_loss, accuracy)
    return global_loss, accuracy

def compute_epsilon(steps):
  print('in compute_epsilon, steps', steps)
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = (batch_size*1.0/config['hist_len']) / num_examples
  print('sampling_probability', sampling_probability)
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=FLAGS.noise_multiplier,
                    steps=steps,
                    orders=orders)
  # Delta is set to 1e-5 because Penn TreeBank has 60000 training points.
  return get_privacy_spent(orders, rdp, target_delta=FLAGS.target_delta)[0]



with tf.Graph().as_default():

    train_op, eval_correct, loss, data_placeholder, label_placeholder, pred = None, None, None, None, None, None

    if FLAGS.dpsgd:
        priv_accountant = accountant.GaussianMomentsAccountant(num_examples)
        gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(priv_accountant, [FLAGS.l2_norm_clip / batch_size, True])
        # for var in training_params:
        #     if "gradient_l2norm_bound" in training_params[var]:
        #         l2bound = training_params[var]["gradient_l2norm_bound"] / batch_size
        #         gaussian_sanitizer.set_option(var, sanitizer.ClipOption(l2bound, True))

        train_op, eval_correct, loss, data_placeholder, label_placeholder, pred = \
            lstm_inference_dp.lstm_model(oneHot=True, bTrain=True, num_classes=DATA.NUM_CLASSES, num_hidden=config['NUM_LAYERS'], learning_rate=FLAGS.learning_rate, \
                                            dpsgd=True, l2_norm_clip=FLAGS.l2_norm_clip, noise_multiplier=FLAGS.noise_multiplier, \
                                            microbatches=microbatches, num_examples=num_examples)
    else:
        train_op, eval_correct, loss, data_placeholder, label_placeholder, pred = \
            lstm_inference_dp.lstm_model(oneHot=True, bTrain=True, num_classes=DATA.NUM_CLASSES, num_hidden=config['NUM_LAYERS'], learning_rate=FLAGS.learning_rate)


    # Usual Tensorflow...
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(tf.all_variables(), reshape=True,  max_to_keep=None)
    batch_len = batch_size #* config['hist_len']

    eps = 0
    # Training loop.
    steps_per_epoch = num_examples // batch_len
    for epoch in range(1, FLAGS.epochs + 1):
        print('epoch', epoch, 'num_examples', num_examples, 'batch_len', batch_len, 'steps_per_epoch', steps_per_epoch)
        # shuffle training data for each round
        idxs = range(num_examples)
        random.shuffle(idxs) # return None
        idxs = idxs[:num_examples]
        data_ind = np.array_split(idxs, batch_len, 0) # split into batches

        global_loss, accuracy = validation(loss, eval_correct)

# save
        if accuracy>0.5:
            fn = "epoch%d_validLoss%.3f_validAcc%.2f_eps%s"% (epoch, global_loss, accuracy*100, str(eps))
            print('saving: ', fn)
            saver.save(sess, os.path.join(save_dir, fn))
        for step in xrange(len(data_ind)):
            batch_ind = data_ind[step]
            feed_dict = {str(data_placeholder.name): [DATA.x_train[int(j)] for j in batch_ind],
                         str(label_placeholder.name): [DATA.y_train[int(j)] for j in batch_ind]}

            _ = sess.run([train_op], feed_dict=feed_dict)


            # train acc
            global_loss = sess.run(loss, feed_dict=feed_dict)
            count = sess.run(eval_correct, feed_dict=feed_dict)
            accuracy = float(count) / float(len(batch_ind))
            if step%100==0: 
                print('step %d in epoch %d, loss: %f, accuracy: %f'% (step, epoch, global_loss, accuracy))

        # Compute the privacy budget expended so far.
        if FLAGS.dpsgd:
            eps = compute_epsilon(epoch * steps_per_epoch)
            print('For delta=%s, the current epsilon is: %.2f' %(FLAGS.target_delta, eps))
        else:
            print('Trained with vanilla non-private SGD optimizer')

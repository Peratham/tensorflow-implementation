import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
from scipy import ndimage
from coco_utils import decode_captions, sample_coco_minibatch


class CaptioningSolver(object):
    """
    CaptioningSolver produces functions:
        - train: trains model and prints loss and useful informations for debugging.  
        - test: generates(or samples) caption and visualizes alpha weights with image.
        Example usage might look something like this:
        data = load_coco_data()
        model = CaptionGenerator(word_to_idx, batch_size= 100, dim_feature=[196, 512], dim_embed=128,
                                   dim_hidden=128, n_time_step=16, cell_type='lstm', dtype=tf.float32)
        solver = CaptioningSolver(model, data, n_epochs=10, batch_size=100, update_rule='adam', 
                                                learning_rate=0.03, print_every=10, save_every=10)
        solver.train()
        solver.test()
    """
    def __init__(self, model, data, **kwargs):
        """
        Required Arguments:
        - model: a caption generator model with following functions:
            - build_model: receives features and captions then build graph where root nodes are loss and logits.  
            - build_sampler: receives features and build graph where root nodes are captions and alpha weights.
        - data: dictionary with the following keys:
            - train_features: feature vectors for train
            - train_captions: captions for train
            - train_image_idxs: captions to image mapping
            - word_to_idx: word to index dictionary
        Optional Arguments:
        - n_epochs: the number of epochs to run for during training.
        - batch_size: mini batch size.
        - update_rule: a string giving the name of an update rule among the followings: 
            - 'adam'
            - 'rmsprop'
            - 'adadelta'
            - 'adagrad'
        - learning_rate: learning rate; default value is 0.03.
        - print_every: Integer; training losses will be printed every print_every iterations.
        - save_every: Integer; model variables will be saved every save_every iterations.
        """
        self.model = model
        self.data = data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.03)
        self.print_every = kwargs.pop('print_every', 10)
        self.save_every = kwargs.pop('save_every', 100)
        self.model_path = kwargs.pop('model_path', './model/')
        self.test_model = kwargs.pop('test_model', './model/model-200')

        # Book-keeping variables 
        #self.best_val_acc = 0
        #self.best_params = {}
        self.loss_history = []
        #self.train_acc_history = []
        #self.val_acc_history = []

        # Set optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer
        elif self.update_rule == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer
        else:
            self.optimizer = tf.train.RMSPropOptimizer



    def train(self):
        """
        Train model and print out some useful information(loss, generated captions) for debugging.  
        """
        n_examples = self.data['train_captions'].shape[0]
        n_iters_per_epoch = n_examples // self.batch_size

        # get data
        features = self.data['train_features']
        captions = self.data['train_captions']

        # build train model graph
        loss, generated_captions = self.model.build_model()
        optimizer = self.optimizer(self.learning_rate).minimize(loss)

        # build test model graph
        alphas, sampled_captions = self.model.build_sampler()    # (N, max_len, L), (N, max_len)
        
        print "num epochs: %d" %self.n_epochs
        print "iterations per epoch: %d" %n_iters_per_epoch
        print "data size: %d" %n_examples
        print "batch size: %d" %self.batch_size

        sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(max_to_keep=10)
        
        for e in range(self.n_epochs):
            # print initial loss
            if e == 0:
                captions_batch, features_batch, _ = sample_coco_minibatch(self.data, self.batch_size, split='train')
                feed_dict = { self.model.features: features_batch, self.model.captions: captions_batch }
                gen_caps, l = sess.run([generated_captions, loss], feed_dict)
                self.loss_history.append(l)
                print ""
                print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
                print "Initial Train Loss: %.5f" %l
                decoded = decode_captions(gen_caps, self.model.idx_to_word)
                for j in range(3):
                    print "Generated Caption: %s" %decoded[j]
                print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
                print ""

            # actual training step
            for i in range(n_iters_per_epoch):
                captions_batch, features_batch, _ = sample_coco_minibatch(self.data, self.batch_size, split='train')
                feed_dict = { self.model.features: features_batch, self.model.captions: captions_batch }
                sess.run(optimizer, feed_dict)

            # save loss history
            l = sess.run(loss, feed_dict)
            self.loss_history.append(l)

            # print info
            if (e+1) % self.print_every == 0:
                gen_caps = sess.run(generated_captions, feed_dict)
                print ""
                print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
                print "Train Loss at Epoch %d: %.5f" %(e+1, l)
                decoded = decode_captions(gen_caps, self.model.idx_to_word)
                for j in range(3):
                    print "Generated Caption: %s" %decoded[j]
                print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
                print ""

            # save model
            if (e+1) % self.save_every == 0:
                saver.save(sess, self.model_path+'model', global_step=e+1)
                print "model-%s saved." %(e+1)


        # actual test step: sample captions and visualize attention
        _, features_batch, image_files = sample_coco_minibatch(self.data, self.batch_size, split='train')
        feed_dict = { self.model.features: features_batch}
        alps, sam_cap = sess.run([alphas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)

        # decode captions
        decoded = decode_captions(sam_cap, self.model.idx_to_word)

        # visualize 10 images and captions 
        for n in range(10):
            print "Sampled Caption: %s" %decoded[n]

            # plot original image
            img_path = './train2014_resized/'+ image_files[n]
            img = ndimage.imread(img_path)
            plt.subplot(4, 5, 1)
            plt.imshow(img)
            plt.axis('off')

            # plot image with attention weights
            words = decoded[n].split(" ")
            for t in range(len(words)):
                if t>18:
                    break
                plt.subplot(4, 5, t+2)
                plt.text(0, 1, words[t], color='black', backgroundcolor='white', fontsize=12)
                plt.imshow(img)
                alp_curr = alps[n,t,:].reshape(14,14)
                alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                plt.imshow(alp_img, alpha=0.8)
                plt.axis('off')
            plt.show()

    
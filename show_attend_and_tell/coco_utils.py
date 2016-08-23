import numpy as np
import cPickle as pickle

def load_coco_data(file):
    
    with open(file, 'rb') as f:
        data = pickle.load(f)
    for k, v in data.iteritems():
        if type(v) == np.ndarray:
            print k, type(v), v.shape, v.dtype
        else:
            print k, type(v), len(v)
    return data


def decode_captions(captions, idx_to_word):
    """
    Inputs:
    - captions: numpy ndarray where the value(word index) is in the range [0, V) of shape (N, T)
    - idx_to_word: index to word mapping dictionary

    Returns:
    - decoded: decoded captions of shape (N, )
    """
    N, T = captions.shape
    decoded = []

    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
        decoded.append(' '.join(words))
    return decoded

def sample_coco_minibatch(data, batch_size, split='train'):
    """
    Inputs: 
    - data: dictionary with following keys:
        - train_features: ndarray of shape (82783, 196, 512), not yet completed
        - train_image_filename: list of length 82783
        - train_captions: ndarray of shape (410000, 17)
        - train_image_idxs: ndarray of shape (410000,)
        - val_features (will be added)
        - val_image_filename (will be added)
        - val_captions (will be added)
        - val_image_idxs (will be added)
    - batch_size: batch size
    - split: train or val
    """

    data_size = data['%s_captions' %split].shape[0]
    mask = np.random.choice(data_size, batch_size)
    captions = data['%s_captions' %split][mask]
    image_idxs = data['%s_image_idxs' %split][mask]
    features = data['%s_features' %split][image_idxs]
    image_files = data['%s_image_filename' %split][image_idxs]

    return captions, features, image_files

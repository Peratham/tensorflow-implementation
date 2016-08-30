import numpy as np


def get_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)

    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]

    return image_idxs

def get_file_names(annotations):

    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['file_name']

    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.array(image_file_names)    # conver list to numpy ndarray
    return file_names, id_to_idx

def preprocess_annotations(annotations):
    '''
    Delete '.', ',', '"' and "'" for each captions and remove long captions. 

    Inputs:
        annotations: uncleaned annotations
    Return:
        annotations: cleaned annotations
    '''
    del_idx = []
    j = 0
    for i in range(len(annotations)):
        sentence = annotations['caption'][i]
        sentence = sentence.replace('.','').replace(',','').replace("'","").replace('"','')
        annotations.set_value(i, 'caption', sentence)

        if len(sentence.split(" ")) > 15:
            del_idx.append(i)

    # delete captions if size is larger than 15
    print "the number of captions before deleting: %d" %len(annotations)
    annotations = annotations.drop(annotations.index[del_idx])
    annotations = annotations.reset_index(drop=True)
    print "the number of captions after deleting: %d" %len(annotations)

    return annotations

def build_caption_vectors(annotations, word_to_idx, max_len=15):
    '''
    Inputs:
        annotations: annotations
        word_to_idx: word to index dictionary
        max_len: max length of captions
    Returns:
        captions: caption vectors
    '''
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_len+2)).astype(np.int32)   # two for <'START'> , <'END'> tokens

    for i, annotation in enumerate(annotations['caption'][:n_examples]):
        sentence = annotation
        words = sentence.lower().split(' ')
        n_words = len(words)
        for j in range(max_len+2):
            if j == 0:
                captions[i,j] = word_to_idx['<START>']
            elif j <= n_words:
                if words[j-1] in word_to_idx:
                    captions[i,j] = word_to_idx[words[j-1]]
                else:
                    captions[i,j] = word_to_idx['<UNK>']
            elif j == n_words+1:
                captions[i,j] = word_to_idx['<END>']
            else:
                captions[i,j] = word_to_idx['<NULL>']            

    print "success building caption vectors: ", captions.shape
    return captions


def build_word_to_idx(sentences, threshold=100): 
    word_counts = {}
    n_sents = 0
    max_len = 0
    for i, sentence in enumerate(sentences):
        n_sents += 1
        
        if len(sentence.split(" ")) > max_len:
            max_len = len(sentence.split(" "))
            
        for word in sentence.lower().split(' '):
            word_counts[word] = word_counts.get(word, 0) + 1
    vocab = [word for word in word_counts if word_counts[word] >= threshold]
    print 'Filtered words from %d to %d' % (len(word_counts), len(vocab))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2, u'<UNK>': 3}
    idx_to_word = {0: u'<NULL>' , 1: u'<START>', 2: u'<END>', 3:u'<UNK>'}
    
    idx = 4
    for word in vocab:
        word_to_idx[word] = idx
        idx_to_word[idx] = word
        idx += 1

    return word_to_idx, idx_to_word, max_len
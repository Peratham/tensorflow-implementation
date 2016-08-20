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
        decoded.append('<START> '+ ' '.join(words))
    return decoded

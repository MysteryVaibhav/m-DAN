import numpy as np
from properties import *


# Returns a dictionary of image_id -> Caption (All 5 concatenated)
def get_captions():
    img_to_caption = {}
    with open(CAPTION_INFO, 'r', encoding='utf-8') as caption_data:
        for lines in caption_data.readlines():
            split_line = lines.split("\t")
            img_id, img_caption_id = split_line[0].split("#")
            caption = split_line[1]
            img_id = img_id.replace(".jpg", "")
            if img_id not in img_to_caption:
                img_to_caption[img_id] = []
            img_to_caption[img_id].append(caption.replace("\n", "").lower())
    img_to_caption, max_len = concatenate_all_captions(img_to_caption)
    return img_to_caption, max_len


def concatenate_all_captions(img_to_caption):
    new_img_to_caption = {}
    max_caption_len = 0
    for key, value in img_to_caption.items():
        sentence_words = get_words(value)
        new_img_to_caption[key] = sentence_words
        if len(sentence_words) > max_caption_len:
            max_caption_len = len(sentence_words)
    print("Max caption len: {}".format(max_caption_len))
    return new_img_to_caption, max_caption_len


def get_words(list_of_sentences):
    words = []
    for sentence in list_of_sentences:
        words.extend(sentence.split(" "))
    return words


def frequency_map(img_caption):
    word_freq = {}
    for _, caption in img_caption.items():
        for word in caption:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
    return word_freq


def construct_vocab(word_freq, k):
    word_idx= {}
    for word, freq in word_freq.items():
        if freq >= k:
            if word not in word_idx:
                word_idx[word] = len(word_idx)
    return word_idx


def img_caption_one_hot(img_caption, word_idx, max_len):
    img_to_one_hot = {}
    for img, caption in img_caption.items():
        one_hot = np.zeros(max_len)
        for i in range(len(caption)):
            if caption[i] in word_idx:
                one_hot[i] = word_idx[caption[i]] + 2
            else:
                # Idx 1 for unknown words
                one_hot[i] = 1
            # Idx 0 for padding
        mask = np.ones(max_len)
        mask[i:] = 0
        img_to_one_hot[img] = (one_hot, mask)
    return img_to_one_hot


def run():
    img_caption, max_len = get_captions()
    word_freq = frequency_map(img_caption)
    word_idx = construct_vocab(word_freq, 5)
    print("Total words in vocabulary: {}".format(len(word_idx) + 2))
    img_one_hot_and_mask = img_caption_one_hot(img_caption, word_idx, max_len)
    return img_one_hot_and_mask


def get_ids(name):
    list = []
    with open(SPLIT_INFO + "{}.lst".format(name), 'r', encoding='utf=8') as f:
        for id in f.readlines():
            list.append(id.split("/")[1].replace("\n", "").replace(".jpg", ""))
    return list


if __name__ == '__main__':
    run()


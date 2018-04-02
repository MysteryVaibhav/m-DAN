import json
import torch
import numpy as np
from properties import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def to_tensor(array):
    return torch.from_numpy(np.array(array)).float()


def to_variable(tensor, requires_grad=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, requires_grad=requires_grad)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


# Returns a dictionary of image_id -> Caption (All 5 concatenated)
def get_captions(caption_file):
    img_to_caption = {}
    id_img = {}
    max_len = 0
    cached_stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    with open(caption_file, 'r', encoding='utf-8') as caption_data:
        i = 0
        for lines in caption_data.readlines():
            split_line = lines.split("\t")
            img_id = split_line[0]
            caption = split_line[1]
            img_id = img_id.replace(".jpg", "")
            if img_id not in img_to_caption:
                img_to_caption[img_id] = process_words(caption.replace("\n", "").lower(), cached_stop_words, lemmatizer)
                if len(img_to_caption[img_id]) > max_len:
                    max_len = len(img_to_caption[img_id])
                id_img[i] = img_id
                i += 1
    return img_to_caption, max_len, id_img


def process_words(sentence, cached_stop_words, lemmatizer):
    #words = [word for word in sentence.split(" ") if word not in cached_stop_words]
    #words = [lemmatizer.lemmatize(word) for word in words]
    return sentence.split(" ")


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
    word_idx = {}
    for word, freq in word_freq.items():
        if freq >= k:
            if word not in word_idx:
                word_idx[word] = len(word_idx)
    return word_idx


def encode_caption(caption, word_idx, max_len):
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
    return one_hot, mask


def img_caption_one_hot(img_caption, word_idx, max_len):
    img_to_one_hot = {}
    for img, caption in img_caption.items():
        img_to_one_hot[img] = encode_caption(caption, word_idx, max_len)
    return img_to_one_hot


def run(caption_file):
    img_caption, max_len, _ = get_captions(caption_file)
    print("Max len: {}".format(max_len))
    word_freq = frequency_map(img_caption)
    word_idx = construct_vocab(word_freq, 5)
    print("Total words in vocabulary: {}".format(len(word_idx) + 2))
    img_one_hot_and_mask = img_caption_one_hot(img_caption, word_idx, max_len)
    return img_one_hot_and_mask


def get_ids(name, split_file, strip=False):
    list = []
    with open(split_file + "{}.lst".format(name), 'r', encoding='utf=8') as f:
        for id in f.readlines():
            base = id.split("/")[1].replace("\n", "").replace(".jpg", "")
            if strip:
                list.append(base)
            else:
                list.append(base + "#0")
                list.append(base + "#1")
                list.append(base + "#2")
                list.append(base + "#3")
                list.append(base + "#4")
    return list


def extract_concept_vectors(concepts_dir, number_of_concepts):
    if os.path.exists('concept_vectors.npy'):
        return np.load('concept_vectors.npy').item()

    scores_dict = {}
    for filename in os.listdir(concepts_dir):
        score = np.zeros(number_of_concepts)   # Len of sin346
        i = 0
        img = filename.replace(".json", "")
        with open(concepts_dir + filename) as json_data:
            d = json.load(json_data)
            # for elem in d['sports487']:
            #     score[i] = float(elem['score'])
            #     i += 1
            # for elem in d['kinetics']:
            #     score[i] = float(elem['score'])
            #     i += 1
            for elem in d['sin346']:
                score[i] = float(elem['score'])
                i += 1
            # for elem in d['places365']:
            #     score[i] = float(elem['score'])
            #     i += 1
            # for elem in d['fcvid']:
            #     score[i] = float(elem['score'])
            #     i += 1
            # for elem in d['ucf101']:
            #     score[i] = float(elem['score'])
            #     i += 1
            # for elem in d['yfcc609']:
            #     score[i] = float(elem['score'])
            #     i += 1
            scores_dict[img] = score
    np.save('concept_vectors.npy', scores_dict)
    return scores_dict


if __name__ == '__main__':
    run(CAPTION_INFO)
    extract_concept_vectors(CONCEPT_DIR)


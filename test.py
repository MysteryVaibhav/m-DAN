from model import mDAN
from process_data import *
from util import *
import os.path
import torch.utils.data

# Load model
model = mDAN()
model.load_state_dict(torch.load('model_weights_full_60_0.254.t7'))
if torch.cuda.is_available():
    model = model.cuda()

# Get the mapping
to_concatenate=True
img_caption, _, _ = get_captions()
word_freq = frequency_map(img_caption)
word_idx = construct_vocab(word_freq, 5)
img_caption, max_len, _ = get_captions(to_concatenate=to_concatenate)
test_ids = get_ids('test')


class CustomDataSet(torch.utils.data.TensorDataset):
    def __init__(self):
        self.ids = test_ids
        self.num_of_samples = len(self.ids)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        #image = np.random.random((NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION))
        image = np.load(TRAIN_IMAGES_DIR + "{}.npy".format(self.ids[idx])).reshape((NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION))
        dummy_one_hot, dummy_mask = encode_caption([x for x in ' '.split(" ")], word_idx, max_len)
        return to_tensor(dummy_one_hot).long(), to_tensor(dummy_mask), to_tensor(image)


# Get the visual context vectors for all the test images:
if os.path.exists('test_visual_context_features.npy'):
    test_z_v = np.load('test_visual_context_features.npy')
else:
    test_z_v = None  # no_of_images * visual_context_features
    data_loader = torch.utils.data.DataLoader(CustomDataSet(), batch_size=BATCH_SIZE,
                                              shuffle=False)
    for (caption_, mask_, image_) in data_loader:
        _, _, z_v = model(to_variable(caption_),
                          to_variable(mask_),
                          to_variable(image_), True)
        if test_z_v is None:
            test_z_v = z_v.data.cpu()
        else:
            test_z_v = torch.cat((test_z_v, z_v.data.cpu()), 0)
    test_z_v = test_z_v.numpy()
    print(test_z_v.shape)
    np.save('test_visual_context_features.npy', test_z_v)


def expand(original):
    o_len = len(original.split(" "))
    n_sentence = original
    n_len = o_len
    while n_len + o_len + 1 < max_len:
        n_sentence += ' ' + original
        n_len += o_len + 1
    return n_sentence


sentence_to_img = []
set_test_ids = set(test_ids)
for key, value in img_caption.items():
    if key not in set_test_ids:
        continue
    if to_concatenate:
        sentence_to_img.append((value, key))
    else:
        for each in value:
            sentence_to_img.append((expand(each), key))


class CustomDataSet1(torch.utils.data.TensorDataset):
    def __init__(self):
        self.sent_to_image = sentence_to_img
        self.num_of_samples = len(self.sent_to_image)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        dummy_image = np.random.random((NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION))
        if to_concatenate:
            dummy_one_hot, dummy_mask = encode_caption(self.sent_to_image[idx][0], word_idx, max_len)
        else:
            dummy_one_hot, dummy_mask = encode_caption([x for x in self.sent_to_image[idx][0].split(" ")], word_idx, max_len)
        return to_tensor(dummy_one_hot).long(), to_tensor(dummy_mask), to_tensor(dummy_image), self.sent_to_image[idx][1]


# Get the textual context vector for the test caption
data_loader = torch.utils.data.DataLoader(CustomDataSet1(), batch_size=BATCH_SIZE, shuffle=False)
r_1 = 0
r_5 = 0
r_10 = 0
for (caption_, mask_, image_, label) in data_loader:
    _, z_u, _ = model(to_variable(caption_),
                      to_variable(mask_),
                      to_variable(image_), True)
    z_u = z_u.data.cpu().numpy()

    # Compute similarity with the existing images
    similarity = np.matmul(test_z_v, z_u.T)
    for column in range(similarity.shape[1]):
        top_10_img_idx = (-similarity[:, column]).argsort()[:10]
        if label[column] == test_ids[top_10_img_idx[0]]:
            r_1 += 1
            r_5 += 1
            r_10 += 1
        elif label[column] in [test_ids[x] for x in top_10_img_idx[1:5]]:
            r_5 += 1
            r_10 += 1
        elif label[column] in [test_ids[x] for x in top_10_img_idx[6:10]]:
            r_10 += 1

print("R@1 : {}".format(r_1/len(sentence_to_img)))
print("R@5 : {}".format(r_5/len(sentence_to_img)))
print("R@10 : {}".format(r_10/len(sentence_to_img)))

# Average features

#Test, individual
#R@1 : 0.1456
#R@5 : 0.3816
#R@10 : 0.4666

#Test, combined
#R@1 : 0.338
#R@5 : 0.676
#R@10 : 0.761

#Val, individual
#R@1 : 0.136
#R@5 : 0.3618
#R@10 : 0.4534

#Val, combined
#R@1 : 0.317
#R@5 : 0.667
#R@10 : 0.748

# All features

#Test, individual
#R@1 : 0.1246
#R@5 : 0.3342
#R@10 : 0.4272

#Test, combined
#R@1 : 0.268
#R@5 : 0.598
#R@10 : 0.701

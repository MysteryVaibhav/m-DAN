import torch.utils.data
from util import *


class CustomDataSet(torch.utils.data.TensorDataset):
    def __init__(self, img_one_hot, ids, regions_in_image, visual_feature_dimension, image_features_dir):
        self.img_one_hot = img_one_hot
        self.ids = ids
        self.num_of_samples = len(ids)
        self.regions_in_image = regions_in_image
        self.visual_feature_dimension = visual_feature_dimension
        self.image_features_dir = image_features_dir

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        input, mask = self.img_one_hot[self.ids[idx]]

        image = np.random.random((self.regions_in_image, self.visual_feature_dimension))
        # image = np.load(self.image_features_dir + "{}.npy".format(self.ids[idx].split("#")[0])).reshape(
        #    (self.regions_in_image, self.visual_feature_dimension))

        r_n = idx
        img_idx = self.ids[idx].split("#")[0]
        r_n_idx = img_idx
        while r_n_idx == img_idx:
            r_n = np.random.randint(self.num_of_samples)
            r_n_idx = self.ids[r_n].split("#")[0]

        # Return negative caption and image
        # image_neg = np.load(self.image_features_dir + "{}.npy".format(self.ids[r_n].split("#")[0])).reshape(
        #    (self.regions_in_image, self.visual_feature_dimension))
        image_neg = np.random.random((self.regions_in_image, self.visual_feature_dimension))

        input_neg, mask_neg = self.img_one_hot[self.ids[r_n]]
        return to_tensor(input).long(), to_tensor(mask), to_tensor(image), \
               to_tensor(input_neg).long(), to_tensor(mask_neg), to_tensor(image_neg)


def get_k_random_numbers(n, curr, k=16):
    random_indices = set()
    while len(random_indices) < k:
        idx = np.random.randint(n)
        if idx != curr and idx not in random_indices:
            random_indices.add(idx)
    return list(random_indices)


class CustomDataSet1(torch.utils.data.TensorDataset):
    def __init__(self, test_ids, regions_in_image, visual_feature_dimension, image_features_dir, max_caption_len):
        self.ids = test_ids
        self.num_of_samples = len(self.ids)
        self.regions_in_image = regions_in_image
        self.visual_feature_dimension = visual_feature_dimension
        self.image_features_dir = image_features_dir
        self.max_caption_len = max_caption_len

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        image = np.random.random((self.regions_in_image, self.visual_feature_dimension))
        # image = np.load(self.image_features_dir + "{}.npy".format(self.ids[idx])).reshape(
        #    (self.regions_in_image, self.visual_feature_dimension))
        dummy_one_hot = np.ones(self.max_caption_len)
        dummy_mask = np.ones(self.max_caption_len)
        return to_tensor(dummy_one_hot).long(), to_tensor(dummy_mask), to_tensor(image)


class CustomDataSet2(torch.utils.data.TensorDataset):
    def __init__(self, img_one_hot, val_ids, regions_in_image, visual_feature_dimension):
        self.img_one_hot = img_one_hot
        self.ids = val_ids
        self.num_of_samples = len(self.ids)
        self.regions_in_image = regions_in_image
        self.visual_feature_dimension = visual_feature_dimension

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        dummy_image = np.random.random((self.regions_in_image, self.visual_feature_dimension))
        dummy_one_hot, dummy_mask = self.img_one_hot[self.ids[idx]]
        return to_tensor(dummy_one_hot).long(), to_tensor(dummy_mask), to_tensor(dummy_image), self.ids[idx].split("#")[
            0]


class DataLoader:
    def __init__(self, params):
        self.params = params
        self.img_one_hot = run()
        self.train_ids = get_ids('train')
        self.val_ids = get_ids('val')
        self.plain_val_ids = get_ids('val', strip=True)
        self.test_ids = get_ids('test')
        self.plain_test_ids = get_ids('test', strip=True)
        self.training_data_loader = torch.utils.data.DataLoader(CustomDataSet(self.img_one_hot,
                                                                              self.train_ids,
                                                                              params.regions_in_image,
                                                                              params.visual_feature_dimension,
                                                                              params.image_features_dir
                                                                              ),
                                                                batch_size=self.params.mini_batch_size,
                                                                shuffle=True)
        self.eval_data_loader_1 = torch.utils.data.DataLoader(CustomDataSet1(self.plain_val_ids,
                                                                             params.regions_in_image,
                                                                             params.visual_feature_dimension,
                                                                             params.image_features_dir,
                                                                             params.max_caption_len
                                                                             ),
                                                              batch_size=self.params.mini_batch_size,
                                                              shuffle=False)
        self.eval_data_loader_2 = torch.utils.data.DataLoader(CustomDataSet2(self.img_one_hot,
                                                                             self.val_ids,
                                                                             params.regions_in_image,
                                                                             params.visual_feature_dimension),
                                                              batch_size=self.params.mini_batch_size,
                                                              shuffle=False)
        self.test_data_loader_1 = torch.utils.data.DataLoader(CustomDataSet1(self.plain_test_ids,
                                                                             params.regions_in_image,
                                                                             params.visual_feature_dimension,
                                                                             params.image_features_dir,
                                                                             params.max_caption_len
                                                                             ),
                                                              batch_size=self.params.mini_batch_size,
                                                              shuffle=False)
        self.test_data_loader_2 = torch.utils.data.DataLoader(CustomDataSet2(self.img_one_hot,
                                                                             self.test_ids,
                                                                             params.regions_in_image,
                                                                             params.visual_feature_dimension),
                                                              batch_size=self.params.mini_batch_size,
                                                              shuffle=False)

    @staticmethod
    def hard_negative_mining(model, pos_cap, pos_mask, pos_image, neg_cap, neg_mask, neg_image):
        similarity = model(to_variable(neg_cap), to_variable(neg_mask), to_variable(pos_image), False)
        s_v_pos_u_neg = similarity.data.cpu().numpy()
        random_indices = [get_k_random_numbers(len(pos_image), curr) for curr in range(len(pos_image))]
        argmax_cap = to_tensor([each[np.argmax(s_v_pos_u_neg[each])] for each in random_indices]).long()
        neg_cap = torch.index_select(neg_cap, 0, argmax_cap)
        neg_mask = torch.index_select(neg_mask, 0, argmax_cap)
        similarity = model(to_variable(pos_cap), to_variable(pos_mask), to_variable(neg_image), False)
        s_u_pos_v_neg = similarity.data.cpu().numpy()
        random_indices = [get_k_random_numbers(len(neg_image), curr) for curr in range(len(neg_image))]
        argmax_img = to_tensor([each[np.argmax(s_u_pos_v_neg[each])] for each in random_indices]).long()
        neg_image = torch.index_select(neg_image, 0, argmax_img)
        return pos_cap, pos_mask, pos_image, neg_cap, neg_mask, neg_image

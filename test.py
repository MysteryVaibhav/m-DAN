from model import mDAN
from process_data import *
from util import *
import os.path
import torch.utils.data


class CustomDataSet1(torch.utils.data.TensorDataset):
    def __init__(self, test_ids):
        self.ids = test_ids
        self.num_of_samples = len(self.ids)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        #image = np.random.random((NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION))
        image = np.load(TRAIN_IMAGES_DIR + "{}.npy".format(self.ids[idx])).reshape((NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION))
        dummy_one_hot = np.ones(MAX_CAPTION_LEN)
        dummy_mask = np.ones(MAX_CAPTION_LEN)
        return to_tensor(dummy_one_hot).long(), to_tensor(dummy_mask), to_tensor(image)


class CustomDataSet2(torch.utils.data.TensorDataset):
    def __init__(self, img_one_hot, val_ids):
        self.img_one_hot = img_one_hot
        self.ids = val_ids
        self.num_of_samples = len(self.ids)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        dummy_image = np.random.random((NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION))
        dummy_one_hot, dummy_mask = self.img_one_hot[self.ids[idx]]
        return to_tensor(dummy_one_hot).long(), to_tensor(dummy_mask), to_tensor(dummy_image), self.ids[idx].split("#")[0]


def recall(model, img_one_hot):
    test_z_v = None  # no_of_images * visual_context_features
    plain_val_ids = get_ids('test', strip=True)
    data_loader = torch.utils.data.DataLoader(CustomDataSet1(plain_val_ids), batch_size=BATCH_SIZE, shuffle=False)
    for (caption_, mask_, image_) in data_loader:
        _, _, z_v = model(to_variable(caption_),
                          to_variable(mask_),
                          to_variable(image_), True)
        if test_z_v is None:
            test_z_v = z_v.data.cpu()
        else:
            test_z_v = torch.cat((test_z_v, z_v.data.cpu()), 0)
    test_z_v = test_z_v.numpy()

    val_ids = get_ids('test')
    data_loader = torch.utils.data.DataLoader(CustomDataSet2(img_one_hot, val_ids), batch_size=BATCH_SIZE, shuffle=False)
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
            if label[column] == plain_val_ids[top_10_img_idx[0]]:
                r_1 += 1
                r_5 += 1
                r_10 += 1
            elif label[column] in [plain_val_ids[x] for x in top_10_img_idx[1:5]]:
                r_5 += 1
                r_10 += 1
            elif label[column] in [plain_val_ids[x] for x in top_10_img_idx[6:10]]:
                r_10 += 1

    return r_1 / len(val_ids), r_5 / len(val_ids), r_10 / len(val_ids)


# Load model
model = mDAN()
model.load_state_dict(torch.load('model_weights_ind_40_0.1942.t7'))
if torch.cuda.is_available():
    model = model.cuda()

# Get the mapping
img_one_hot = run()
r_1, r_5, r_10 = recall(model, img_one_hot)

print("R@1 : {}".format(r_1))
print("R@5 : {}".format(r_5))
print("R@10 : {}".format(r_10))

# Average features

#Test, individual (After 40 epochs)
#R@1 : 0.2048
#R@5 : 0.4924
#R@10 : 0.597

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

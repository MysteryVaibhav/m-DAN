import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import numpy as np
from model import mDAN
from timeit import default_timer as timer
from properties import *
from util import *
import sys
from process_data import run, get_ids
from tqdm import tqdm


class CustomDataSet(torch.utils.data.TensorDataset):
    def __init__(self, img_one_hot, ids, is_train):
        self.img_one_hot = img_one_hot
        self.ids = ids
        self.num_of_samples = len(ids)
        self.is_train = is_train

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        input, mask = self.img_one_hot[self.ids[idx]]
        
        #image = np.random.random((NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION))
        image = np.load(TRAIN_IMAGES_DIR + "{}.npy".format(self.ids[idx].split("#")[0])).reshape((NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION))
        if not self.is_train:
            return to_tensor(input).long(), to_tensor(mask), to_tensor(image)
        r_n = idx
        img_idx = self.ids[idx].split("#")[0]
        r_n_idx = img_idx
        while r_n_idx == img_idx:
            r_n = np.random.randint(self.num_of_samples)
            r_n_idx = self.ids[r_n].split("#")[0]
        # Return negative caption and image

        image_neg = np.load(TRAIN_IMAGES_DIR + "{}.npy".format(self.ids[r_n].split("#")[0])).reshape((NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION))
        #image_neg = np.random.random((NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION))

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
    plain_val_ids = get_ids('val', strip=True)
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

    val_ids = get_ids('val')
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


def init_xavier(m):
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)
        m.bias.data.zero_()


class margin_loss(nn.Module):
    def __init__(self):
        super(margin_loss, self).__init__()

    def forward(self, s_v, s_v_, s_u_):
        loss = ((MARGIN - s_v + s_v_).clamp(min=0) + (MARGIN - s_v + s_u_).clamp(
            min=0)).sum()
        return loss


def train():
    # Get one-hot for caption for all images:
    img_one_hot = run()
    train_ids = get_ids('train')
    train_data_loader = torch.utils.data.DataLoader(CustomDataSet(img_one_hot, train_ids, True), batch_size=BATCH_SIZE,
                                                    shuffle=True)
    model = mDAN()
    model.apply(init_xavier)
    loss_function = margin_loss()
    if torch.cuda.is_available():
        model = model.cuda()
        loss_function = loss_function.cuda()
    #model.load_state_dict(torch.load('model_weights_ind_33_0.1908.t7'))
    #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.0005)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    prev_best = 0
    for epoch in range(EPOCHS):
        scheduler.step()
        minibatch = 1
        losses = []
        start_time = timer()
        num_of_mini_batches = len(train_ids) // BATCH_SIZE
        for (caption, mask, image, neg_cap, neg_mask, neg_image) in train_data_loader:
            caption, mask, image, neg_cap, neg_mask, neg_image = hard_negative_mining(model, caption, mask, image, neg_cap, neg_mask, neg_image)
            optimizer.zero_grad()
            # Run our forward pass.
            similarity = model(to_variable(caption), to_variable(mask), to_variable(image), False)
            similarity_neg_1 = model(to_variable(neg_cap), to_variable(neg_mask), to_variable(image), False)
            similarity_neg_2 = model(to_variable(caption), to_variable(mask), to_variable(neg_image), False)
            # Compute the loss, gradients, and update the parameters by calling optimizer.step()
            loss = loss_function(similarity, similarity_neg_1, similarity_neg_2)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            optimizer.step()
            sys.stdout.write("[%d/%d] :: Training Loss: %f   \r" % (
                minibatch, num_of_mini_batches, np.asscalar(np.mean(losses))))
            sys.stdout.flush()
            minibatch += 1
        r_at_1, r_at_5, r_at_10 = recall(model, img_one_hot)
        print("Epoch {} : Training Loss: {:.5f}, R@1 : {}, R@5 : {}, R@10 : {}, Time elapsed {:.2f} mins"
              .format(epoch, np.asscalar(np.mean(losses)), r_at_1, r_at_5, r_at_10, (timer() - start_time) / 60))
        if r_at_1 > prev_best:
            print("Recall at 1 increased....saving weights !!")
            prev_best = r_at_1
            torch.save(model.state_dict(), 'model_weights_ind_{}_{}.t7'.format(epoch+1, r_at_1))


if __name__ == '__main__':
    train()

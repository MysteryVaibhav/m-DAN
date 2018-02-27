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
        image = np.load(TRAIN_IMAGES_DIR + "{}.npy".format(self.ids[idx])).reshape((NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION))
        if not self.is_train:
            return to_tensor(input).long(), to_tensor(mask), to_tensor(image)

        # Return negative caption and image
        image_neg = np.load(TRAIN_IMAGES_DIR + "{}.npy".format(self.ids[(idx + 100) % self.num_of_samples])).reshape((NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION))
        #image_neg = np.random.random((NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION))

        input_neg, mask_neg = self.img_one_hot[self.ids[(idx + 100) % self.num_of_samples]]
        return to_tensor(input).long(), to_tensor(mask), to_tensor(image), \
               to_tensor(input_neg).long(), to_tensor(mask_neg), to_tensor(image_neg)


def recall_at_1(model, val_data_loader):
    all_z_u = None
    all_z_v = None
    for (caption, mask, image) in val_data_loader:
        _, z_u, z_v = model(to_variable(caption), to_variable(mask), to_variable(image), True)
        if all_z_u is None:
            all_z_u = z_u
            all_z_v = z_v
        else:
            all_z_u = torch.cat((all_z_u, z_u), 0)
            all_z_v = torch.cat((all_z_v, z_v), 0)
    similarity_matrix = torch.mm(all_z_u, all_z_v.t())
    max, _ = torch.max(similarity_matrix, 1)
    r_at_1 = torch.sum(torch.diag(similarity_matrix, 0) == max)
    return r_at_1.data.cpu()[0] / 1000


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
    val_data_loader = torch.utils.data.DataLoader(CustomDataSet(img_one_hot, get_ids('val'), False), batch_size=BATCH_SIZE,
                                                    shuffle=True)

    model = mDAN()

    loss_function = margin_loss()
    if torch.cuda.is_available():
        model = model.cuda()
        loss_function = loss_function.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    prev_best = 0
    for epoch in range(EPOCHS):
        minibatch = 1
        losses = []
        start_time = timer()
        num_of_mini_batches = len(train_ids) // BATCH_SIZE
        for (caption, mask, image, neg_cap, neg_mask, neg_image) in train_data_loader:
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
        print("Epoch {} : Training Loss: {:.5f}, Time elapsed {:.2f} mins"
              .format(epoch, np.asscalar(np.mean(losses)), (timer() - start_time) / 60))
        if (epoch + 1) % 5 == 0:
            r_at_1 = recall_at_1(model, val_data_loader)
            print("R@1 after {} epochs: {}".format(epoch + 1, r_at_1))
            if r_at_1 > prev_best:
                print("Recall at 1 increased....saving weights !!")
                prev_best = r_at_1
                torch.save(model.state_dict(), 'model_weights_{}_{}.t7'.format(epoch+1, r_at_1))


if __name__ == '__main__':
    train()

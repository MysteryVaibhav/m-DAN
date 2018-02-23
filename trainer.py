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


class CustomDataSet(torch.utils.data.TensorDataset):
    def __init__(self, img_one_hot, ids):
        self.img_one_hot = img_one_hot
        self.ids = ids
        self.num_of_samples = len(ids)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        input, mask = self.img_one_hot[self.ids[idx]]
        # Return negative caption and image
        input_neg, mask_neg = self.img_one_hot[self.ids[(idx + 100) % self.num_of_samples]]
        return to_tensor(input).long(), to_tensor(mask), to_tensor(np.random.random((NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION))),\
               to_tensor(input_neg).long(), to_tensor(mask_neg), to_tensor(np.random.random((NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION)))


class margin_loss(nn.Module):
    def __init__(self):
        super(margin_loss, self).__init__()

    def forward(self, s_v, s_v_, s_u_):
        loss = ((MARGIN - s_v + s_v_).clamp(min=0) + (MARGIN - s_v + s_u_).clamp(min=0)).sum()  #TODO: Add similarity for negative samples
        return loss


def train():
    # Get one-hot for caption for all images:
    img_one_hot = run()
    train_data_loader = torch.utils.data.DataLoader(CustomDataSet(img_one_hot, get_ids('train')), batch_size=BATCH_SIZE, shuffle=True)

    # Get pre-trained embeddings
    pre_trained_embeddings = np.random.random((VOCAB_SIZE, EMBEDDING_DIMENSION))
    pre_trained_embeddings[0, :] = 0

    model = mDAN(pre_trained_embeddings)

    loss_function = margin_loss()
    if torch.cuda.is_available():
        model = model.cuda()
        loss_function = loss_function.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(EPOCHS):
        minibatch = 1
        losses = []
        start_time = timer()
        num_of_mini_batches = len(train_data_loader) // BATCH_SIZE
        for (caption, mask, image, neg_cap, neg_mask, neg_image) in train_data_loader:
            optimizer.zero_grad()
            # Run our forward pass.
            similarity = model(to_variable(caption), to_variable(mask), to_variable(image))
            similarity_neg_1 = model(to_variable(neg_cap), to_variable(neg_mask), to_variable(image))
            similarity_neg_2 = model(to_variable(caption), to_variable(mask), to_variable(neg_image))
            # Compute the loss, gradients, and update the parameters by calling optimizer.step()
            loss = loss_function(similarity, similarity_neg_1, similarity_neg_2)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            optimizer.step()
            sys.stdout.write("[%d/%d] :: Training Loss: %f   \r" % (
                minibatch, num_of_mini_batches, np.asscalar(np.mean(losses))))
            sys.stdout.flush()
            minibatch += 1
            if minibatch > num_of_mini_batches:
                break
        print("Epoch {} : Training Loss: {:.5f}, Time elapsed {:.2f} mins"
              .format(epoch, np.asscalar(np.mean(losses)), (timer() - start_time) / 60))
    torch.save(model.state_dict(), 'model_weights_{}.t7'.format(HIDDEN_DIMENSION))


if __name__ == '__main__':
    train()

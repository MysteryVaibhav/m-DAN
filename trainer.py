import torch.optim as optim
import torch.utils.data
import numpy as np
from model import mDAN
from timeit import default_timer as timer
from properties import *
from util import *
import sys


def train():
    dataset = torch.utils.data.TensorDataset(torch.LongTensor(np.zeros((NO_OF_IMAGES, MAX_CAPTION_LEN))),
                                                     to_tensor(np.random.random((NO_OF_IMAGES,
                                                                                NO_OF_REGIONS_IN_IMAGE, VISUAL_FEATURE_DIMENSION))))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Get pre-trained embeddings
    pre_trained_embeddings = np.random.random((VOCAB_SIZE, EMBEDDING_DIMENSION))

    model = mDAN(pre_trained_embeddings)

    loss_function = torch.nn.BCELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        loss_function = loss_function.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(EPOCHS):
        minibatch = 1
        losses = []
        start_time = timer()
        num_of_mini_batches = NO_OF_IMAGES // BATCH_SIZE
        for (caption, image) in data_loader:
            model.zero_grad()
            # Run our forward pass.
            scores = model(to_variable(caption), to_variable(image))
            # Compute the loss, gradients, and update the parameters by calling optimizer.step()
            loss = loss_function(scores, to_variable(image))
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

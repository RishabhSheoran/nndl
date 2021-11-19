#!/usr/bin/env python
# coding: utf-8

# # Baseline

# ## Description
# 
# Transformer architectures are a standard for natural language processing. In the below implementation, we have mapped this multi-label image classification task to that of a language modelling task. To mimic words and sets of words, the input images are cut up into patches, where each patch represents a word in the set. These sets of words are then passed through a single layer transformer along with a class token - this is considered as the encoding of the input set of image patches. The encoding corresponding to the class token is then passed through a linear layer whose outputs are interpreted as classification scores.
# 

# ## Architecture (Equation)
# 
# ![ANN Model Architecture](./images/ann_baseline.png "ANN Baseline Architecture")

# ## Implementation

# In[1]:


import sys

sys.path.append("..")


# In[ ]:


get_ipython().system(' pip install einops')


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from einops import rearrange  # ! pip install einops

from src.dataset import ImageDataset, CLASSES
from src.utils import get_error


# ### Model definition

# In[3]:


PATCH_SIZE = 16


def generate_positional_encoding(seq_length, dim):
    assert dim == 2 * (dim // 2)  # check if dim is divisible by 2
    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, nb_heads):
        super(TransformerEncoder, self).__init__()
        assert hidden_size == nb_heads * (hidden_size // nb_heads)
        self.MHA = nn.MultiheadAttention(hidden_size, nb_heads)
        self.LLcat = nn.Linear(2 * hidden_size, hidden_size)
        self.LL1 = nn.Linear(hidden_size, hidden_size)
        self.LL2 = nn.Linear(hidden_size, hidden_size)
        self.LN1 = nn.LayerNorm(hidden_size)
        self.LN2 = nn.LayerNorm(hidden_size)

    def forward(self, g_seq, pos):
        seq_length = g_seq.size(0)
        bs = g_seq.size(1)
        pos = pos.unsqueeze(dim=1).repeat_interleave(
            bs, dim=1
        )

        h_cat = self.LLcat(
            torch.cat((g_seq, pos), dim=2)
        )

        h_MHA_seq, _ = self.MHA(h_cat, h_cat, h_cat)
        h = self.LN1(h_cat + h_MHA_seq)

        h_MLP = self.LL2(torch.relu(self.LL1(h)))
        h_seq = self.LN2(h + h_MLP)
        return h_seq


class MultiLabelANN(nn.Module):
    def __init__(self, hidden_size, nb_heads, no_classes):
        super(MultiLabelANN, self).__init__()

        # This "classification token" will be added to each image and used as a proxy for image encoding
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        self.encoder = TransformerEncoder(hidden_size, nb_heads)
        self.LL = nn.Linear(hidden_size, no_classes)
        

    def expand_cls_to_batch(self, batch):
        """
        Args:
            batch: batch size
        Returns: cls token expanded to the batch size
        """
        return self.cls_token.expand([batch, -1, -1])

    def forward(self, img, pos):
        bs = img.shape[0]
        img_patches = rearrange(
            img,
            "b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)",
            patch_x=PATCH_SIZE,
            patch_y=PATCH_SIZE,
        )

        img_patches = torch.cat((self.expand_cls_to_batch(bs), img_patches), dim=1)
        
        h_seq = self.encoder(img_patches, pos)

        score_seq = self.LL(h_seq[:, 0, :])

        return score_seq


# In[4]:


BATCH_SIZE = 1
train_data = ImageDataset(train=True)
valid_data = ImageDataset(train=False)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)


# In[5]:


### Check if inference works correctly

# nb_heads = 16
# hidden_size = 768
# no_classes = 6

# net = MultiLabelANN(hidden_size, nb_heads, no_classes)
# print(net)
# dataiter = iter(train_loader)
# images, labels = dataiter.next().values()
# seq_length = 144
# pos = generate_positional_encoding(BATCH_SIZE, 768)
# scores = net(images.view(BATCH_SIZE, 3, 144, 256), pos)
# scores.shape
# print(scores)
# scores = torch.sigmoid(scores)
# scores


# In[1]:


# Param declarations
N_EPOCHS = 10
LR = 0.001

NUM_HEADS = 16
HIDDEN_SIZE = 768
NUM_CLASSES = 6


# ### Model instantiation

# In[7]:


# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model instance
model = MultiLabelANN(HIDDEN_SIZE, NUM_HEADS, NUM_CLASSES)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ### Train

# In[8]:


for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
    train_running_loss = 0.0
    train_err = 0.0
    model.train()

    # TRAINING ROUND
    for i, data in enumerate(train_loader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # get the inputs
        inputs, labels = data.values()

        inputs = inputs.view(BATCH_SIZE, 3, 144, 256)
        inputs = inputs.to(device)
        labels = labels.to(device)

        pos = generate_positional_encoding(BATCH_SIZE, 768)
        pos = pos.to(device)

        outputs = model(inputs, pos)

        labels.squeeze_()
        outputs.squeeze_()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_err += get_error(outputs.detach(), labels, BATCH_SIZE)

    model.eval()
    print(
        "Epoch:  %d | Loss: %.4f | Train Error: %.4f"
        % (epoch, train_running_loss / i, train_err / i)
    )


# ### Test

# In[9]:


test_err = 0.0
for i, data in enumerate(valid_loader, 0):
    inputs, labels = data.values()

    pos = generate_positional_encoding(BATCH_SIZE, 768)
    pos = pos.to(device)

    outputs = model(inputs.view(BATCH_SIZE, 3, 144, 256).to(device), pos)
    outputs = torch.sigmoid(outputs)
    #     print((outputs.detach() > 0.5).float(), labels)

    test_err += get_error(outputs.detach(), labels.to(device), BATCH_SIZE)

print("Validation Error: %.4f" % (test_err / i))


# In[10]:


actors = np.array(CLASSES)
print("actors: ", actors)


# In[14]:


model.eval()
test_running_error = 0.0
output_list = []
target_list = []
for counter, data in enumerate(valid_loader):

    image, target = data["image"].to(device), data["label"]
    # get all the index positions where value == 1
    target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]

    pos = generate_positional_encoding(BATCH_SIZE, 768)

    outputs = model(image, pos.to(device))
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    lab = torch.where(outputs >= 0.3, 1, 0)[0]
    pred = torch.where(lab == 1)
    error = get_error(outputs, target, 1)

    output_list.append(outputs.numpy())
    target_list.append(target.numpy())

    test_running_error += error

    string_predicted = ""
    string_actual = ""
    for i in range(len(pred)):
        string_predicted += f"{actors[pred[i]]}    "
    for i in range(len(target_indices)):
        string_actual += f"{actors[target_indices[i]]}    "

    if 10 < counter < 20:
        image = image.squeeze(0)
        image = image.detach().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
        plt.show()

test_error = test_running_error / counter
print(f"Test Error: {test_error} %")

out = np.array(output_list).squeeze(axis=1)
tar = np.array(target_list).squeeze(axis=1)


# ## Results
# 
# For an ANN with one layer of encoding and no regularization, the below are the results.
# 
# * In training for 10 epochs, the error converges only after 2 epochs with an error rate of 23.24%.  The loss does change between epoch (still around ~0.51) but does not make any substantial improvement.
# * In testing, as can be seen in the above, the model also predicts the class "Monica" despite the input. Although the test error is just 25.45%, it does not capture the true essence of the results and so we cannot consider that as a reliable measure of quality. The quality of prediction are poor and in the next section, let us discuss some of the factors influencing the results.

# ## Discussion
# 
# A number of factors influence the above results and some of those are,
# 
# * **Dataset Size** - for the given dataset of 3000 images, the model overfits very early on (on epoch 2). If the dataset size can be increased to the order of millions of images, then we can expect the results to improve.
# * **Regularization** - another reason for overfitting could be the lack of regularization. One could add dropouts or apply image augmentation techniques to prevent the model from overfitting
# * **Class distribution** - based on EDA, one can see that the class "Monica" is over-represented in training data and that could also play a role in the model memorizing that particular class
# * **Inductive Bias** - Unlike CNNs, transformers have less image-specific inductive bias as the attention layers are global. This means that the model has to learn the positional relations between image pixels from scratch which is not the case with CNNs.
# 
# We try to address some of the above concerns in the ANN improvement discussed next.

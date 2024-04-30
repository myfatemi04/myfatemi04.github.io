---
title: 'Sparse Autoencoder'
date: 2023-10-22
permalink: /posts/2023/10/celeba-sparse-autoencoder
tags:
  - projects
  - interpretability
  - machine-learning
---

What kinds of useful features can we extract from images?

I wonder how we can meaningfully code for orientation and scale in images through neural networks. My goal is to maximize the number of explainable features that neural networks generate.

I experiment with the CelebA dataset and finetune a pretrained model (from ImageNet) to disambiguate between faces. The goal is that the model learns to output a fine-tuned vector space that distinguishes between faces. I could use a VAE but that takes a while; instead I will estimate mean and variance from a decent batch size and use KL divergence on that.

## Download Dataset

We'll be using the CelebA dataset for this. I'll assume you have a Kaggle account and can download with that. To save me from the hassle of `kaggle.json`, I'm using environment variables.


```python
# First, download CelebA dataset.
import os
import getpass

os.environ['KAGGLE_USERNAME'] = getpass.getpass("Username")
os.environ['KAGGLE_KEY'] = getpass.getpass("API Key")

!kaggle datasets download -d jessicali9530/celeba-dataset -p CelebA
```

    Username··········
    API Key··········
    Downloading celeba-dataset.zip to CelebA
    100% 1.33G/1.33G [00:12<00:00, 230MB/s]
    100% 1.33G/1.33G [00:12<00:00, 117MB/s]



```python
# Our images unzip to `./img_align_celeba/img_align_celeba` (relative to the current directory; not the CelebA folder for some reason.)
!unzip CelebA/celeba-dataset.zip -d data > /dev/null
```

## Load dataset

Loading images from a folder is a common process, so Torchvision has a way for us to do this automatically. To test that everything works, I set up a quick visualization. It renders the first 16 images as a 4x4 grid using `matplotlib`.


```python
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def visualize_4x4(images: torch.Tensor, title=None):
    # Shape: [N, C, H, W]

    # Make a grid of images
    grid = torchvision.utils.make_grid(images, nrow=4).numpy()

    # Render
    plt.figure(figsize=(4, 4))
    plt.title(title or "Celebrities!")
    plt.imshow(grid.transpose((1, 2, 0)))
    plt.axis("off")
    plt.show()

# Now, train a contrastive model. Use CelebA dataset.
# dataset[i] = (PIL.Image.Image, <dummy class label>)
dataset = ImageFolder(root="./data", transform=T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
]))

visualize_4x4([dataset[i][0] for i in range(16)])

```

    /usr/local/lib/python3.10/dist-packages/torchvision/transforms/functional.py:1603: UserWarning: The default value of the antialias parameter of all the resizing transforms (Resize(), RandomResizedCrop(), etc.) will change from None to True in v0.17, in order to be consistent across the PIL and Tensor backends. To suppress this warning, directly pass antialias=True (recommended, future default), antialias=None (current default, which means False for Tensors and True for PIL), or antialias=False (only works on Tensors - PIL will still use antialiasing). This also applies if you are using the inference transforms from the models weights: update the call to weights.transforms(antialias=True).
      warnings.warn(



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_5_1.png)
    


## Training

We'll use a `resnet18` pretrained model which is pretty standard, and it should have decent feature representations already, which should easily be finetuned to this domain.


```python
from torchvision.models import resnet

# Decide whether to use the GPU. This speeds up training a *LOT* if it's available.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# Remove the fully-connected layer because we're interested in embeddings.
model.fc = torch.nn.Identity()
model = model.to(device)

```

    Using cache found in /root/.cache/torch/hub/pytorch_vision_v0.10.0
    /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)


## Training the model

Because this is a smaller model, its capacity is strained for distinguishing between faces. This model was trained to detect objects of a certain range of classes (in ImageNet), and its internal/latent representation is a continuous vector. Classifications are made by having class-specific vectors that are dotted with the latent vector to create scores. In this case, we want to minimize the similarity between vectors belonging to different faces; so we use a contrastive loss. This forces the model to output dissimilar vectors for dissimilar faces. By inspecting the mapping the model creates as a result (which we assume will create similar vectors for similar faces, as the model is forced to compress some of the information it sees), we can try to extract salient facial features.

The loss function is as follows. Let $X$ be the set of input images.
$$
  y \leftarrow \text{Model}(X) \\
  \text{Loss}(y) := L_{ce} + L_{orient}
$$

Where Cross Entropy is defined as
$$
  L_{ce}(logprobs, target) := -\sum_{i=0}^{k} target_i logprobs_i
$$

Where $logprobs$ is the log-Softmax output of the prediction vector.

To train the model to incorporate implicit orientation, I train it to reduce the mean-squared-error between the encoding of the input image and the negative of the encoding of the flipped image.
$$
  L_{orient}(y_{normal}, y_{flipped}) := L_{mse}(y_{normal}, -y_{flipped})
$$


```python
import tqdm

batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)

N_EPOCHS = 1
for epoch in range(N_EPOCHS):
  with tqdm.tqdm(dataloader, desc="Training...") as pbar:
    for (images, _) in pbar:
      images = images.to(device)
      # Train model to disambiguate between these images.
      # Use embeddings as features. Instead of retraining
      # all of EfficientNet, add a linear layer at the end
      # to get embeddings. Rotate vector around first dimension.
      # 180º rotated images should have negative embeddings of each other.
      images_rotated = torch.rot90(images, 2, (2, 3))
      # [N, embedding_dim]
      embeddings = model(images)
      embeddings_rotated = model(images_rotated)

      # ORIENTATION LOSS
      loss_orientation = torch.nn.functional.mse_loss(embeddings, -embeddings_rotated)

      # CONTRASTIVE LOSS
      # Dot each of these embeddings with each other and treat these as logits.
      # Then use Cross Entropy loss.
      embeddings_normalized = torch.nn.functional.normalize(embeddings, dim=1)
      scores = embeddings @ embeddings.T
      loss_contrastive = torch.nn.functional.cross_entropy(scores, torch.arange(len(scores), device=device))

      loss = loss_contrastive + loss_orientation

      optim.zero_grad()
      loss.backward()
      optim.step()
      pbar.set_postfix({"loss": loss.item(), "contrastive_loss": loss_contrastive.item(), "orientation_loss": loss_orientation.item()})

```

    Training...: 100%|██████████| 1583/1583 [34:38<00:00,  1.31s/it, loss=0.0944, contrastive_loss=0.00796, orientation_loss=0.0864]


## Save our model.

This took like half an hour. I trained on the whole of CelebFace while writing the below content. Let's please save this model lol.


```python
torch.save(model.state_dict(), "resnet_18_celebface_state_dict.pt")
```

## Create inferences.

We did a single pass over this dataset. Let's create vectors for each of the faces in the dataset now.


```python
results = []
model.eval()

# Notably, we *DO NOT SHUFFLE*!
dataloader_eval = DataLoader(dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():
  with tqdm.tqdm(dataloader_eval, desc="Generating output vectors...") as pbar:
    for (images, _) in pbar:
      images = images.to(device)
      embeddings = model(images)
      results.append(embeddings.to('cpu'))

# Took about 17m50s for 1583 batches (202599 input faces)
```


```python
# Generate a single tensor
face_vectors = torch.cat(results, dim=0)

# See how much data we got :0
# Each vector has 512 dimensions.
print(face_vectors.shape)

# Store this tensor
torch.save(face_vectors, "face_vectors.pt")
```

    torch.Size([202599, 512])


After training for a bit, we hopefully have a model that is strong at distinguishing different faces! (I only trained on 1/3 of the dataset, but it's probably fine... the dataset is big and this is just for experiment.)

I want to see what the most salient features are for this model, too. How can we do this?

A [page from Stanford CS231n](https://cs231n.github.io/understanding-cnn/) shares a nice overview of ways to evaluate CNNs. One [blog post](https://ml4a.github.io/ml4a/visualizing_convnets/) shares a suite of ways to visualize convolutional neural network predictions. We can:
 * Visualize maximally-activating patches (i.e., see what inputs for the receptive field of a given output cause the highest activation). We do this by inputting a bunch of images and see which ones are the most strongly correlated with certain features.
 * Occlusion and gradient-based experiments are useful for identifying *what* contributed the most to an output. Examples include [SHAP](https://shap-lrjball.readthedocs.io/en/latest/index.html), [GradCAM](https://arxiv.org/abs/1610.02391), and [Integrated Gradients](https://arxiv.org/abs/1703.01365).
   * We can look at the gradient of a classification w.r.t a specific part of the input image, or we can occlude parts of the image and see how our classification changes.
 * Deconv/Guided backprop are ways to *synthetically generate* maximally-activating patches. Basically, we are causing the model to hallucinate a given response, and inspecting what input it's "imagining". One popular example is [DeepDream](https://en.wikipedia.org/wiki/DeepDream). See [Peeking Inside ConvNets](https://www.auduno.com/2016/06/18/peeking-inside-convnets/) for a page with nice examples.
   * On a technical level, what we're doing is using the same technique for optimizing model weights against a loss function to optimize an image input against a class activation. We start with an image and perform gradient descent, using the class activation as the objective (so $loss = -activation$).
 * Embeddings: We can use methods like [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) to reduce the dimensionality of input images. Essentially, output vectors (which can be in a massive $512$-dimension vector space, etc.) are converted to 2D vectors, in a way that keeps neighboring vectors in the larger space close together in the 2D space. This gives a nice at-a-glance way to visualize what images the network perceives as similar to each other.

With respect to language models, [Anthropic](https://www.anthropic.com/) introduces a [Sparse Autoencoder](https://transformer-circuits.pub/2023/monosemantic-features/index.html), which takes the set of output features (which form a densely-packed vector space of real-valued dimensions) and outputs a large set of disrete features. This is similar to [work done by OpenAI to discover multimodal neurons](https://openai.com/research/multimodal-neurons).
 * The sparse autoencoder is used specifically in *language models*. The goal of the sparse autoencoder is to discover features with *monosemanticity*. Because the inner workings of language models are not regularized at all, the intermediate vector space can look like a mess, as long as the optimizer knows how to improve predictions given more input. Although it makes optimization simply "work", it's bad for us because we don't know what anything means. Additionally, because we can assume that most words don't have all possible semantic meanings at once, some sort of compression is happening, resulting in an overpopulated basis. What this means is that $1024$ binary features might be stored in a $512$-dimensional real-valued vector. Why is this bad? Because we can't point to any one feature and say, "this is the poetry detector". What the sparse autoencoder does is take the range of internal representations the model creates across a wide variety of inputs, and translates it to a vector space where each dimension of the vector corresponds to a different feature.

I realized I forgot to store the output vectors of the model. It's fine; using `torch.no_grad()` for non-training tasks makes compute take less time. Also, this is just for experimentation.

The question is now, will the model identify *my* face separately from someone elses? (And what features might the model be using?)


## Testing Methods

We'll now try various attribution methods to understand the features of our model. I'll use the sparse autoencoder proposed by Anthropic for this. The method resembles a standard linear encoder and decoder, with the difference that decoder bias is subtracted from the input before being encoded.
$$
\overline{x} = x - b_d \\
f = \text{ReLU}(W_e\overline{x} + b_e) \\
\hat{x} = W_df+b_d \\
L = \frac{1}{|X|} \sum_{x \in X} \|x - \hat{x}\|_2^2 + \lambda\|\textbf{f}\|_1
$$

The loss function $L$ decomposes into:
 * A reconstruction loss (input vs. reconstructed input; MSE)
 * A regularization loss (L1 norm of sparse features)


```python
class SparseAutoencoder(torch.nn.Module):
  def __init__(self, input_feature_dim, sparse_feature_dim):
    super().__init__() # necessary for any torch.nn.Module subclass

    self.weight_encoder = torch.nn.Parameter(torch.zeros((input_feature_dim, sparse_feature_dim), dtype=torch.float32))
    self.weight_decoder = torch.nn.Parameter(torch.zeros((sparse_feature_dim, input_feature_dim), dtype=torch.float32))
    self.bias_decoder = torch.nn.Parameter(torch.zeros(input_feature_dim, dtype=torch.float32))
    self.bias_encoder = torch.nn.Parameter(torch.zeros(sparse_feature_dim, dtype=torch.float32))
    # initialize weight matrices
    torch.nn.init.normal_(self.weight_encoder)
    torch.nn.init.normal_(self.weight_decoder)

  def encode(self, input_features):
    # (n, input_feature_dim)
    xbar = input_features - self.bias_decoder
    # (n, sparse_feature_dim)
    f = torch.nn.functional.relu(xbar @ self.weight_encoder + self.bias_encoder)
    return f

  def decode(self, sparse_features):
    # (n, sparse_feature_dim)
    xhat = sparse_features @ self.weight_decoder + self.bias_decoder
    return xhat

def sparse_autoencoder_loss(model, input_features, l1_lambda):
  # Assume input is batched: (n, input_feature_dim)
  f = model.encode(input_features)
  x_hat = model.decode(f)
  # L_reconstruction = torch.nn.functional.mse_loss(input_features, x_hat)
  L_reconstruction = torch.norm(input_features - x_hat, 2)
  L_complexity = l1_lambda * torch.norm(f, 1)
  return L_reconstruction, L_complexity

```

## Detecting Important Features

Here, we use the Sparse Autoencoder to detect monosemantic features. (I could have added an L1 loss to the feature vectors during training at minimal overhead...)

We first load the face vectors we saved. (My runtime crashed, lol.)


```python
import PIL.Image
```


```python
import torch

face_vectors = torch.load("face_vectors.pt")
```


```python
sparse_feature_dim = 512
sae = SparseAutoencoder(input_feature_dim=512, sparse_feature_dim=sparse_feature_dim).to('cuda')
sae_optim = torch.optim.Adam(sae.parameters(), lr=1e-5, weight_decay=1e-4)
```


```python
import tqdm

# l_recon = 30 or so.
# Takes like 1 minute train this many epochs, because of CUDA and the small size of the dataset.
for epoch in range(30):
  i = 0
  sae_batch_size = 512
  N = len(face_vectors)
  order = torch.randperm(N)
  with tqdm.tqdm(total=N, desc="Training sparse autoencoder...") as pbar:
    while i < N:
      batch = face_vectors[order[i:i + 512]].to('cuda')
      # L_reconstruction, L_complexity = sparse_autoencoder_loss(sae, torch.nn.functional.normalize(batch) * torch.sqrt(torch.tensor(512.0)), l1_lambda=0.1)
      L_reconstruction, L_complexity = sparse_autoencoder_loss(sae, torch.nn.functional.normalize(batch), l1_lambda=0.1)
      loss_sae = L_reconstruction + L_complexity
      sae_optim.zero_grad()
      loss_sae.backward()
      sae_optim.step()
      pbar.set_postfix({"loss_sae": loss_sae.item(), "recon": L_reconstruction.item(), "compl": L_complexity.item()})
      pbar.update(sae_batch_size)
      i += sae_batch_size
```


```python
# Save our sparse autoencoder.
torch.save(sae.state_dict(), "sparse_autoencoder_weights.pt")
```

## Generate features for each image

Now that we're done training, let's infer the feature set for each image.


```python
# Let's sort images by their activation of certain neurons.
sparse_features = []

with torch.no_grad():
  i = 0
  sae_batch_size = 512
  N = len(face_vectors)
  with tqdm.tqdm(total=N, desc="Extracting sparse features...") as pbar:
    while i < N:
      batch = face_vectors[i:i + 512].to('cuda')
      sparse_features.append(sae.encode(batch).to('cpu'))
      i += sae_batch_size
      pbar.update(sae_batch_size)

feats = torch.cat(sparse_features, dim=0)
```

    Extracting sparse features...: 202752it [00:00, 419656.01it/s]                            



```python
# Save our sparse features.
torch.save(feats, "face_vectors_sparse.pt")
```

## Inspecting Results

The sparse autoencoder has now been trained. Now, let's look at which images maximally activate each feature. I'll normalize the features using `torch.nn.functional.normalize` before sorting so we can look at which images have the strongest *relative* activation for each feature.

### Result Without Sparse Autoencoder

Let's see if the initial face vector (i.e., before being encoded) has interpretable feature representations.


```python
feats_dense_normed = torch.nn.functional.normalize(face_vectors)

for feature_id in range(16):
  # Sort images by feature `feature_id`.
  order = torch.argsort(feats_dense_normed[:, feature_id], descending=True)
  visualize_4x4(
      [dataset[order[i]][0] for i in range(16)],
      title="Maximally-Activating Examples",
  )
```

    /usr/local/lib/python3.10/dist-packages/torchvision/transforms/functional.py:1603: UserWarning: The default value of the antialias parameter of all the resizing transforms (Resize(), RandomResizedCrop(), etc.) will change from None to True in v0.17, in order to be consistent across the PIL and Tensor backends. To suppress this warning, directly pass antialias=True (recommended, future default), antialias=None (current default, which means False for Tensors and True for PIL), or antialias=False (only works on Tensors - PIL will still use antialiasing). This also applies if you are using the inference transforms from the models weights: update the call to weights.transforms(antialias=True).
      warnings.warn(



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_1.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_2.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_3.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_4.png)
    



<!--     
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_5.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_6.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_7.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_8.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_9.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_10.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_11.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_12.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_13.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_14.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_15.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_30_16.png) -->
    


### Result With Sparse Autoencoder


```python
feats_normed = torch.nn.functional.normalize(feats)

for feature_id in range(16):
  # Sort images by feature `feature_id`.
  order = torch.argsort(feats_normed[:, feature_id], descending=True)
  visualize_4x4(
      [dataset[order[i]][0] for i in range(16)],
      title="Maximally-Activating Examples",
  )
```

    /usr/local/lib/python3.10/dist-packages/torchvision/transforms/functional.py:1603: UserWarning: The default value of the antialias parameter of all the resizing transforms (Resize(), RandomResizedCrop(), etc.) will change from None to True in v0.17, in order to be consistent across the PIL and Tensor backends. To suppress this warning, directly pass antialias=True (recommended, future default), antialias=None (current default, which means False for Tensors and True for PIL), or antialias=False (only works on Tensors - PIL will still use antialiasing). This also applies if you are using the inference transforms from the models weights: update the call to weights.transforms(antialias=True).
      warnings.warn(



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_1.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_2.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_3.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_4.png)
    



<!--     
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_5.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_6.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_7.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_8.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_9.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_10.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_11.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_12.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_13.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_14.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_15.png)
    



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_28_16.png)
     -->



## Unbiased Blind Interpretability Test

To see whether there is a measurable difference between the feature encoders, I'll take a few features I haven't seen before and rate each feature for its explainability. Then, I'll see whether I categorized sparse autoencoder features or regular features as being more or less interpretable.


```python
import random
import time

test_features = [*range(16, 32)]
orders = []

for feature_id in test_features:
  # Sort images by feature `feature_id`.
  order_dense = torch.argsort(feats_dense_normed[:, feature_id], descending=True)
  order_sparse = torch.argsort(feats_normed[:, feature_id], descending=True)
  orders.append((order_dense, "dense:" + str(feature_id)))
  orders.append((order_sparse, "sparse:" + str(feature_id)))

random.shuffle(orders)
results = []
for (order, feature_id) in orders:
  visualize_4x4(
      [dataset[order[i]][0] for i in range(16)],
      title="Maximally-Activating Examples",
  )
  time.sleep(0.5)
  rating = int(input("How visually-consistent do images for this feature appear? (1 - 7)\n"))
  results.append((feature_id, rating))

```


    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_0.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    5



    
<!-- ![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_2.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    7



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_4.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    2



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_6.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    4



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_8.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    3



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_10.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    4



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_12.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    6



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_14.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    3



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_16.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    1



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_18.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    3



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_20.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    3



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_22.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    5



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_24.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    3



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_26.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    2



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_28.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    3



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_30.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    2



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_32.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    5



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_34.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    2



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_36.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    3



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_38.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    4



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_40.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    4



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_42.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    2



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_44.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    4



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_46.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    6



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_48.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    6



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_50.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    6



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_52.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    5



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_54.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    2



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_56.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    6



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_58.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    2



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_60.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    5 -->


...



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_32_62.png)
    


    How visually-consistent do images for this feature appear? (1 - 7)
    5



```python
# Let's see how the ratings turned out.

ratings = {"sparse": [], "dense": []}
for (feature_id, rating) in results:
  vector_type = feature_id.split(":")[0]
  ratings[vector_type].append(rating)

# Create a figure and axis
fig, ax = plt.subplots()

# Create histograms
n, bins, patches = ax.hist([ratings['sparse'], ratings['dense']], bins=[1, 2, 3, 4, 5, 6, 7], label=["Sparse Features", "Dense Features"], alpha=0.5)

# Calculate the positions for centered ticks
tick_positions = 0.5 * (bins[:-1] + bins[1:])
tick_labels = [str(int(tick)) for tick in tick_positions]

# Set the x-tick positions and labels
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

# Set labels and legend
ax.set_xlabel("Interpretability Rating 1-7")
ax.set_title("Human-Annotated Blind Interpretability Ratings")
ax.legend()

# Show the plot
plt.show()

```


    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_33_0.png)
    



```python
# Let's get the mean and standard deviation of these ratings through a Pandas summary.
# We'll also do a t-test to see if sparse features are truly more explainable than dense features.
import pandas as pd
from scipy.stats import ttest_ind

display(pd.DataFrame(ratings).describe())

ttest_result = ttest_ind(ratings['sparse'], ratings['dense'])
print(f"t-stat: {ttest_result.statistic:.4f}, pvalue: {ttest_result.pvalue:.4f}")
```



  <div id="df-8922efca-ad0d-462f-be47-e6a9cfadfb82" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sparse</th>
      <th>dense</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.437500</td>
      <td>3.250000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.547848</td>
      <td>1.437591</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.500000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>4.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.000000</td>
      <td>6.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8922efca-ad0d-462f-be47-e6a9cfadfb82')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-8922efca-ad0d-462f-be47-e6a9cfadfb82 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8922efca-ad0d-462f-be47-e6a9cfadfb82');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-d831b675-d5c9-4826-9ce2-261ef1f54825">
  <button class="colab-df-quickchart" onclick="quickchart('df-d831b675-d5c9-4826-9ce2-261ef1f54825')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-d831b675-d5c9-4826-9ce2-261ef1f54825 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>



    t-stat: 2.2486, pvalue: 0.0320


## Assessment of Results

We can see that some features correspond to the background color of the images (because that was useful when the model was trained to disambiguate between images). We also see that some features correspond specifically to women with blond hair, or images of celebrities in front of specifically dark / specifically light backgrounds. Some neurons are less interpretable and may be combinations of more subtle features discovered during the disambiguation step.

Overall, this analysis provides insight into the types of images that neural networks perceive as similar to each other, and what components each feature vector is made of.

The sparse autoencoder is a lightweight and useful way to uncover monosemantic features, as backed by a t-test. This is simply based on the incorporation of an L1-loss on the sparse feature set.

## Ignore

This is me verifying that the ImageDataset loaded images in alphabetical order, for reproducibility after closing this notebook.


```python
visualize_4x4([dataset[267][0]] * 16)
```

    /usr/local/lib/python3.10/dist-packages/torchvision/transforms/functional.py:1603: UserWarning: The default value of the antialias parameter of all the resizing transforms (Resize(), RandomResizedCrop(), etc.) will change from None to True in v0.17, in order to be consistent across the PIL and Tensor backends. To suppress this warning, directly pass antialias=True (recommended, future default), antialias=None (current default, which means False for Tensors and True for PIL), or antialias=False (only works on Tensors - PIL will still use antialiasing). This also applies if you are using the inference transforms from the models weights: update the call to weights.transforms(antialias=True).
      warnings.warn(



    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_37_1.png)
    



```python
import PIL.Image
PIL.Image.open("data/img_align_celeba/img_align_celeba/000268.jpg")
```




    
![png](/images/posts/2023/10/celeba-sparse-autoencoder/output_38_0.png)
    



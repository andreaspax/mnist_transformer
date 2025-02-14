import torchvision as tv
import torch
import matplotlib.pyplot as plt
import random
import utils
class MNISTDataset(torch.utils.data.IterableDataset):
  def __init__(self, train=True, single=False, total_samples=60000, seed=2):
    transform = tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.1307], std=[0.3081]), #important for getting values between -1 and 1
            ]
    )

    torch.manual_seed(seed)
    random.seed(seed)    

    self.dataset = tv.datasets.MNIST(
            root="./data", train=train, download=True, transform=transform
        )
    self.single = single
    self.total_samples = total_samples
    self.vocab = utils.vocab
  def __len__(self):
          return len(self.dataset)

  def _create_patches(self, image: torch.Tensor, patch_dim: int):
        patch_size = image.shape[0] // patch_dim  # 56 / 4 = 14

        image = image.unfold(0, patch_size, patch_size)  # shape: ( 4 x 56 x 14)
        image = image.unfold(1, patch_size, patch_size)  # shape: ( 4 x 4 x 14 x 14)

        patches = image.reshape(
            shape=(-1, patch_size, patch_size)
        )  # shape: (16 x 14 x 14)

        return patches

  def __iter__(self):
    count = 0
    while count < self.total_samples:
      labels = []
      images = []

      if self.single:
        rand_idx = random.randint(1, len(self.dataset)-1) 
        img, num = self.dataset[rand_idx]
        labels.append(num)
        combined = img[0]

      elif not self.single:
        for _ in range(4):
          rand_idx = random.randint(1, len(self.dataset)-1) 
          img, num = self.dataset[rand_idx]
          labels.append(num)
          images.append(img[0])  # img[0] removes channel dimension

        top = torch.cat([images[0], images[1]], dim=1)
        bottom = torch.cat([images[2], images[3]], dim=1)
        combined = torch.cat([top, bottom], dim=0)  

       # split into 16 images of 14 x 14 or 7 x 7 for single
      patches = self._create_patches(combined, 4)  # shape: (16 x 14 x 14)

      # Reshape from
      flattened = patches.reshape(16, -1)  # shape: (16 x 196)

      labels = torch.tensor(labels, dtype=torch.long)  # Convert to 1D tensor

      # Create input sequence with <start> token prepended
      input_seq = torch.cat([torch.tensor([10], dtype=torch.long), labels])

      # Create target sequence with <end> token appended
      target_seq = torch.cat([labels, torch.tensor([11], dtype=torch.long)])

      yield combined, flattened, input_seq, target_seq
      count += 1




if __name__ == '__main__':

  test_ds = MNISTDataset(train=True, single=False)
  print(len(test_ds))
  loader = torch.utils.data.DataLoader(test_ds, batch_size=1)
  
  # Get a single batch
  cmb, flt, in_lbl, tgt_lbl = next(iter(loader))
  print(cmb.shape)
  print(flt.shape)
  print(in_lbl)
  print(tgt_lbl)
  # # Remove batch dimension since we used batch_size=1
  img_cmb = cmb[0].squeeze(0)
  # img_flt = flt[0].squeeze(0)

  # print("Image shape:", img_cmb.shape)
  # print("Numbers:", lbl)
  plt.imshow(img_cmb, cmap='grey')
  plt.show()

  # # for patches check
  # fig, axs = plt.subplots(nrows=4, ncols=4)
  # # Plot the images in the subplots
  # for i, ax in enumerate(axs.flat):
  #   ax.imshow(img_flt[i])
  #   ax.axis('off')  # Turn off the axis
  # # Layout so plots do not overlap
  # fig.tight_layout()
  # plt.show()

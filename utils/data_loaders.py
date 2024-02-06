from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToTensor, RandomHorizontalFlip, Resize, Compose, ToPILImage
from skimage import color

class ColorizationDataset(Dataset):
  def __init__(self, image_folder_path, split = 'train', image_size = (256, 256)):
    self.dataset_image_size = image_size
    self.file_paths = image_folder_path
    torch_transforms = [
        ToTensor(),
        Resize(self.dataset_image_size, Image.BICUBIC),
        RandomHorizontalFlip(),
    ]

    if split == "train":
      self.transformations = Compose(torch_transforms)
    elif split == "validation":
      self.transformations = Compose(torch_transforms[ : 2] + torch_transforms[3 : ])

  def __len__(self):
    return len(self.file_paths)

  def __getitem__(self, idx):
    image = Image.open(self.file_paths[idx]).convert("RGB")
    image = self.transformations(image)
    image = color.rgb2lab(image.permute(1, 2, 0)).astype("float32")
    image = ToTensor()(image)
    image[[0], :, :] = image[[0], :, :] / 50. - 1. # Between -1 and 1
    image[[1, 2], :, :] = image[[1, 2], :, :] / 110. # Between -1 and 1
    return image
# Quick fix for the corrupted image error
# Run this in your notebook to set num_workers=0 which avoids the multiprocessing issue

# Replace the DataLoader section with this:

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

def collate_fn_safe(batch):
    """
    Safe collate function that filters out None values from corrupted images
    """
    # Filter out None entries
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    images, encoded_texts, texts = zip(*batch)

    # Stack images
    images = torch.stack(images, dim=0)

    # Get text lengths
    text_lengths = torch.LongTensor([len(t) for t in encoded_texts])

    # Concatenate all texts (CTC expects concatenated targets)
    encoded_texts = torch.cat(encoded_texts)

    return images, encoded_texts, text_lengths, texts


# Modify the Dataset class __getitem__ to handle errors
class IAMHandwritingDatasetSafe(IAMHandwritingDataset):
    def __getitem__(self, idx):
        try:
            return super().__getitem__(idx)
        except Exception as e:
            # Skip corrupted images
            print(f"Skipping corrupted image at index {idx}: {e}")
            return None


# Recreate datasets with safe version
train_dataset_safe = IAMHandwritingDatasetSafe(
    train_annotations, image_dir, char_to_idx,
    img_height=32, img_width=128, train=True
)

val_dataset_safe = IAMHandwritingDatasetSafe(
    val_annotations, image_dir, char_to_idx,
    img_height=32, img_width=128, train=False
)

# Create dataloaders with num_workers=0 to avoid multiprocessing issues
BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset_safe,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn_safe,
    num_workers=0,  # Set to 0 to avoid multiprocessing errors
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset_safe,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn_safe,
    num_workers=0,  # Set to 0
    pin_memory=torch.cuda.is_available()
)

print(f"✅ Train batches: {len(train_loader)}")
print(f"✅ Val batches:   {len(val_loader)}")
print("\n✅ Safe dataloaders created!")

"""
PyTorch datasets for cross-modal neural encoding.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class VGCOCODataset(Dataset):
    """Dataset for VG-COCO overlapping images and captions.

    This dataset loads image-caption pairs on-demand to avoid loading all
    images into memory at once.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing image paths and captions
    image_dir : Path
        Base directory for image files
    image_path_column : str
        Column name containing image paths
    text_column : str
        Column name containing text captions
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        image_path_column: str = "filepath",
        text_column: str = "sentences_raw"
    ):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.image_path_column = image_path_column
        self.text_column = text_column

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[Image.Image, str]:
        """Load image and text for given index.

        Parameters
        ----------
        idx : int
            Index of the sample to load

        Returns
        -------
        tuple[Image.Image, str]
            PIL Image and corresponding text
        """
        row = self.dataframe.iloc[idx]

        # Resolve image path
        rel_path = str(row[self.image_path_column])
        if rel_path.startswith("/"):
            img_path = Path(rel_path)
        else:
            img_path = self.image_dir / rel_path

        # Load image — fall back to a blank image on I/O errors (e.g. corrupt
        # Lustre blocks) so one bad file doesn't abort the whole job.
        try:
            image = Image.open(img_path).convert("RGB")
        except OSError as e:
            warnings.warn(f"Could not read {img_path}, substituting blank image: {e}")
            image = Image.new("RGB", (224, 224), color=0)

        # Get text
        text = str(row[self.text_column])

        return image, text

    @property
    def image_paths(self) -> list[str]:
        """Get all image paths in the dataset."""
        return self.dataframe[self.image_path_column].astype(str).tolist()

    @property
    def texts(self) -> list[str]:
        """Get all texts in the dataset."""
        return self.dataframe[self.text_column].astype(str).tolist()
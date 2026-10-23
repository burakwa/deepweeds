import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import tempfile
import shutil


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def class_names():
    return (
        "Chinee apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly acacia",
        "Rubber vine", "Siam weed", "Snake weed", "Negative"
    )


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with mock data."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "id": [f"T03_{i:05d}" for i in range(n)],
        "path": [f"images/T03_{i:05d}.jpg" for i in range(n)],
        "label": np.random.randint(0, 9, n),
        "group_id": [f"group_{i % 10}" for i in range(n)],
    })
    return df


@pytest.fixture
def mock_images(temp_dir, sample_dataframe):
    """Create mock image files for testing."""
    img_dir = temp_dir / "images"
    img_dir.mkdir()
    
    for _, row in sample_dataframe.iterrows():
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(img_dir / Path(row["path"]).name)
    
    return img_dir


@pytest.fixture
def sample_batch():
    """Create a sample batch of images and labels."""
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 9, (batch_size,))
    return images, labels
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

class SlideDataset(Dataset):
    def __init__(self, metadata, transform=None, base_path=None):
        """
        Args:
            metadata: DataFrame with 'filepath' and 'label' columns
            transform: Optional transform to apply to images
            base_path: Base path for resolving relative file paths. 
                      If None, uses current working directory.
        """
        self.metadata = metadata
        self.transform = transform
        # Set base_path to project root if not provided
        if base_path is None:
            # Try to find project root by looking for common markers
            current = Path.cwd()
            # Look for project root indicators
            project_root = current
            while project_root != project_root.parent:
                if (project_root / 'src').exists() or (project_root / 'data').exists():
                    break
                project_root = project_root.parent
            self.base_path = project_root
        else:
            self.base_path = Path(base_path)

    def __len__(self):
        return len(self.metadata)

    def _resolve_path(self, filepath):
        """
        Resolve file path, handling both absolute Linux-style paths and relative paths.
        
        Args:
            filepath: Path string from metadata
            
        Returns:
            Resolved Path object
        """
        path = Path(filepath)
        
        # If it's an absolute path starting with /data/raw/, convert to relative
        if path.is_absolute() and str(path).startswith('/data/'):
            # Convert /data/... to data/... relative to project root
            # path.parts[0] is '/', path.parts[1] is 'data', so we skip the first part
            relative_path = Path(*path.parts[1:])  # Skip '/', keep data/...
            resolved = self.base_path / relative_path
            if resolved.exists():
                return resolved
        
        # If it's already absolute and exists, use it as-is
        if path.is_absolute() and path.exists():
            return path
        
        # If it's relative, resolve against base_path
        if not path.is_absolute():
            resolved = self.base_path / path
            if resolved.exists():
                return resolved
        
        # Fallback: try as-is
        return path

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filepath = self._resolve_path(row['filepath'])
        image = Image.open(filepath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, row['label']

def create_data_loader(metadata, batch_size, shuffle=True, num_workers=2, transform=None, base_path=None):
    """
    Create a DataLoader for slide images.
    
    Args:
        metadata: DataFrame with 'filepath' and 'label' columns
        batch_size: Batch size for data loading
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        transform: Optional transform to apply to images
        base_path: Base path for resolving file paths (defaults to project root)
    
    Returns:
        DataLoader instance
    """
    dataset = SlideDataset(metadata, transform=transform, base_path=base_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
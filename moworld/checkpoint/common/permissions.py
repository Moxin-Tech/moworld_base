import os
from pathlib import Path
from checkpoint.common.constant import DIR_MODE, FILE_MODE


def set_directory_permissions(target_dir: str | Path) -> None:
    # Set root directory permissions
    target_dir.chmod(DIR_MODE)

    # Traverse the directory and set appropriate permissions for all files and subdirectories
    try:
        for root, dirs, files in os.walk(target_dir):
            # Set subdirectory permissions using pathlib operations
            root_path = Path(root)
            for directory in dirs:
                (root_path / directory).chmod(DIR_MODE)

            # Set file permissions using pathlib operations
            for file in files:
                (root_path / file).chmod(FILE_MODE)
    except OSError as e:
        raise OSError(f"Error occurred while setting permissions: {target_dir}") from e

from typing import Tuple
from huggingface_hub import snapshot_download
from pathlib import Path
from helpers.logger import logger

def download_model() -> Tuple[bool, str]:
    model_dir = Path("./tmp/model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model = model_dir / "model_final.pth"

    if model.is_file():
        logger.info("Model file already exists")
        return (True, model)
    
    local_dir = snapshot_download(
        repo_id="struan-mclean1/DartBoard", 
        local_dir=str(model_dir),
        local_dir_use_symlinks=False
    )

    print(local_dir)
    logger.info("Downloaded Model")

    return (True, model)

if __name__ == "__main__":
    download_model()

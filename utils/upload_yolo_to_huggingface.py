import os
import shutil

from huggingface_hub import HfApi, HfFolder, Repository, create_repo


def upload_yolo_to_huggingface(
    repo_name, model_path, hf_token, repo_type="model", private=False
):
    """
    Uploads a YOLO model to Hugging Face Hub.

    Args:
        repo_name (str): The name of the repository on Hugging Face.
        model_path (str): Path to the directory containing the YOLO model files.
        hf_token (str): Your Hugging Face authentication token.
        repo_type (str): Type of repository ("model", "dataset", etc.). Default is "model".
        private (bool): Whether the repository should be private. Default is False.
    """
    try:
        # Initialize Hugging Face API
        api = HfApi()

        # Create the repository on Hugging Face
        repo_url = create_repo(
            repo_name,
            token=hf_token,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )
        print(f"Repository created or found: {repo_url}")

        # Clone the repository to a temporary directory
        repo_local_path = f"./temp_{repo_name}"
        repo = Repository(
            local_dir=repo_local_path, clone_from=repo_url, use_auth_token=hf_token
        )

        # Copy YOLO model files to the cloned repository
        if not os.path.exists(repo_local_path):
            os.makedirs(repo_local_path)
        for item in os.listdir(model_path):
            s = os.path.join(model_path, item)
            d = os.path.join(repo_local_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

        # Add and push files to the repository
        repo.push_to_hub(
            commit_message="Upload YOLO model files", use_auth_token=hf_token
        )
        print(f"Model successfully uploaded to {repo_url}")

    except Exception as e:
        print(f"Error uploading model: {e}")

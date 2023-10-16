import os
from dotenv import load_dotenv

# Load the stored environment variables
load_dotenv()


def get_env_variable(var_name: str, allow_empty: bool = False) -> str:
    """
    Get the environment variable or raise an exception
    """
    env_var = os.getenv(var_name)
    if not allow_empty and env_var in (None, ""):
        raise KeyError(
            f"Environment variable {var_name} not set, and allow_empty is False"
        )
    return env_var

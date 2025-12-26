from openai import OpenAI
import httpx
from utils.config import RED, RESET, GREEN

# NOTE: Inside a standard Docker container, 'localhost' refers to the container itself.
# If Ollama is running on your host machine, you may need to adjust this URL

def check_connection(client: OpenAI) -> bool:
    """
    Attempts to list models from the Ollama server to check connectivity.
    """
    try:
        models = client.models.list()
        print(f"{GREEN}Ollama API is UP and accessible.{RESET}")
        return True
    except httpx.ConnectError:
        print(f"{RED}Connection Error:{RESET} Ollama server is not running or port 11434 is blocked.")
        print(f"{YELLOW}Tip:{RESET} If Ollama is on the host, check network settings.")
        return False
    except Exception as e:
        print(f"{RED}An unexpected error occurred:{RESET} {e}")
        return False
import requests
import base64
import json
import time
from typing import Dict

NEXAR_URL = "https://api.nexar.com/graphql"
PROD_TOKEN_URL = "https://identity.nexar.com/connect/token"

def get_token(client_id, client_secret):
    """Retrieve the Nexar API token using the provided client_id and client_secret."""

    if not client_id or not client_secret:
        raise Exception("client_id and/or client_secret are empty")

    token = {}
    try:
        token = requests.post(
            url=PROD_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret
            },
            allow_redirects=False,
        ).json()

    except Exception:
        raise

    return token
def decode_jwt(token: str) -> Dict:
    """Decode the JWT token payload."""
    padding = token.split(".")[1] + "=="
    decoded_bytes = base64.urlsafe_b64decode(padding)
    return json.loads(decoded_bytes.decode("utf-8"))

class NexarClient:
    def __init__(self, client_id: str, client_secret: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = requests.Session()
        self.session.keep_alive = False

        self.token_info = get_token(client_id, client_secret)
        self.session.headers.update({"token": self.token_info['access_token']})
        self.expiration = decode_jwt(self.token_info['access_token'])['exp']

    def check_expiration(self) -> None:
        """Check if the token is about to expire and refresh it if necessary."""
        if time.time() > self.expiration - 300:
            self.token_info = get_token(self.client_id, self.client_secret)
            self.session.headers.update({"token": self.token_info['access_token']})
            self.expiration = decode_jwt(self.token_info['access_token'])['exp']

    def execute_query(self, query: str, variables: Dict) -> Dict:
        """Execute a GraphQL query with the provided variables."""
        self.check_expiration()
        try:
            response = self.session.post(
                NEXAR_URL,
                json={"query": query, "variables": variables}
            )
            response.raise_for_status()
            result = response.json()

            if "errors" in result:
                error_messages = [error["message"] for error in result["errors"]]
                raise RuntimeError(f"GraphQL errors: {', '.join(error_messages)}")

            return result["data"]
        except requests.RequestException as e:
            raise RuntimeError("Error during Nexar API query") from e

import requests
import time
from typing import Dict

DIGIKEY_TOKEN_URL = "https://api.digikey.com/v1/oauth2/token"
DIGIKEY_SEARCH_URL = "https://api.digikey.com/products/v4/search/keyword"
DIGIKEY_MEDIA_URL = "https://api.digikey.com/products/v4/search/{}/media"
DIGIKEY_CATEGORIES_URL = "https://api.digikey.com/products/v4/search/categories"

def get_token(client_id: str, client_secret: str) -> Dict:
    """Retrieve the Digi-Key API token using the provided client_id and client_secret."""
    if not client_id or not client_secret:
        raise ValueError("client_id and/or client_secret are empty")

    try:
        response = requests.post(
            url=DIGIKEY_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret
            },
        )
        response.raise_for_status()
        token = response.json()
    except requests.RequestException as e:
        raise RuntimeError("Error obtaining Digi-Key token") from e

    return token

class DigiKeyClient:
    def __init__(self, client_id: str, client_secret: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = requests.Session()

        self.token_info = get_token(client_id, client_secret)
        self.session.headers.update({
            "Authorization": f"Bearer {self.token_info['access_token']}",
            "X-DIGIKEY-Client-Id": self.client_id
        })
        self.expiration = time.time() + self.token_info.get('expires_in', 3600)

    def check_expiration(self) -> None:
        """Check if the token is about to expire and refresh it if necessary."""
        if time.time() > self.expiration - 300:
            self.token_info = get_token(self.client_id, self.client_secret)
            self.session.headers.update({
                "Authorization": f"Bearer {self.token_info['access_token']}"
            })
            self.expiration = time.time() + self.token_info.get('expires_in', 3600)

    def search_products(self, body) -> dict:
        """Search for products using a keyword."""
        self.check_expiration()
        try:
            response = self.session.post(
                DIGIKEY_SEARCH_URL,
                json=body,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError("Error during Digi-Key product search") from e

    def search_product_media(self, part_number) -> dict:
        """Search for products using a keyword."""
        self.check_expiration()
        try:
            response = self.session.get(
                DIGIKEY_MEDIA_URL.format(part_number),
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code == 404:
                return {"res": "Not found"}
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError("Error during Digi-Key media search") from e

    def get_categories(self) -> dict:
        """Search for products using a keyword."""
        self.check_expiration()
        try:
            response = self.session.get(
                DIGIKEY_CATEGORIES_URL,
            )
            if response.status_code == 200:
                return response.json(), response.headers
            if response.status_code == 404:
                return {"res": "Not found"}
            response.raise_for_status()

            return response.json(), response.headers
        except requests.RequestException as e:
            raise RuntimeError("Error during Digi-Key media search") from e



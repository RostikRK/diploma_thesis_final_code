import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
from selenium.common.exceptions import NoSuchElementException, TimeoutException


def is_valid_pdf(filename, min_size=30000):
    """
    Checks if the file is at least min_size bytes
    and starts with '%PDF'.
    """
    if not os.path.exists(filename):
        return False

    if os.path.getsize(filename) < min_size:
        return False

    with open(filename, 'rb') as f:
        header = f.read(5)
        if not header.startswith(b'%PDF-'):
            return False

    return True


def get_final_url(starting_url):
    """
    Gets the final url after the redirect.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    service = Service(ChromeDriverManager().install())
    browser = webdriver.Chrome(service=service, options=options)

    browser.get(starting_url)
    time.sleep(5)
    final_url = browser.current_url
    print(f"Final URL after redirect: {final_url}")
    browser.quit()
    return final_url


def download_pdf(url, filename):
    """
    Simple Download function by requests.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    try:
        response = requests.get(url, headers=headers, allow_redirects=True)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded file as {filename}")
            if not is_valid_pdf(filename):
                os.remove(filename)
                print(f"{filename} is invalid or too small. Marking download as failed.")
                return False
            return True
        else:
            print(f"Failed to download from {url}. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Exception downloading from {url}: {e}")
        return False


def download_pdf_viewer_button(url, filename):
    """
    Fallback download using the UI button:
    1) Open the page with Selenium
    2) Look for the download link
    3) Extract the real PDF URL from 'href'
    4) Download via requests
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    service = Service(ChromeDriverManager().install())
    browser = webdriver.Chrome(service=service, options=options)

    try:
        browser.get(url)
        time.sleep(5)  # Give time to load

        try:
            download_button = browser.find_element(By.ID, "download")
        except NoSuchElementException:
            print("Could not find an element with id='download'")
            return False

        # Extract the link's href
        pdf_href = download_button.get_attribute("href")
        if not pdf_href:
            print("'download' element has no href attribute.")
            return False

        if pdf_href.startswith('/'):
            from urllib.parse import urljoin
            pdf_href = urljoin(browser.current_url, pdf_href)

        print(f"Found PDF link: {pdf_href}")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            )
        }
        r = requests.get(pdf_href, headers=headers, allow_redirects=True)
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(r.content)
            print(f"[Viewer] Downloaded file as {filename}")

            if not is_valid_pdf(filename):
                os.remove(filename)
                print(f"{filename} is invalid or too small. Marking download as failed.")
                return False
            return True
        else:
            print(f"PDF link returned status code {r.status_code}")
            return False
    except (NoSuchElementException, TimeoutException) as e:
        print(f"Could not complete PDF download from {url}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error from {url}: {e}")
        return False
    finally:
        browser.quit()


def advanced_pdf_download(url, filename):
    if download_pdf(url, filename):
        return True
    if download_pdf_viewer_button(url, filename):
        return True

    print(f"All download methods failed for {url}")
    return False

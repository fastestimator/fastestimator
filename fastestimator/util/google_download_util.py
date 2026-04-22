# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import random
import time

import gdown
import requests
from tqdm import tqdm

from fastestimator.util.util import is_valid_file

# Prefer the OS-managed system CA bundle over everything else.
# The system bundle is updated via update-ca-certificates and includes BOTH
# public CAs and any proxy root CAs (e.g. GE HealthCare TLS inspection
# certs). Env vars like REQUESTS_CA_BUNDLE are intentionally NOT honoured here
# because they are often set to a single  cert file, which replaces the
# full public CA chain and causes verification failures for external hosts.
_SYSTEM_CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"  # Debian/Ubuntu path


def _get_ca_bundle() -> str:
    """Return the best available CA bundle path for this environment.

    Priority:
      1. OS system bundle at /etc/ssl/certs/ca-certificates.crt — most complete;
         contains public CAs plus any proxy CAs added via
         update-ca-certificates. This is preferred over env-var overrides because
         REQUESTS_CA_BUNDLE is often set to a single-cert file that would break
         verification of public hosts.
      2. certifi default — fallback for systems without a system bundle (e.g. macOS
         without the system bundle at the expected path).
    """
    if os.path.isfile(_SYSTEM_CA_BUNDLE):
        return _SYSTEM_CA_BUNDLE
    import certifi
    return certifi.where()


def download_url(url: str, destination: str, max_retries: int = 3) -> None:
    """Download a file from a public HTTP/HTTPS URL with retry logic.

    Uses requests so that redirects (e.g. GitHub releases → S3) are followed automatically.
    Downloads to a temporary file first so a failed attempt never leaves a partial file at the
    destination path.

    Args:
        url: The URL to download from.
        destination: The local file path to save the download to.
        max_retries: Maximum number of download attempts.

    Raises:
        ValueError: If the file could not be downloaded after all retries.
    """
    if is_valid_file(destination):
        print(f"File {destination} already exists, skipping download.")
        return

    tmp_path = destination + ".tmp"
    ca_bundle = _get_ca_bundle()
    for attempt in range(max_retries):
        if is_valid_file(destination):
            return
        if attempt > 0:
            wait = (2**(attempt - 1)) * random.uniform(5, 15)
            time.sleep(wait)
        try:
            response = requests.get(url, stream=True, timeout=60, verify=ca_bundle)
            response.raise_for_status()
            total = int(response.headers.get('Content-Length', 0))
            with open(tmp_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True,
                                                  desc=os.path.basename(destination)) as bar:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
                    bar.update(len(chunk))
            os.rename(tmp_path, destination)
            return
        except Exception as e:
            print(f"\nException occurred while downloading {destination} (attempt {attempt + 1}/{max_retries}): {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    raise ValueError(f"Couldn't download {destination} after {max_retries} retries.")


def download_url_with_fallback(primary_url: str, fallback_gdrive_id: str, destination: str,
                               max_retries: int = 3) -> None:
    """Download a file, trying the primary URL first and falling back to Google Drive if it fails.

    Args:
        primary_url: The preferred public HTTP/HTTPS URL to try first.
        fallback_gdrive_id: Google Drive file ID to use if the primary URL fails all retries.
        destination: The local file path to save the download to.
        max_retries: Maximum number of attempts for each source.

    Raises:
        ValueError: If the file could not be downloaded from either source.
    """
    if is_valid_file(destination):
        print(f"File {destination} already exists, skipping download.")
        return

    try:
        download_url(primary_url, destination, max_retries=max_retries)
        return
    except ValueError:
        pass

    print(f"\nPrimary URL failed. Falling back to Google Drive for {os.path.basename(destination)} ...")
    download_file_from_google_drive(fallback_gdrive_id, destination, max_retries=max_retries)


def _download_file_from_google_drive(file_id: str, destination: str) -> None:
    """Download the data from the Google drive public URL.

    Uses gdown which handles Google Drive's virus-scan confirmation pages,
    cookies, and connection resets robustly.

    Args:
        file_id: File ID of Google drive URL.
        destination: Destination path where the data needs to be stored.
    """
    gdown.download(id=file_id, output=destination, quiet=False, resume=True, fuzzy=True)


def download_file_from_google_drive(file_id: str, destination: str, max_retries: int = 3) -> None:
    """Download the data from the Google drive public URL.

    This method will try to download the file for given number of retries till successful.

    Args:
        file_id: File ID of Google drive URL.
        destination: Destination path where the data needs to be stored.
        max_retries: max number of retries.
    """
    if is_valid_file(destination):
        print(f"File {destination} already exists, skipping download.")
        return

    for attempt in range(max_retries):
        if is_valid_file(destination):
            return
        if attempt > 0:
            # Exponential backoff with jitter; base of 30s gives Google throttling windows time to clear
            wait = (2**(attempt - 1)) * random.uniform(30, 60)
            time.sleep(wait)
        if is_valid_file(destination):
            # Check again in case some other thread came through and downloaded while you were sleeping
            return
        try:
            _download_file_from_google_drive(file_id=file_id, destination=destination)
            return
        except Exception as e:
            print(f"Exception occurred while downloading {destination} (attempt {attempt + 1}/{max_retries}): {e}")
    raise ValueError(f"Couldn't download {destination} after {max_retries} retries.")

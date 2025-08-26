"""Google Drive API integration for fetching PDF documents."""

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger(__name__)

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GoogleDriveClient:
    """Client for interacting with Google Drive API."""

    def __init__(self, credentials_path: str, token_path: Optional[str] = None):
        """
        Initialize Google Drive client.

        Args:
            credentials_path: Path to Google Drive API credentials JSON file
            token_path: Path to store/load OAuth tokens
        """
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path) if token_path else None
        self.service = None
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with Google Drive API."""
        creds = None

        # Load existing token if available
        if self.token_path and self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            if self.token_path:
                self.token_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.token_path, "w") as token:
                    token.write(creds.to_json())

        self.service = build("drive", "v3", credentials=creds)
        logger.info("Successfully authenticated with Google Drive API")

    def list_files_in_folder(
        self, folder_id: str, file_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        List files in a specific Google Drive folder.

        Args:
            folder_id: Google Drive folder ID
            file_types: List of file MIME types to filter (e.g., ['application/pdf'])

        Returns:
            List of file metadata dictionaries
        """
        if file_types is None:
            file_types = ["application/pdf"]

        query = f"'{folder_id}' in parents and trashed=false"
        if file_types:
            mime_conditions = " or ".join([f"mimeType='{ft}'" for ft in file_types])
            query += f" and ({mime_conditions})"

        try:
            results = (
                self.service.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType, size, webViewLink, modifiedTime)",
                )
                .execute()
            )
            files = results.get("files", [])
            logger.info(f"Found {len(files)} files in folder {folder_id}")
            return files

        except Exception as e:
            logger.error(f"Error listing files in folder {folder_id}: {e}")
            raise

    def download_file(self, file_id: str, output_path: Optional[Path] = None) -> bytes:
        """
        Download a file from Google Drive.

        Args:
            file_id: Google Drive file ID
            output_path: Optional path to save the file

        Returns:
            File content as bytes
        """
        try:
            request = self.service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)

            done = False
            while done is False:
                status, done = downloader.next_chunk()
                logger.debug(f"Download progress: {int(status.progress() * 100)}%")

            file_bytes = file_content.getvalue()

            # Save to file if output path provided
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(file_bytes)
                logger.info(f"File saved to {output_path}")

            return file_bytes

        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
            raise

    def get_file_metadata(self, file_id: str) -> Dict:
        """
        Get metadata for a specific file.

        Args:
            file_id: Google Drive file ID

        Returns:
            File metadata dictionary
        """
        try:
            file_metadata = (
                self.service.files()
                .get(
                    fileId=file_id,
                    fields="id, name, mimeType, size, webViewLink, modifiedTime, parents",
                )
                .execute()
            )
            return file_metadata

        except Exception as e:
            logger.error(f"Error getting metadata for file {file_id}: {e}")
            raise

    def fetch_pdfs_from_folder(self, folder_id: str) -> List[Dict]:
        """
        Fetch all PDF files from a Google Drive folder.

        Args:
            folder_id: Google Drive folder ID

        Returns:
            List of dictionaries with file metadata and content
        """
        files = self.list_files_in_folder(folder_id, ["application/pdf"])
        pdf_data = []

        for file_info in files:
            try:
                logger.info(f"Downloading {file_info['name']}")
                content = self.download_file(file_info["id"])

                pdf_data.append(
                    {
                        "id": file_info["id"],
                        "name": file_info["name"],
                        "content": content,
                        "size": int(file_info.get("size", 0)),
                        "url": file_info.get("webViewLink", ""),
                        "modified_time": file_info.get("modifiedTime", ""),
                        "mime_type": file_info["mimeType"],
                    }
                )

            except Exception as e:
                logger.error(f"Failed to download {file_info['name']}: {e}")
                continue

        logger.info(f"Successfully fetched {len(pdf_data)} PDF files")
        return pdf_data

#!/usr/bin/env python3
"""Create a public Google Drive client for our RAG system."""

import requests
import re
import logging
from typing import Dict, List

class PublicGoogleDriveClient:
    """Client for accessing public Google Drive folders without authentication."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_file_ids_from_folder(self, folder_url: str) -> List[str]:
        """Extract file IDs from public Google Drive folder."""
        try:
            response = requests.get(folder_url, timeout=30)
            if response.status_code != 200:
                self.logger.error(f"Cannot access folder: {response.status_code}")
                return []
            
            # Extract file IDs using regex
            file_id_pattern = r'data-id="([a-zA-Z0-9_-]{28,})"'
            file_ids = re.findall(file_id_pattern, response.text)
            
            self.logger.info(f"Found {len(file_ids)} file IDs in public folder")
            return file_ids
            
        except Exception as e:
            self.logger.error(f"Error extracting file IDs: {e}")
            return []
    
    def get_file_metadata_public(self, file_id: str) -> Dict:
        """Get basic metadata for a public file."""
        # For now, return minimal metadata
        # In a real implementation, we'd extract more details from the folder page
        return {
            "id": file_id,
            "name": f"document_{file_id}.pdf",
            "mimeType": "application/pdf",
            "size": "unknown",
            "webViewLink": f"https://drive.google.com/file/d/{file_id}/view"
        }
    
    def download_public_file(self, file_id: str) -> bytes:
        """Download a file from public Google Drive."""
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        try:
            response = requests.get(download_url, timeout=60, allow_redirects=True)
            
            if response.status_code == 200 and len(response.content) > 1000:
                if response.content.startswith(b'%PDF'):
                    self.logger.info(f"Successfully downloaded PDF {file_id} ({len(response.content)} bytes)")
                    return response.content
                else:
                    self.logger.warning(f"Downloaded content for {file_id} is not a PDF")
            else:
                self.logger.error(f"Failed to download {file_id}: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error downloading {file_id}: {e}")
        
        return b""
    
    def fetch_pdfs_from_public_folder(self, folder_id: str) -> List[Dict]:
        """Fetch all PDFs from a public Google Drive folder."""
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        
        # Extract file IDs
        file_ids = self.extract_file_ids_from_folder(folder_url)
        
        if not file_ids:
            self.logger.warning("No file IDs found in public folder")
            return []
        
        pdf_data = []
        
        for file_id in file_ids:
            try:
                self.logger.info(f"Downloading file {file_id}")
                content = self.download_public_file(file_id)
                
                if content:
                    metadata = self.get_file_metadata_public(file_id)
                    
                    pdf_data.append({
                        "id": file_id,
                        "name": metadata["name"],
                        "content": content,
                        "size": len(content),
                        "url": metadata["webViewLink"],
                        "modified_time": "",
                        "mime_type": "application/pdf",
                    })
                    
            except Exception as e:
                self.logger.error(f"Failed to process file {file_id}: {e}")
                continue
        
        self.logger.info(f"Successfully fetched {len(pdf_data)} PDFs from public folder")
        return pdf_data

# Test the client
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    client = PublicGoogleDriveClient()
    folder_id = "1h6GptTW3DPCdhu7q5tY-83CXrpV8TmY_"
    
    print(f"Testing PublicGoogleDriveClient with folder: {folder_id}")
    
    # Test with just the first file to avoid downloading everything
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    file_ids = client.extract_file_ids_from_folder(folder_url)
    
    if file_ids:
        print(f"✅ Found {len(file_ids)} files")
        
        # Test download of first file
        test_file_id = file_ids[0]
        content = client.download_public_file(test_file_id)
        
        if content:
            print(f"✅ Successfully downloaded test file: {len(content)} bytes")
        else:
            print("❌ Failed to download test file")
    else:
        print("❌ No files found")

#!/usr/bin/env python3
"""Test script to check if we can download PDFs from public Google Drive folder."""

import requests
import sys

def test_download_public_pdf():
    """Test downloading a PDF from the public Google Drive folder."""
    
    # Let's try to download one of the smaller PDFs first
    # "Docker.pdf" - 344 KB - should be quick to test
    
    # The file IDs would normally be extracted from the folder listing
    # For now, let's try the direct download approach
    folder_url = "https://drive.google.com/drive/folders/1h6GptTW3DPCdhu7q5tY-83CXrpV8TmY_"
    
    print(f"Testing public Google Drive folder access...")
    print(f"Folder URL: {folder_url}")
    
    # Try to access the folder page to see if it's truly public
    try:
        response = requests.get(folder_url, timeout=10)
        print(f"Folder access status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Folder is accessible!")
            print(f"Response content length: {len(response.content)} bytes")
            
            # Check if we can see file information in the response
            if "Accounting Basics" in response.text:
                print("✅ Can see file names in folder!")
            else:
                print("❌ Cannot see file details - may need authentication")
                
        else:
            print(f"❌ Cannot access folder: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error accessing folder: {e}")

    # Test Google Drive API v3 for public access
    print("\nTesting Google Drive API v3 for public folder...")
    api_url = "https://www.googleapis.com/drive/v3/files"
    folder_id = "1h6GptTW3DPCdhu7q5tY-83CXrpV8TmY_"
    
    params = {
        "q": f"'{folder_id}' in parents and trashed=false",
        "fields": "files(id,name,size,mimeType)"
    }
    
    try:
        api_response = requests.get(api_url, params=params, timeout=10)
        print(f"API response status: {api_response.status_code}")
        
        if api_response.status_code == 200:
            files = api_response.json().get("files", [])
            print(f"✅ Found {len(files)} files via API!")
            for file in files[:3]:  # Show first 3 files
                print(f"  - {file.get('name', 'Unknown')} ({file.get('size', 'Unknown')} bytes)")
        else:
            print(f"❌ API access failed: {api_response.status_code}")
            print(f"Response: {api_response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ API error: {e}")

if __name__ == "__main__":
    test_download_public_pdf()

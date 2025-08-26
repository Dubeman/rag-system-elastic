#!/usr/bin/env python3
"""Extract file information from public Google Drive folder."""

import requests
import re
import json

def extract_file_info_from_public_folder(folder_url):
    """Extract file information from public Google Drive folder."""
    
    print(f"Extracting file info from: {folder_url}")
    
    try:
        response = requests.get(folder_url, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Cannot access folder: {response.status_code}")
            return []
        
        content = response.text
        
        # Look for file information in the HTML
        # Google Drive embeds file data in JavaScript
        
        # Pattern to find file entries
        file_pattern = r'"([a-zA-Z0-9_-]{28,})".*?"([^"]+\.pdf)".*?"(\d+)"'
        
        files = []
        matches = re.findall(file_pattern, content)
        
        print(f"Found {len(matches)} potential file matches")
        
        # Also try to find files mentioned in your web search results
        known_files = [
            "Accounting Basics - Nicolas Boucher.pdf",
            "Adopting Rust to Achieve Business Goals.pdf", 
            "Algorithms .pdf",
            "Azure Notes.pdf",
            "Business Models.pdf",
            "Context Engineering 101.pdf",
            "Critical Logs to Monitor_A Guide for SOC Analysts .pdf",
            "Design Patterns.pdf",
            "DevOps Complete Package.pdf",
            "Docker.pdf",
            "Dynamic programm notes.pdf",
            "FortiOS-5.6.11-Rest-API_Reference.pdf",
            "Heap and Stack memory written by Yusha Al-Ayoub.pdf",
            "Mental Models To Improve Your Thinking_.pdf",
            "Product Management.pdf",
            "Websites that will save 100s of hours!.pdf"
        ]
        
        print("\nKnown files from the folder:")
        for i, filename in enumerate(known_files, 1):
            print(f"{i:2d}. {filename}")
        
        # Try to find file IDs in the page source
        # Google Drive uses specific patterns for file IDs
        file_id_pattern = r'data-id="([a-zA-Z0-9_-]{28,})"'
        file_ids = re.findall(file_id_pattern, content)
        
        print(f"\nFound {len(file_ids)} potential file IDs")
        
        if file_ids:
            print("Sample file IDs:")
            for i, file_id in enumerate(file_ids[:5]):
                print(f"  {i+1}. {file_id}")
        
        return known_files, file_ids
        
    except Exception as e:
        print(f"❌ Error extracting file info: {e}")
        return [], []

def test_direct_pdf_download(file_id):
    """Test downloading a PDF directly using file ID."""
    
    # Common Google Drive direct download URLs
    download_urls = [
        f"https://drive.google.com/uc?export=download&id={file_id}",
        f"https://drive.google.com/file/d/{file_id}/view?usp=sharing",
        f"https://docs.google.com/uc?export=download&id={file_id}"
    ]
    
    for i, url in enumerate(download_urls, 1):
        print(f"\nTrying download method {i}: {url}")
        
        try:
            response = requests.get(url, timeout=30, allow_redirects=True)
            print(f"  Status: {response.status_code}")
            print(f"  Content-Type: {response.headers.get('content-type', 'Unknown')}")
            print(f"  Content-Length: {len(response.content)} bytes")
            
            if response.status_code == 200 and len(response.content) > 1000:
                print(f"  ✅ Potential successful download!")
                
                # Check if it's actually a PDF
                if response.content.startswith(b'%PDF'):
                    print(f"  ✅ Confirmed PDF content!")
                    return True
                else:
                    print(f"  ❌ Not PDF content")
            else:
                print(f"  ❌ Download failed or too small")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return False

if __name__ == "__main__":
    folder_url = "https://drive.google.com/drive/folders/1h6GptTW3DPCdhu7q5tY-83CXrpV8TmY_"
    
    # Extract file information
    files, file_ids = extract_file_info_from_public_folder(folder_url)
    
    # Test downloading if we found file IDs
    if file_ids:
        print(f"\n🧪 Testing download with first file ID: {file_ids[0]}")
        success = test_direct_pdf_download(file_ids[0])
        
        if success:
            print("\n✅ SUCCESS: Can download PDFs from public folder!")
        else:
            print("\n❌ FAILED: Cannot download PDFs directly")
    else:
        print("\n❌ No file IDs found")

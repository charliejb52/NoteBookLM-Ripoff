"""
SEC 8-K Filing Download Script for Apple (AAPL)

This script downloads recent 8-K filings (current reports) for Apple Inc. (AAPL)
from the SEC EDGAR database using the sec-edgar-downloader library.

8-K filings are current reports that companies must file within 4 business days
of certain significant events, such as:
- Quarterly earnings announcements
- Press releases about material events
- Changes in management
- Acquisition or disposition of assets
- Changes in financial condition

These filings are useful for RAG systems as they contain recent, event-driven
information about the company.
"""

from sec_edgar_downloader import Downloader
import os
from pathlib import Path
from datetime import datetime


def download_apple_8k_filings(count=20, output_dir="data/apple_8k_filings"):
    """
    Download recent 8-K filings for Apple Inc. (AAPL).
    
    Args:
        count (int): Number of recent 8-K filings to download (default: 20)
        output_dir (str): Directory to save the filings (default: "data/apple_8k_filings")
        
    Returns:
        str: Path to the output directory where filings were saved
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the SEC EDGAR downloader
    # SEC requires a company name and email for identification purposes
    # These are used to identify who is accessing the SEC database
    # The download_folder parameter specifies where to save the filings
    downloader = Downloader(
        company_name="Local RAG Project",  # Dummy company name
        email_address="user@example.com",  # Dummy email address
        download_folder=output_dir         # Directory to save filings
    )
    
    print(f"Downloading {count} most recent 8-K filings for AAPL...")
    print(f"Output directory: {output_dir}")
    print()
    
    # Download 8-K filings for Apple (ticker: AAPL)
    # 8-K filings are current reports filed for significant events
    try:
        downloader.get(
            "8-K",              # Form type (current report)
            "AAPL",             # Ticker symbol
            limit=count,        # Number of filings to download
            download_details=True  # Download detailed filing information
        )
        
        print(f"✓ Filings downloaded successfully!")
        print(f"  Location: {output_dir}")
        print()
        
        return output_dir
        
    except Exception as e:
        print(f"Error downloading filings: {e}")
        raise


def find_latest_filing_files(directory="data/apple_8k_filings", n=5):
    """
    Find the last N 8-K filing files in the directory, sorted by date.
    
    Args:
        directory (str): Directory containing the downloaded filings
        n (int): Number of files to return (default: 5)
        
    Returns:
        list: List of file paths, sorted by date (most recent first)
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    filing_files = []
    
    # Walk through the directory structure
    # sec-edgar-downloader typically organizes files as:
    # directory/sec-edgar-filings/AAPL/8-K/YYYY-MM-DD/filing.html
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Look for .htm, .html, or .txt files
            # Prefer .txt files (full-submission.txt) as they contain complete content
            if file.endswith(('.htm', '.html', '.txt')):
                file_path = os.path.join(root, file)
                
                # Try to extract date from directory structure
                # This is more reliable than file modification time
                path_parts = Path(file_path).parts
                filing_date = None
                
                for part in path_parts:
                    # Check if this part looks like a date (YYYY-MM-DD format)
                    try:
                        filing_date = datetime.strptime(part, "%Y-%m-%d")
                        break
                    except ValueError:
                        continue
                
                # If we found a date in the path, use it for comparison
                # Otherwise, fall back to file modification time
                if filing_date:
                    filing_files.append((filing_date, file_path))
                else:
                    # Fallback: use file modification time if no date in path
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    filing_files.append((file_mtime, file_path))
    
    # Sort by date (most recent first) and return top N
    filing_files.sort(key=lambda x: x[0], reverse=True)
    return [file_path for _, file_path in filing_files[:n]]


def find_latest_filing_file(directory="data/apple_8k_filings"):
    """
    Locate the latest .htm or .txt file in the downloaded 8-K filings directory.
    
    This function searches through the directory structure created by sec-edgar-downloader
    and finds the most recent filing file (either .htm or .txt format) based on the
    filing date in the directory structure, which is more reliable than file modification time.
    
    Args:
        directory (str): Directory containing the downloaded filings
        
    Returns:
        str: Path to the latest filing file, or None if no files found
    """
    files = find_latest_filing_files(directory, n=1)
    if files:
        latest_file = files[0]
        # Extract date for display
        path_parts = Path(latest_file).parts
        for part in path_parts:
            try:
                filing_date = datetime.strptime(part, "%Y-%m-%d")
                print(f"Latest filing date: {filing_date.strftime('%Y-%m-%d')}")
                break
            except ValueError:
                continue
        print(f"Latest filing file: {latest_file}")
        return latest_file
    else:
        print("No .htm, .html, or .txt files found in the directory.")
        return None


if __name__ == "__main__":
    try:
        # Download recent 8-K filings (default: 20 most recent)
        # 8-K filings are filed more frequently than 10-K, so we download by count, not years
        import sys
        
        count = 20  # Default to 20 most recent filings
        if len(sys.argv) > 1:
            try:
                count = int(sys.argv[1])
            except ValueError:
                print(f"Invalid count: {sys.argv[1]}. Using default: {count}")
        
        output_directory = download_apple_8k_filings(count=count, output_dir="data/apple_8k_filings")
        
        print()
        print("=" * 60)
        print("Finding Latest Filing")
        print("=" * 60)
        
        # Find and display the latest filing file
        latest_file = find_latest_filing_file(output_directory)
        
        if latest_file:
            print()
            print("=" * 60)
            print("Ready for Ingestion")
            print("=" * 60)
            print(f"You can use this file path in your ingestion script:")
            print(f"  {latest_file}")
            print()
            print("To find multiple recent filings, use:")
            print(f"  from download_apple_8k import find_latest_filing_files")
            print(f"  files = find_latest_filing_files('{output_directory}', n=5)")
        else:
            print("\nWarning: Could not locate a filing file.")
            print("Please check the download directory manually.")
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


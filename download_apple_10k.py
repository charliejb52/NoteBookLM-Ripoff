"""
SEC 10-K Filing Download Script for Apple (AAPL)

This script downloads the last 5 years of 10-K filings for Apple Inc. (AAPL)
from the SEC EDGAR database using the sec-edgar-downloader library.

10-K filings are annual reports that contain comprehensive information about
a company's financial performance, business operations, and risk factors.
"""

from sec_edgar_downloader import Downloader
import os
from pathlib import Path
from datetime import datetime


def download_apple_10k_filings(years=5, output_dir="data/apple_filings"):
    """
    Download the last N years of 10-K filings for Apple Inc. (AAPL).
    
    Args:
        years (int): Number of years of filings to download (default: 5)
        output_dir (str): Directory to save the filings (default: "data/apple_filings")
        
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
    
    print(f"Downloading last {years} years of 10-K filings for AAPL...")
    print(f"Output directory: {output_dir}")
    print()
    
    # Download 10-K filings for Apple (ticker: AAPL)
    # The downloader will automatically fetch filings from the last N years
    try:
        downloader.get(
            "10-K",           # Form type (filing type)
            "AAPL",           # Ticker symbol
            limit=years,      # Number of filings to download
            download_details=True  # Download detailed filing information
        )
        
        print(f"✓ Filings downloaded successfully!")
        print(f"  Location: {output_dir}")
        print()
        
        return output_dir
        
    except Exception as e:
        print(f"Error downloading filings: {e}")
        raise


def find_latest_filing_file(directory="data/apple_filings"):
    """
    Locate the latest .htm or .txt file in the downloaded filings directory.
    
    This function searches through the directory structure created by sec-edgar-downloader
    and finds the most recent filing file (either .htm or .txt format) based on the
    filing date in the directory structure, which is more reliable than file modification time.
    
    Args:
        directory (str): Directory containing the downloaded filings
        
    Returns:
        str: Path to the latest filing file, or None if no files found
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    latest_file = None
    latest_filing_date = None
    
    # Walk through the directory structure
    # sec-edgar-downloader typically organizes files as:
    # directory/sec-edgar-downloads/sec-edgar-downloads/AAPL/10-K/YYYY-MM-DD/filing.html
    # or simply: directory/AAPL/10-K/YYYY-MM-DD/filing.html
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Look for .htm, .html, or .txt files
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
                    if latest_filing_date is None or filing_date > latest_filing_date:
                        latest_filing_date = filing_date
                        latest_file = file_path
                else:
                    # Fallback: use file modification time if no date in path
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if latest_filing_date is None or file_mtime > latest_filing_date:
                        latest_filing_date = file_mtime
                        latest_file = file_path
    
    if latest_file:
        if latest_filing_date:
            print(f"Latest filing date: {latest_filing_date.strftime('%Y-%m-%d')}")
        print(f"Latest filing file: {latest_file}")
        return latest_file
    else:
        print("No .htm, .html, or .txt files found in the directory.")
        return None


if __name__ == "__main__":
    try:
        # Download the last 5 years of 10-K filings
        output_directory = download_apple_10k_filings(years=5, output_dir="data/apple_filings")
        
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
        else:
            print("\nWarning: Could not locate a filing file.")
            print("Please check the download directory manually.")
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


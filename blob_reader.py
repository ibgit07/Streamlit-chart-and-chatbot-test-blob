from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import StringIO

def get_latest_csv_df(connection_string, container_name, folder_path):
    """
    Fetch the newest CSV file from a given Azure Blob folder
    and return it as a Pandas DataFrame.
    """

    # Connect to Azure Blob Storage
    service = BlobServiceClient.from_connection_string(connection_string)
    container_client = service.get_container_client(container_name)

    # List only CSV files
    csv_blobs = [
        blob for blob in container_client.list_blobs(name_starts_with=folder_path)
        if blob.name.lower().endswith(".csv")
    ]

    if not csv_blobs:
        raise Exception("No CSV files found in the folder.")

    # Sort by last modified (newest first)
    csv_blobs.sort(key=lambda b: b.last_modified, reverse=True)

    # Pick the newest CSV file
    newest_blob = csv_blobs[0]
    print(f"Latest CSV File: {newest_blob.name}")

    # Download CSV content
    blob_client = container_client.get_blob_client(newest_blob.name)
    csv_text = blob_client.download_blob().content_as_text()

    # Convert to DataFrame
    df = pd.read_csv(StringIO(csv_text))
    print("file Downloaded and converted to DataFrame")
    return df

import os
import io
import logging
from typing import Optional
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from app.config import settings

logger = logging.getLogger("chatbot")

class DriveService:
    def __init__(self):
        self.credentials_path = settings.GD_CREDENTIALS_PATH
        self.folder_id = settings.GD_FOLDER_ID
        self.service = None
        
        # Initialize Google Drive client if credentials and folder ID exist
        if os.path.exists(self.credentials_path) and self.folder_id:
            try:
                creds = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=['https://www.googleapis.com/auth/drive.readonly']
                )
                self.service = build('drive', 'v3', credentials=creds)
                logger.info("Google Drive service successfully initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Drive API: {e}")
        else:
            logger.warning(
                f"Google Drive Service is currently unconfigured. Missing either "
                f"credentials file '{self.credentials_path}' or GD_FOLDER_ID in environment."
            )

    def fetch_pyq_file(self, subject_name: str, year: str) -> Optional[str]:
        """
        Searches the configured Google Drive folder for a PDF file that matches
        the given subject name and year.
        If found:
          - Downloads it and saves it in static/pyqs/
          - Returns the relative URL '/static/pyqs/<filename>' for the client
        If not found or service unconfigured:
          - Returns None
        """
        if not self.service or not self.folder_id:
            return None
            
        # Refined query: find a PDF within the parent folder that contains the subject & year in name
        clean_subject = subject_name.replace("'", "\\'")
        query = (
            f"'{self.folder_id}' in parents and "
            f"mimeType = 'application/pdf' and "
            f"name contains '{clean_subject}' and "
            f"name contains '{year}' and "
            f"trashed = false"
        )
        
        try:
            results = self.service.files().list(
                q=query, 
                spaces='drive', 
                fields='files(id, name)',
                pageSize=1
            ).execute()
            items = results.get('files', [])
            
            if not items:
                logger.info(f"No matching file found on Google Drive for '{subject_name}' ({year})")
                return None
                
            file_id = items[0]['id']
            file_name = items[0]['name']
            
            # Save it locally inside static/pyqs/
            os.makedirs("static/pyqs", exist_ok=True)
            local_path = os.path.join("static", "pyqs", file_name)
            
            # Download file content if not already cached locally
            if not os.path.exists(local_path):
                logger.info(f"Downloading {file_name} from Google Drive...")
                request = self.service.files().get_media(fileId=file_id)
                fh = io.FileIO(local_path, 'wb')
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                logger.info(f"Successfully cached {file_name} locally.")
                
            return f"/static/pyqs/{file_name}"
            
        except Exception as e:
            logger.error(f"Error searching/downloading file from Google Drive: {e}")
            return None

# Singleton instance of DriveService
drive_service = DriveService()

def get_drive_service() -> DriveService:
    return drive_service

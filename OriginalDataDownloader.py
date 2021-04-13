from google_drive_downloader import GoogleDriveDownloader as gdd
import os
import shutil

class OriginalDataDownloader:
    def __init__(self):
        self.rawDatasetZipFileId = '1YDH9kuu629EsRdRsf694-XBh8NyOGEqH'
        self.embeddingsZipFileId = '1ABLLvqDYBC-PsNxb9KgviNiSjTB1WG5S'

    def download(self):
        self.__removeDataFolder()

        self.__downloadRawDataset()
        self.__downloadEmbeddings()

    def __downloadRawDataset(self):
        zipPath = 'Data/RawDataset.zip'

        gdd.download_file_from_google_drive(file_id=self.rawDatasetZipFileId,
                                    dest_path=zipPath,
                                    showsize=True,
                                    unzip=True)

        os.remove(zipPath)

    def __downloadEmbeddings(self):
        zipPath = 'Data/Embeddings.zip'

        gdd.download_file_from_google_drive(file_id=self.embeddingsZipFileId,
                                    dest_path=zipPath,
                                    showsize=True,
                                    unzip=True)

        os.remove(zipPath)

    def __removeDataFolder(self):
        if os.path.exists('Data') and os.path.isdir('Data'):
            shutil.rmtree('Data')
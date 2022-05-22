from google_drive_downloader import GoogleDriveDownloader as gdd
import os
import shutil

class OriginalDataDownloader:
    def __init__(self):
        self._rawDatasetZipFileId = '1mSRGL57389is7r5U70b_5yZ6whTeuWz2'
        self._embeddingsZipFileId = '11AUF2HklrWND4_eF18VpMHvYVJTCTyOv'

    def download(self):
        self.__removeDataFolder()

        self.__downloadRawDataset()
        self.__downloadEmbeddings()

    def __downloadRawDataset(self):
        zipPath = 'Data/RawDataset.zip'

        gdd.download_file_from_google_drive(file_id=self._rawDatasetZipFileId,
                                    dest_path=zipPath,
                                    showsize=True,
                                    unzip=True)

        os.remove(zipPath)

    def __downloadEmbeddings(self):
        zipPath = 'Data/Embeddings.zip'

        gdd.download_file_from_google_drive(file_id=self._embeddingsZipFileId,
                                    dest_path=zipPath,
                                    showsize=True,
                                    unzip=True)

        os.remove(zipPath)

    def __removeDataFolder(self):
        if os.path.exists('Data') and os.path.isdir('Data'):
            shutil.rmtree('Data')
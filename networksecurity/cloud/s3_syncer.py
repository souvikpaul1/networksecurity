import os

class s3Sync:
    def sync_folder_to_s3(self, local_folder: str, s3_bucket: str):
        """
        Syncs a local folder to an S3 bucket.

        :param local_folder: Path to the local folder to sync.
        :param s3_bucket: Name of the S3 bucket.
        :param s3_folder: Path in the S3 bucket where the folder will be synced.
        """
        command= f"aws s3 sync {local_folder} {s3_bucket}"
        os.system(command)

    def sync_folder_from_s3(self, s3_bucket: str, local_folder: str):
        """
        Syncs an S3 bucket to a local folder.

        :param s3_bucket: Name of the S3 bucket.
        :param local_folder: Path to the local folder where the bucket will be synced.
        """
        command = f"aws s3 sync {s3_bucket} {local_folder}"
        os.system(command)
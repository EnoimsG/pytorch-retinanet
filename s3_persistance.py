import boto3
from botocore.exceptions import ClientError
import os

class S3:
    def __init__(self):
        self.s3 = boto3.client('s3')

    def put_obj(self, file_name, object_name):
        try:
            response = self.s3.upload_file(file_name, 'tesi-micc-sgori', object_name)
        except ClientError as e:
            print(e)
            return False
        return True

    def put(self, file_name, path, object_name):
        self.put_obj(file_name, os.path.join(path, object_name))
#
# s3 = S3()
# s3.put('README.md', 'simonegori/exp_name/run_name', 'leggimi')
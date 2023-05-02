# coding: utf-8

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
FILE: blob_samples_hello_world.py
DESCRIPTION:
    This sample demos basic blob operations like getting a blob client from container, uploading and downloading
    a blob using the blob_client.
USAGE: python blob_samples_hello_world.py
    Set the environment variables with your own values before running the sample:
    1) AZURE_STORAGE_CONNECTION_STRING - the connection string to your storage account
"""

import os


# set up
SOURCE_FILE = '3_image_sample.png'
AZURE_STORAGE_CONNECTION_STRING = 'con_string'


class BlobSamples(object):

    connection_string = AZURE_STORAGE_CONNECTION_STRING

    # --Begin Blob Samples-----------------------------------------------------------------

    def create_container_sample(self):

        # Instantiate a new BlobServiceClient using a connection string
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string)

        # Instantiate a new ContainerClient
        container_client = blob_service_client.get_container_client(
            "name")

        # Create new container in the service
        container_client.create_container()

        # List containers in the storage account
        list_response = blob_service_client.list_containers()
        print(list_response.next())

        # Delete the container
        # blob_service_client.delete_container('name')

    def block_blob_sample(self):

        # Instantiate a new BlobServiceClient using a connection string
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string)

        # Instantiate a new ContainerClient
        container_client = blob_service_client.get_container_client(
            "name")

        # Create new Container in the service
        # container_client.create_container()

        # Instantiate a new BlobClient
        blob_client = container_client.get_blob_client("mypicture.png")

        # [START upload_a_blob]
        # Upload content to block blob
        with open(SOURCE_FILE, "rb") as data:
            blob_client.upload_blob(data, blob_type="BlockBlob")
        # [END upload_a_blob]

        # [START download_a_blob]
        # with open(DEST_FILE, "wb") as my_blob:
        #     download_stream = blob_client.download_blob()
        #     my_blob.write(download_stream.readall())
        # [END download_a_blob]

        # [START delete_blob]
        # blob_client.delete_blob()
        # [END delete_blob]

        # Delete the container
        # container_client.delete_container()
        # blob_service_client.delete_container('mark')


if __name__ == '__main__':
    sample = BlobSamples()
    # sample.create_container_sample()
    sample.block_blob_sample()

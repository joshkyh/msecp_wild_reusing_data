import os
from azure.storage.blob import BlockBlobService
#from Modelling.Python.util.runsettings import get_runsettings
from zipfile import ZipFile, ZIP_DEFLATED

def download_word_vectors():
    '''
    data.zip contains the training, dev, and test sets as well as the word vectors. We use a Azure Storage Blob because
    of the size of the zip file.
    '''

    block_blob_service = BlockBlobService(account_name='eqclinicblob',
                                          account_key='BLABLA')

    container_name = 'wordvectors'


    # List the blobs in the container
    print("\nList blobs in the container")
    generator = block_blob_service.list_blobs(container_name)
    for blob in generator:
        print("\t Downloading: " + blob.name)
        # Download file is always data.zip, removing the timestamp.
        full_path_to_file = os.path.join('../saved_models/', blob.name)
        block_blob_service.get_blob_to_path(container_name, blob.name, full_path_to_file)

if __name__ =='__main__':
    download_word_vectors()

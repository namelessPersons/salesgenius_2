import type { NextApiRequest, NextApiResponse } from 'next';
import { BlobServiceClient } from '@azure/storage-blob';

const { AZURE_BLOB_CONN_STR, AZURE_BLOB_CONTAINER } = process.env;

const blobService = BlobServiceClient.fromConnectionString(AZURE_BLOB_CONN_STR!);
const containerClient = blobService.getContainerClient(AZURE_BLOB_CONTAINER!);

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const files: string[] = [];
  for await (const blob of containerClient.listBlobsFlat()) {
    files.push(blob.name);
  }
  res.status(200).json(files);
}

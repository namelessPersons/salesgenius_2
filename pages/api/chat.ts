import type { NextApiRequest, NextApiResponse } from 'next';
import { AzureKeyCredential, SearchClient } from '@azure/search-documents';
import { BlobServiceClient } from '@azure/storage-blob';

const {
  AZURE_OPENAI_ENDPOINT,
  AZURE_OPENAI_KEY,
  AZURE_OPENAI_CHAT_DEPLOYMENT,
  AZURE_SEARCH_ENDPOINT,
  AZURE_SEARCH_KEY,
  AZURE_SEARCH_INDEX,
  AZURE_BLOB_CONN_STR,
  AZURE_BLOB_CONTAINER
} = process.env;

const openaiUrl = `${AZURE_OPENAI_ENDPOINT}/openai/deployments/${AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions?api-version=2024-02-15-preview`;

const searchClient = new SearchClient(AZURE_SEARCH_ENDPOINT!, AZURE_SEARCH_INDEX!, new AzureKeyCredential(AZURE_SEARCH_KEY!));
const blobService = BlobServiceClient.fromConnectionString(AZURE_BLOB_CONN_STR!);
const containerClient = blobService.getContainerClient(AZURE_BLOB_CONTAINER!);

async function performSearch(query: string) {
  const results = searchClient.search(query, { select: ['path'] });
  const files: string[] = [];
  for await (const r of results.results) {
    if (r.document.path) files.push(r.document.path);
  }
  return files;
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).end();
  const { message } = req.body;
  const files = await performSearch(message);
  const aiRes = await fetch(openaiUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'api-key': AZURE_OPENAI_KEY!
    },
    body: JSON.stringify({ messages: [{ role: 'user', content: message }] })
  });
  const aiData = await aiRes.json();
  const reply = aiData.choices?.[0]?.message?.content ?? '';
  res.status(200).json({ reply, files });
}

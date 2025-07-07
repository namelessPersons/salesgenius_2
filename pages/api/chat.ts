import type { NextApiRequest, NextApiResponse } from 'next';
import { Configuration, OpenAIApi } from 'openai';
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

const openai = new OpenAIApi(new Configuration({
  apiKey: AZURE_OPENAI_KEY,
  basePath: `${AZURE_OPENAI_ENDPOINT}/openai/deployments/${AZURE_OPENAI_CHAT_DEPLOYMENT}`,
}));

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
  const completion = await openai.createChatCompletion({
    model: AZURE_OPENAI_CHAT_DEPLOYMENT!,
    messages: [{ role: 'user', content: message }]
  });
  const reply = completion.data.choices[0].message?.content ?? '';
  res.status(200).json({ reply, files });
}

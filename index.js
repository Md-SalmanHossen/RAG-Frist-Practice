import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';


import * as dotenv from 'dotenv';
dotenv.config();

async function indexDocument(){

   //step1:

   const PDF_PATH = './dsa.pdf';
   const pdfLoader = new PDFLoader(PDF_PATH);
   const rawDocs = await pdfLoader.load();
   // console.log(rawDocs.length);


  //step2: chungking
   const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
   });
   const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
   //console.log(chunkedDocs);

   //step3: vector embedding model
   const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
   });

   //step4: configure db
   const pinecone = new Pinecone();
   const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

   //step5: langchain(chunking,embedding,database);
   await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
      pineconeIndex,
      maxConcurrency: 5,
   });

   

} 

indexDocument();

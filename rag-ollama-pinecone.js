import { Pinecone } from "@pinecone-database/pinecone";
import { ChatOllama } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"; // तुकडे करण्यासाठी
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import * as fs from "fs/promises";
import * as dotenv from "dotenv";

dotenv.config();

async function runFilePineconeRag() {
  // 1) pinecone setuop and namespace creation
  const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
  const indexName = process.env.INDEX_NAME;
  const indexHost = process.env.INDEX_HOST;
  const namespace = pc
    .index(indexName, indexHost)
    .namespace("project-knowledge");

  try {
    // 2) Read the file content that contains the knowledge you want to RAG with
    const filePath = "./Priyesh_Suryavanshi_CV_2026.pdf";
    console.log(`1. Reading file from path: ${filePath}...`);
    const fileContent = await fs.readFile(filePath, "utf-8");

    // 3) Split the file content into smaller chunks (Chunking)
    console.log("2. Splitting file content into smaller chunks...");
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 300,
      chunkOverlap: 50,
    });
    const chunks = await textSplitter.splitText(fileContent);
    console.log(`Created ${chunks.length} chunks from the file.`);

    // 4) Convert these chunks to Pinecone format and upsert
    console.log("3. Uploading file chunks to Pinecone Database...");
    const recordsToUpsert = chunks.map((chunkText, index) => ({
      _id: `file_chunk_${index}_${Date.now()}`, // युनिक आयडी
      text: chunkText,
      source: filePath,
    }));

    // create batches of records to upsert to avoid overwhelming the Pinecone API
    const BATCH_SIZE = 50;
    for (let i = 0; i < recordsToUpsert.length; i += BATCH_SIZE) {
      const batch = recordsToUpsert.slice(i, i + BATCH_SIZE);
      console.log(
        `Uploading batch ${Math.floor(i / BATCH_SIZE) + 1} (${batch.length} chunks)...`,
      );

      await namespace.upsertRecords(batch);
    }

    // await namespace.upsertRecords(recordsToUpsert);
    console.log("All file data successfully saved in Pinecone!");

    // User's question that we want to answer using RAG
    const userQuery = "what is the email id of priyesh suryavanshi?";
    console.log(`\nUser Question: ${userQuery}`);

    // 5) Convert user query to embeddings using Pinecone's in-built model and search for relevant chunks
    console.log("🔍 Converting query and searching Pinecone...");
    const queryEmbedding = await pc.inference.embed(
      "multilingual-e5-large",
      [userQuery],
      { inputType: "query" },
    );

    let vectorValues = Array.isArray(queryEmbedding)
      ? queryEmbedding[0]?.values
      : queryEmbedding.data[0]?.values;

    const searchResult = await namespace.query({
      vector: vectorValues,
      topK: 1,
      includeMetadata: true,
    });

    if (!searchResult.matches || searchResult.matches.length === 0) {
      console.log("Pinecone database मध्ये कोणतीही मॅच सापडली नाही!");
      return;
    }

    const context = searchResult.matches[0].metadata?.text;
    console.log(`Pinecone found this relevant chunk from file: \n"${context}"`);

    // 6) Use the retrieved context and user query to generate an answer using Ollama
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "You are an expert technical assistant. Answer the user's question ONLY using the provided context from the project document. If you don't know, say 'I don't know'.\n\nContext:\n{context}",
      ],
      ["human", "{question}"],
    ]);

    const model = new ChatOllama({ model: "llama3", temperature: 0.1 });
    const chain = prompt.pipe(model).pipe(new StringOutputParser());

    console.log("\nOllama is generating final answer...");
    const finalAnswer = await chain.stream({
      context: context,
      question: userQuery,
    });

    console.log("\nAI Answer:");
    for await (const chunk of finalAnswer) {
      process.stdout.write(chunk);
    }
  } catch (error) {
    console.error("Error in File RAG Pipeline:", error);
  }
}

runFilePineconeRag();

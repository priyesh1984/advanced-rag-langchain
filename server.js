import express from "express";
import { Pinecone } from "@pinecone-database/pinecone";
import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import path from "path";
import { fileURLToPath } from "url";
import * as dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3001;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// 1) setup pinecone client and namespace
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const indexName = process.env.INDEX_NAME;
const indexHost = process.env.INDEX_HOST;
const namespace = pc.index(indexName, indexHost).namespace("cv-knowledge");

// 2) API endpoint to handle chat requests from frontend
app.post("/api/chat", async (req, res) => {
  const { question } = req.body;

  if (!question) {
    return res.status(400).json({ error: "Question is required" });
  }

  // Server-Sent Events for streaming response
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  try {
    // 3) Convert user query to embeddings using Pinecone's in-built model and search for relevant chunks
    const queryEmbedding = await pc.inference.embed(
      "multilingual-e5-large",
      [question],
      { inputType: "query" },
    );
    let vectorValues = Array.isArray(queryEmbedding)
      ? queryEmbedding[0]?.values
      : queryEmbedding.data[0]?.values;

    const searchResult = await namespace.query({
      vector: vectorValues,
      topK: 6,
      includeMetadata: true,
    });

    // If no relevant chunks found, return an error message
    if (!searchResult.matches || searchResult.matches.length === 0) {
      return res
        .status(404)
        .json({ error: "No relevant content found in CV." });
    }

    // Sort the matched chunks based on their original order in the document (assuming _id format is 'file_chunk_{index}_{timestamp}')
    const sortedMatches = [...searchResult.matches].sort((a, b) => {
      const numA = parseInt(a.id.split("_")[2]) || 0;
      const numB = parseInt(b.id.split("_")[2]) || 0;
      return numA - numB; // चंक्स पहिल्या पानापासून शेवटच्या पानापर्यंत क्रमाने लागतील
    });

    // join the sorted chunks to create the context for the LLM
    const context = sortedMatches
      .map((match) => match.metadata?.text)
      .filter(Boolean)
      .join("\n\n");

    console.log(
      `📌 Pinecone found ${searchResult.matches.length} chunks for context.`,
    );

    // prompt template for the LLM, with clear instructions to format the output in a specific way (bullet points with new lines)
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are an expert technical assistant analyzing Priyesh Suryavanshi's CV.
        
        CRITICAL FORMATTING INSTRUCTIONS:
        - You must format the list of companies using plain text bullet points.
        - Every single company MUST be on a brand new line. Do NOT combine them into a single paragraph.
        - Start each line with a star (*) or dash (-) followed by a space.
        - Example format:
          * Company Name 1 (Dates)
          * Company Name 2 (Dates)
        - Put a clear blank line before the final "Note".
        
        Answer the user's question accurately using ONLY the context provided below.
        
        Context:
        {context}`,
      ],
      ["human", "{question}"],
    ]);

    const model = new ChatOllama({ model: "llama3", temperature: 0.1 });
    const chain = prompt.pipe(model).pipe(new StringOutputParser());

    // Stream the AI's response back to the frontend as it is generated
    const aiStream = await chain.stream({
      context: context,
      question: question,
    });

    for await (const chunk of aiStream) {
      res.write(`data: ${JSON.stringify({ text: chunk })}\n\n`);
    }

    // Once the streaming is done, send a [DONE] signal to the frontend
    res.write("data: [DONE]\n\n");
    res.end();
  } catch (error) {
    console.error("Backend Streaming Error:", error);
    res.write(`data: ${JSON.stringify({ error: "Something went wrong" })}\n\n`);
    res.end();
  }
});

app.listen(PORT, () => {
  console.log(`🚀 AI Server is running on http://localhost:${PORT}`);
});

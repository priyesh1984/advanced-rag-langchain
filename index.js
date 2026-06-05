import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { JsonOutputParser } from "@langchain/core/output_parsers";

async function generateLocalItinerary(city) {
  // set ollama model
  const model = new ChatOllama({
    baseUrl: "http://localhost:11434",
    model: "llama3",
    temperature: 0.7,
  });

  // prompt template
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are an expert travel guide. Give a 2-day travel itinerary for the requested city. " +
        "You must respond ONLY with a raw JSON object. Do not include any markdown formatting, " +
        "do not include ```json blocks, and do not write any introductory or concluding text. " +
        'The response must exactly match this structure: {{"day1": [], "day2": []}}', // <--- इथे {{ आणि }} केले आहे
    ],
    ["human", "Tell me about {city}"],
  ]);

  // output parser
  const parser = new JsonOutputParser();

  // LCL chain
  const chain = prompt.pipe(model).pipe(parser);

  console.log(
    `⏱️ Local AI (Ollama) is generating plan for ${city}... Please wait...`,
  );

  try {
    // Run the chain with the city input
    const response = await chain.invoke({ city });
    console.log("\n✅ Free & Local AI Generated Structured Response:");
    console.log(JSON.stringify(response, null, 2));
  } catch (error) {
    console.error("Error running LangChain with Ollama:", error);
  }
}

// Run the function with a sample city
generateLocalItinerary("Mumbai");

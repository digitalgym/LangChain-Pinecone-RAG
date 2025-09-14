import express from 'express';
import cors from 'cors';
import { config } from 'dotenv';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { PineconeStore } from '@langchain/pinecone';
import { Pinecone } from '@pinecone-database/pinecone';
import { RetrievalQAChain } from 'langchain/chains';
import { PromptTemplate } from '@langchain/core/prompts';
import { BufferMemory } from 'langchain/memory';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

config();

const app = express();
const port = process.env.PORT || 8000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../static'), { index: false }));

// Initialize OpenAI
const llm = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "gpt-4o",
  temperature: 1,
  maxTokens: 500,
});

const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "text-embedding-3-large",
});

// Initialize Pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const indexName = process.env.PINECONE_INDEX_NAME;
let vectorStore;

try {
  const index = pinecone.index(indexName);
  vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
    pineconeIndex: index,
  });
} catch (error) {
  console.error('Error initializing Pinecone vector store:', error);
}

// Create custom prompt template
const promptTemplate = new PromptTemplate({
  template: `You are an assistant for question-answering tasks. 
  Use the following pieces of retrieved context to answer the question. 
  If you don't know the answer, just say that you don't know. 
  Use three sentences maximum and keep the answer concise.
  
  Context: {context}
  Question: {question}
  
  Answer:`,
  inputVariables: ["context", "question"],
});

// In-memory session storage for conversation memory
const sessions = new Map();

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../static/index.html'));
});

app.post('/api/chat', async (req, res) => {
  try {
    const { message, session_id = 'default' } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }
    
    if (!vectorStore) {
      return res.status(500).json({ error: 'Vector store not initialized' });
    }
    
    // Initialize session memory if it doesn't exist
    if (!sessions.has(session_id)) {
      sessions.set(session_id, new BufferMemory({
        memoryKey: "chat_history",
        returnMessages: true,
      }));
    }
    
    const memory = sessions.get(session_id);
    
    // Create retrieval chain
    const chain = RetrievalQAChain.fromLLM(llm, vectorStore.asRetriever(3), {
      prompt: promptTemplate,
      memory: memory,
    });
    
    // Get response
    const result = await chain.call({
      query: message,
    });
    
    res.json({
      response: result.text,
      session_id: session_id
    });
    
  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'healthy' });
});

app.delete('/api/chat/:session_id', (req, res) => {
  const { session_id } = req.params;
  if (sessions.has(session_id)) {
    sessions.delete(session_id);
  }
  res.json({ message: 'Chat history cleared' });
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(port, () => {
  console.log(`LangChain.js server running at http://localhost:${port}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
});

export default app;

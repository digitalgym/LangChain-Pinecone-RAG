import express from 'express';
import cors from 'cors';
import { config } from 'dotenv';
import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';
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

// Serve static files with proper MIME types
app.use('/static', express.static(path.join(__dirname, '../static'), {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.css')) {
      res.setHeader('Content-Type', 'text/css');
    } else if (filePath.endsWith('.js')) {
      res.setHeader('Content-Type', 'application/javascript');
    } else if (filePath.endsWith('.html')) {
      res.setHeader('Content-Type', 'text/html');
    }
  }
}));

// Also serve static files from root for direct access
app.use(express.static(path.join(__dirname, '../static')));

// Initialize OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Initialize Pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const indexName = process.env.PINECONE_INDEX_NAME;
let index;

try {
  index = pinecone.index(indexName);
} catch (error) {
  console.error('Error initializing Pinecone index:', error);
}

// In-memory session storage
const sessions = new Map();

// Helper function to get embeddings
async function getEmbeddings(text) {
  try {
    const response = await openai.embeddings.create({
      model: "text-embedding-3-large",
      input: text,
    });
    return response.data[0].embedding;
  } catch (error) {
    console.error('Error getting embeddings:', error);
    throw new Error(`Error getting embeddings: ${error.message}`);
  }
}

// Helper function to query Pinecone
async function queryPinecone(queryText, k = 3) {
  try {
    // Get embeddings for the query
    const queryEmbedding = await getEmbeddings(queryText);
    
    // Query Pinecone
    const results = await index.query({
      vector: queryEmbedding,
      topK: k,
      includeMetadata: true,
    });
    
    // Extract text content from results
    const docs = results.matches
      .filter(match => match.metadata && match.metadata.text)
      .map(match => match.metadata.text);
    
    return docs;
  } catch (error) {
    console.error('Error querying Pinecone:', error);
    throw new Error(`Error querying Pinecone: ${error.message}`);
  }
}

// Helper function to chat with OpenAI
async function chatWithOpenAI(messages) {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: messages,
      temperature: 1,
      max_tokens: 500,
    });
    return response.choices[0].message.content;
  } catch (error) {
    console.error('Error with OpenAI:', error);
    throw new Error(`Error with OpenAI: ${error.message}`);
  }
}

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
    
    // Initialize session if it doesn't exist
    if (!sessions.has(session_id)) {
      sessions.set(session_id, [
        { role: "system", content: "You are an assistant for question-answering tasks." }
      ]);
    }
    
    const sessionMessages = sessions.get(session_id);
    
    // Add user message to session
    sessionMessages.push({ role: "user", content: message });
    
    // Query Pinecone for relevant documents
    const docs = await queryPinecone(message);
    const docsText = docs.join('\n');
    
    // Create system prompt with context
    const systemPrompt = `You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Context: ${docsText}`;
    
    // Prepare messages for OpenAI
    const messages = [...sessionMessages, { role: "system", content: systemPrompt }];
    
    // Get response from OpenAI
    const result = await chatWithOpenAI(messages);
    
    // Add AI response to session
    sessionMessages.push({ role: "assistant", content: result });
    
    res.json({
      response: result,
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
  console.log(`Server running at http://localhost:${port}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
});

export default app;

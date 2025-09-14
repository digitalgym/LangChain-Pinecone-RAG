# Node.js RAG Implementation

This directory contains Node.js implementations of the RAG (Retrieval-Augmented Generation) system, offering significant advantages over the Python version.

## Bundle Size Comparison

| Implementation | Bundle Size | Dependencies | Files |
|---------------|-------------|--------------|-------|
| Python (Full LangChain) | ~200-300MB | 99 packages | - |
| Python (Lightweight) | ~15-20MB | 7 packages | - |
| **Node.js (Direct API)** | **~9MB** | **5 packages** | **~3K files** |
| **Node.js (LangChain.js)** | **~15MB** | **7 packages** | **~4K files** |

## Available Implementations

### 1. Direct API Approach (Recommended for Serverless)
- **File**: `api/index.js`
- **Package**: `package.json`
- **Dependencies**: OpenAI + Pinecone clients only
- **Size**: ~9MB total
- **Best for**: Serverless deployments, minimal overhead

### 2. LangChain.js Approach
- **File**: `api/index-langchain.js`
- **Package**: `package-langchain.json`
- **Dependencies**: Full LangChain.js ecosystem
- **Size**: ~15MB total
- **Best for**: Complex chains, advanced features

## Quick Start

### Option 1: Direct API (Lightweight)
```bash
# Copy package.json to use this approach
cp package.json package.json.backup
npm install
npm run dev
```

### Option 2: LangChain.js (Full-featured)
```bash
# Copy package-langchain.json to use this approach
cp package-langchain.json package.json
npm install
npm run dev
```

## Environment Variables
Create a `.env` file with:
```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
PORT=8000
```

## API Endpoints

- `GET /` - Serve the web interface
- `POST /api/chat` - Chat with RAG system
- `GET /api/health` - Health check
- `DELETE /api/chat/:session_id` - Clear chat history

## Advantages of Node.js Implementation

1. **Smaller Bundle**: 4-8x reduction in size vs Python
2. **Faster Cold Starts**: Better serverless performance
3. **Lower Costs**: Reduced infrastructure requirements
4. **Framework Support**: Both LangChain.js and direct API approaches
5. **Modern Ecosystem**: Native async/await, ES modules

## Framework Equivalents

| Python | Node.js | Status |
|--------|---------|--------|
| LangChain | LangChain.js | ✅ Full parity |
| Semantic Kernel | kerneljs.com | ⚠️ Unofficial port |
| Direct OpenAI/Pinecone | Direct OpenAI/Pinecone | ✅ Same APIs |

## Deployment

The Node.js implementation is optimized for:
- Vercel Functions
- AWS Lambda
- Netlify Functions
- Docker containers
- Traditional Node.js servers

Choose the direct API approach for maximum serverless compatibility and minimum cold start times.

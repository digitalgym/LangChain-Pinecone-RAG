# RAG Chatbot - Vercel Deployment

This is a Vercel-compatible version of the Streamlit RAG chatbot, built with FastAPI and a modern web interface.

## Files Structure

```
├── api/
│   └── index.py          # FastAPI backend
├── static/
│   ├── index.html        # Frontend HTML
│   ├── style.css         # Styling
│   └── script.js         # JavaScript functionality
├── vercel.json           # Vercel configuration
├── requirements-vercel.txt # Python dependencies for Vercel
└── .env.example          # Environment variables template
```

## Deployment Steps

### 1. Environment Variables
Create a `.env` file or set these environment variables in Vercel:

```bash
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_pinecone_index_name_here
```

### 2. Deploy to Vercel

#### Option A: Vercel CLI
```bash
npm i -g vercel
vercel --prod
```

#### Option B: GitHub Integration
1. Push your code to GitHub
2. Connect your repository to Vercel
3. Set environment variables in Vercel dashboard
4. Deploy automatically

### 3. Configure Environment Variables in Vercel
1. Go to your Vercel project dashboard
2. Navigate to Settings → Environment Variables
3. Add the required environment variables:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY` 
   - `PINECONE_INDEX_NAME`

## Key Differences from Streamlit Version

1. **Backend**: FastAPI instead of Streamlit
2. **Frontend**: Custom HTML/CSS/JS interface
3. **Session Management**: In-memory storage (consider Redis for production)
4. **API Endpoints**:
   - `POST /api/chat` - Send messages
   - `DELETE /api/chat/{session_id}` - Clear chat history
   - `GET /api/health` - Health check

## Features

- ✅ Modern, responsive web interface
- ✅ Real-time chat functionality
- ✅ RAG-powered responses using Pinecone and OpenAI
- ✅ Session management
- ✅ Clear chat history
- ✅ Loading indicators
- ✅ Error handling

## Local Development

1. Install dependencies:
```bash
pip install -r requirements-vercel.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your actual API keys
```

3. Run the development server:
```bash
uvicorn api.index:app --reload
```

4. Open http://localhost:8000 in your browser

## Production Considerations

- Replace in-memory session storage with Redis or a database
- Add rate limiting
- Implement user authentication if needed
- Add logging and monitoring
- Consider using a CDN for static assets

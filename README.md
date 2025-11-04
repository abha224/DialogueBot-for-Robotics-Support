# Robotics Support Dialogue System

An AI-powered chatbot for robotics technical support that understands natural language questions and provides accurate, context-aware answers using GPT-4 and intelligent document retrieval.

## What Does It Do?

This system acts as a virtual support agent for robotics teams. Users can ask questions about robot troubleshooting, maintenance, calibration, and error codes in plain English, and receive intelligent, step-by-step guidance.

**Example Interaction:**
```
User: "My robotic arm won't respond to commands"
Bot: "Let's troubleshoot your robotic arm. First, follow these steps:
      1. Power off the robot completely
      2. Wait 30 seconds
      3. Check all cable connections
      4. Power on and wait for initialization beep
      5. Run calibration sequence
      
      If the problem persists, check error code E101 for communication issues."
```

## How It Works

```
User Question
    ↓
 Intent Classification (GPT-4)
    ↓
 Search Knowledge Base (FAISS Vector Search)
    ↓
 Retrieve Relevant Documents (RAG)
    ↓
 Generate Answer (GPT-4 + Context)
    ↓
Response to User
```

## Key Features

- **Smart Understanding**: Classifies questions into categories (troubleshooting, maintenance, etc.)
- **Context Memory**: Remembers your conversation for follow-up questions
- **Accurate Answers**: Uses RAG (Retrieval-Augmented Generation) to find relevant info before answering
- **Fast Search**: FAISS vector database for instant knowledge retrieval
- **REST API**: Easy integration with any application
- **Session Management**: Multiple users can have separate conversations

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | FastAPI | REST API server |
| **AI Model** | GPT-4 (OpenAI) | Natural language understanding & generation |
| **Orchestration** | LangChain | Chains AI components together |
| **Vector DB** | FAISS | Fast similarity search |
| **Embeddings** | OpenAI Embeddings | Convert text to vectors |
| **RAG** | ConversationalRetrievalChain | Retrieve context before answering |

## Quick Start

### Prerequisites

- Python 3.9 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# 1. Install dependencies
pip install fastapi uvicorn langchain openai faiss-cpu tiktoken pydantic

# 2. Set your OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"

# 3. Run the server
python main.py
```

The API will start at `http://localhost:8000`

### Test It Out

**Option 1: Web Browser**
- Open `http://localhost:8000/docs` for interactive API testing

**Option 2: Command Line**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset the robotic arm?", "session_id": "user123"}'
```

**Option 3: Python Script**
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "Motor is overheating, what should I check?"}
)
print(response.json()["response"])
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/query` | POST | Ask a question |
| `/reset/{session_id}` | POST | Clear conversation history |
| `/health` | GET | Check system status |

## Knowledge Base

The system comes pre-loaded with robotics support knowledge including:

- Robotic arm troubleshooting
- Motor diagnostics
- Gripper calibration
- Error code explanations
- Battery maintenance
- Sensor calibration
- Safety procedures
- Software updates

### Adding New Knowledge

Simply add entries to the `ROBOTICS_KNOWLEDGE` list:

```python
{
    "content": "Your troubleshooting steps or information here...",
    "metadata": {
        "category": "troubleshooting",
        "component": "laser_scanner"
    }
}
```

The vector store automatically rebuilds on startup.

## Use Cases

1. **24/7 Support**: Instant answers when human support isn't available
2. **Training**: Help new technicians learn procedures
3. **Quick Reference**: Fast lookup for error codes and procedures
4. **Consistency**: Same quality answers every time
5. **Scalability**: Handle unlimited simultaneous users


## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Server                       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   Dialogue Manager                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Intent     │  │   Session    │  │   Vector     │  │
│  │ Classifier   │  │   Manager    │  │   Store      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                    LangChain RAG                         │
│         (Retrieval + Memory + Generation)                │
└─────────────────────────────────────────────────────────┘
                            ↓
┌──────────────┐                           ┌──────────────┐
│  FAISS Index │ ←────────────────────────→│   GPT-4      │
│  (Vectors)   │      Retrieve Context     │   (OpenAI)   │
└──────────────┘                           └──────────────┘
```


## Important Notes

- **API Key Security**: Never commit your `.env` file or hardcode API keys
- **Costs**: GPT-4 API calls cost money. Monitor your usage at [OpenAI Platform](https://platform.openai.com/usage)
- **Rate Limits**: Free tier has request limits. Consider caching or upgrading for production


## Troubleshooting

**"OpenAI API key not found"**
→ Make sure you've set the environment variable or added it to `.env`

**"Module not found"**
→ Install all dependencies: `pip install -r requirements.txt`

**"Port already in use"**
→ Change the port: `uvicorn.run(app, port=8001)`

**Slow responses**
→ Switch to GPT-3.5-turbo or reduce the number of retrieved documents

---

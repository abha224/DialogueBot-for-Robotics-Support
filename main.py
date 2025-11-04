"""
Task-Oriented Dialogue Pipeline for Robotics Support
Uses: LangChain, GPT-4, FAISS, RAG, FastAPI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
from datetime import datetime

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Initialize FastAPI
app = FastAPI(title="Robotics Support Dialogue API", version="1.0.0")

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
VECTOR_STORE_PATH = "./faiss_index"

# Initialize components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# Sample robotics knowledge base
ROBOTICS_KNOWLEDGE = [
    {
        "content": "To reset a robotic arm, follow these steps: 1) Power off the robot completely. 2) Wait 30 seconds. 3) Check all cable connections. 4) Power on and wait for initialization beep. 5) Run calibration sequence.",
        "metadata": {"category": "troubleshooting", "component": "robotic_arm"}
    },
    {
        "content": "Motor overheating can be caused by: excessive load, insufficient cooling, blocked vents, damaged bearings, or continuous operation without breaks. Check ambient temperature and reduce duty cycle.",
        "metadata": {"category": "diagnostics", "component": "motor"}
    },
    {
        "content": "Gripper calibration procedure: 1) Open gripper fully. 2) Access calibration menu. 3) Set open position. 4) Close gripper completely. 5) Set closed position. 6) Test with object.",
        "metadata": {"category": "calibration", "component": "gripper"}
    },
    {
        "content": "Error code E101 indicates communication timeout with the main controller. Check Ethernet cables, verify IP configuration, and ensure firewall settings allow port 502.",
        "metadata": {"category": "error_codes", "component": "controller"}
    },
    {
        "content": "Battery maintenance: Charge every 3 months if not in use. Optimal operating range: 20-80% charge. Replace after 500 cycles or 3 years. Store at 50% charge in cool, dry location.",
        "metadata": {"category": "maintenance", "component": "battery"}
    },
    {
        "content": "Sensor recalibration is needed when: readings drift over time, after physical impacts, temperature changes exceed 20°C, or every 6 months as preventive maintenance.",
        "metadata": {"category": "maintenance", "component": "sensors"}
    },
    {
        "content": "Emergency stop procedure: Press red E-stop button immediately. Do not reset until hazard is cleared. Document incident. Inspect robot before resuming operation.",
        "metadata": {"category": "safety", "component": "general"}
    },
    {
        "content": "Software update process: 1) Backup current configuration. 2) Download firmware from official site. 3) Disconnect from network. 4) Install via USB. 5) Verify checksum. 6) Restart system.",
        "metadata": {"category": "software", "component": "firmware"}
    }
]

# Intent classification prompt
INTENT_TEMPLATE = """You are an AI assistant for robotics support. Analyze the user's query and classify it into one of these intents:
- troubleshooting: User has a problem with their robot
- maintenance: User needs maintenance guidance
- calibration: User needs to calibrate components
- error_code: User is asking about error codes
- general_info: User needs general information

User Query: {query}

Respond with only the intent category and a brief explanation in this format:
Intent: <category>
Explanation: <brief explanation>
"""

# Response generation prompt
RESPONSE_TEMPLATE = """You are a helpful robotics support assistant. Use the following context from the knowledge base to answer the user's question.
If the context doesn't contain relevant information, provide general guidance and suggest contacting technical support.

Context: {context}

Chat History: {chat_history}

User Question: {question}

Provide a clear, step-by-step answer. If it's a troubleshooting issue, guide the user through diagnosis. Be concise but thorough.
"""


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    

class QueryResponse(BaseModel):
    response: str
    intent: str
    sources: List[Dict]
    session_id: str
    timestamp: str


class VectorStoreManager:
    """Manages FAISS vector store for RAG"""
    
    def __init__(self):
        self.vector_store = None
        self.initialize_vector_store()
    
    def initialize_vector_store(self):
        """Create and populate FAISS vector store"""
        documents = [
            Document(page_content=item["content"], metadata=item["metadata"])
            for item in ROBOTICS_KNOWLEDGE
        ]
        
        # Split documents for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Create FAISS index
        self.vector_store = FAISS.from_documents(split_docs, embeddings)
        print(f"✓ Vector store initialized with {len(split_docs)} chunks")
    
    def similarity_search(self, query: str, k: int = 3):
        """Perform similarity search"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        return self.vector_store.similarity_search(query, k=k)
    
    def save_local(self, path: str):
        """Save FAISS index locally"""
        self.vector_store.save_local(path)
    
    def load_local(self, path: str):
        """Load FAISS index from disk"""
        self.vector_store = FAISS.load_local(path, embeddings)


class DialogueManager:
    """Manages conversation state and intent classification"""
    
    def __init__(self):
        self.sessions = {}
        self.vector_manager = VectorStoreManager()
    
    def classify_intent(self, query: str) -> tuple:
        """Classify user intent using GPT-4"""
        prompt = INTENT_TEMPLATE.format(query=query)
        response = llm.predict(prompt)
        
        # Parse intent
        lines = response.strip().split('\n')
        intent = "general_info"
        explanation = ""
        
        for line in lines:
            if line.startswith("Intent:"):
                intent = line.split(":", 1)[1].strip()
            elif line.startswith("Explanation:"):
                explanation = line.split(":", 1)[1].strip()
        
        return intent, explanation
    
    def get_or_create_session(self, session_id: str):
        """Get or create conversation memory for session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "memory": ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                ),
                "created_at": datetime.now()
            }
        return self.sessions[session_id]["memory"]
    
    def process_query(self, query: str, session_id: str) -> Dict:
        """Main query processing pipeline"""
        
        # Step 1: Intent Classification
        intent, explanation = self.classify_intent(query)
        print(f"Intent classified as: {intent}")
        
        # Step 2: Retrieve relevant documents (RAG)
        relevant_docs = self.vector_manager.similarity_search(query, k=3)
        
        # Step 3: Get conversation memory
        memory = self.get_or_create_session(session_id)
        
        # Step 4: Create retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_manager.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template=RESPONSE_TEMPLATE,
                    input_variables=["context", "chat_history", "question"]
                )
            }
        )
        
        # Step 5: Generate response
        result = qa_chain({"question": query})
        
        # Step 6: Format sources
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            for doc in result.get("source_documents", [])
        ]
        
        return {
            "response": result["answer"],
            "intent": intent,
            "sources": sources,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }


# Initialize dialogue manager
dialogue_manager = DialogueManager()


# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Robotics Support Dialogue API",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - Submit a support query",
            "/health": "GET - Health check",
            "/reset": "POST - Reset conversation session"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process user query through the dialogue pipeline"""
    try:
        result = dialogue_manager.process_query(
            query=request.query,
            session_id=request.session_id
        )
        return QueryResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/reset/{session_id}")
async def reset_session(session_id: str):
    """Reset conversation history for a session"""
    if session_id in dialogue_manager.sessions:
        del dialogue_manager.sessions[session_id]
        return {"message": f"Session {session_id} reset successfully"}
    return {"message": f"Session {session_id} not found"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_store": "initialized" if dialogue_manager.vector_manager.vector_store else "not initialized",
        "active_sessions": len(dialogue_manager.sessions)
    }


# Example usage function
def example_usage():
    """Example of how to use the system"""
    print("\n=== Robotics Support Dialogue System ===\n")
    
    # Example queries
    queries = [
        "My robotic arm won't move, what should I do?",
        "How do I calibrate the gripper?",
        "What does error code E101 mean?",
        "The motor is getting very hot during operation"
    ]
    
    for query in queries:
        print(f"Query: {query}")
        result = dialogue_manager.process_query(query, "demo_session")
        print(f"Intent: {result['intent']}")
        print(f"Response: {result['response'][:200]}...")
        print(f"Sources used: {len(result['sources'])}")
        print("-" * 80)


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Robotics Support Dialogue API...")
    print("Documentation available at: http://localhost:8000/docs")
    
    # Run example
    # example_usage()
    
    # Start API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
Enterprise API System
Complete REST and GraphQL API with authentication, rate limiting, and enterprise features
"""

import os
import jwt
import redis
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from functools import wraps
import hashlib
import secrets

# FastAPI for REST
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

# GraphQL
import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.permission import BasePermission
from strawberry.types import Info

# Database
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext

# Monitoring
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import opentelemetry
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Logging
import structlog
from loguru import logger

# Import our systems
from ..core.multi_agent_system import get_athena, AgentType
from ..core.rag_system import ENTERPRISE_RAG
from ..core.consciousness_simulation import CONSCIOUSNESS_SIMULATOR

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Database models
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    api_key = Column(String, unique=True, index=True)
    rate_limit = Column(Integer, default=1000)  # requests per hour
    permissions = Column(JSON, default=list)
    organization_id = Column(String)
    
class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    user_id = Column(Integer)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    last_used = Column(DateTime)
    permissions = Column(JSON, default=list)
    rate_limit = Column(Integer, default=1000)
    
class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    action = Column(String)
    resource = Column(String)
    resource_id = Column(String)
    ip_address = Column(String)
    user_agent = Column(String)
    request_data = Column(JSON)
    response_code = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
# Pydantic models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=8)
    organization_id: Optional[str] = None
    
class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime
    api_key: Optional[str]
    
class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    
class TaskRequest(BaseModel):
    type: str
    agent_type: Optional[str] = None
    payload: Dict[str, Any]
    priority: int = Field(default=5, ge=1, le=10)
    
class TaskResponse(BaseModel):
    task_id: str
    status: str
    submitted_at: datetime
    
class WorkflowRequest(BaseModel):
    name: str
    description: Optional[str] = None
    steps: List[Dict[str, Any]]
    
class AgentStatusResponse(BaseModel):
    agent_id: str
    agent_type: str
    status: str
    current_task: Optional[str]
    metrics: Dict[str, Any]
    
# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Metrics
api_requests_total = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
api_request_duration = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
active_users = Gauge('api_active_users', 'Currently active users')
api_errors_total = Counter('api_errors_total', 'Total API errors', ['type'])

# Create FastAPI app
app = FastAPI(
    title="AI-ARTWORKS Enterprise API",
    description="Enterprise-grade AI creative suite API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*.ai-artworks.com", "localhost"]
)

# Database setup
engine = create_engine(os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/aiartworks"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Redis setup for rate limiting and caching
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
        
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Rate limiting decorator
def rate_limit(calls: int = 10, period: int = 60):
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Get client identifier
            client_id = request.client.host
            if hasattr(request.state, "user"):
                client_id = f"user:{request.state.user.id}"
                
            # Check rate limit
            key = f"rate_limit:{client_id}:{func.__name__}"
            try:
                current = redis_client.incr(key)
                if current == 1:
                    redis_client.expire(key, period)
                if current > calls:
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limit exceeded. Maximum {calls} calls per {period} seconds."
                    )
            except redis.RedisError:
                # If Redis is down, allow the request
                logger.error("Redis connection failed for rate limiting")
                
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

# Audit logging
async def audit_log(
    db: Session,
    user: User,
    action: str,
    resource: str,
    resource_id: str,
    request: Request,
    response_code: int
):
    log_entry = AuditLog(
        user_id=user.id,
        action=action,
        resource=resource,
        resource_id=resource_id,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent"),
        request_data=await request.json() if request.method in ["POST", "PUT", "PATCH"] else None,
        response_code=response_code
    )
    db.add(log_entry)
    db.commit()

# API Routes

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Initialize rate limiter
    await FastAPILimiter.init(redis_client)
    
    # Initialize tracing
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Initialize multi-agent system
    athena = get_athena()
    
    logger.info("AI-ARTWORKS Enterprise API started")

@app.post("/api/v1/auth/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if user exists
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
        
    # Create user
    api_key = secrets.token_urlsafe(32)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=get_password_hash(user.password),
        api_key=api_key,
        organization_id=user.organization_id
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse(
        id=db_user.id,
        username=db_user.username,
        email=db_user.email,
        is_active=db_user.is_active,
        created_at=db_user.created_at,
        api_key=api_key
    )

@app.post("/api/v1/auth/token", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login and get access token"""
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create tokens
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": user.username})
    
    # Track active users
    active_users.inc()
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@app.post("/api/v1/auth/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str, db: Session = Depends(get_db)):
    """Refresh access token"""
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=400, detail="Invalid token type")
        username: str = payload.get("sub")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
        
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Create new access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,  # Return same refresh token
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@app.get("/api/v1/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user info"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        is_active=current_user.is_active,
        created_at=current_user.created_at
    )

@app.post("/api/v1/tasks", response_model=TaskResponse)
@rate_limit(calls=100, period=60)
async def create_task(
    request: Request,
    task: TaskRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Submit a new task"""
    # Validate permissions
    if task.agent_type and f"agent:{task.agent_type}" not in current_user.permissions:
        raise HTTPException(status_code=403, detail="Permission denied for this agent type")
        
    # Submit to Athena
    athena = get_athena()
    task_data = {
        "type": task.type,
        "agent_type": task.agent_type,
        "payload": task.payload,
        "priority": task.priority,
        "user_id": current_user.id
    }
    
    task_id = athena.submit_task(task_data)
    
    # Audit log
    await audit_log(db, current_user, "create_task", "task", task_id, request, 201)
    
    # Update metrics
    api_requests_total.labels(method="POST", endpoint="/tasks", status=201).inc()
    
    return TaskResponse(
        task_id=task_id,
        status="submitted",
        submitted_at=datetime.utcnow()
    )

@app.get("/api/v1/tasks/{task_id}")
async def get_task(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get task status"""
    # Check if user owns this task or is admin
    # Implementation would check task ownership
    
    # Get from Athena
    athena = get_athena()
    
    # Check active tasks
    if task_id in athena.active_tasks:
        task = athena.active_tasks[task_id]
        return {
            "task_id": task_id,
            "status": task["status"],
            "agent_id": task.get("agent_id"),
            "submitted_at": task.get("submitted_at"),
            "progress": task.get("progress", 0)
        }
        
    # Check completed tasks
    if task_id in athena.completed_tasks:
        task = athena.completed_tasks[task_id]
        return {
            "task_id": task_id,
            "status": task["status"],
            "result": task.get("result"),
            "completed_at": task.get("completed_at")
        }
        
    raise HTTPException(status_code=404, detail="Task not found")

@app.post("/api/v1/workflows", response_model=Dict[str, str])
@rate_limit(calls=10, period=60)
async def create_workflow(
    request: Request,
    workflow: WorkflowRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new workflow"""
    # Validate workflow
    for step in workflow.steps:
        if step.get("agent") and f"agent:{step['agent']}" not in current_user.permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied for agent: {step['agent']}"
            )
            
    # Create workflow
    athena = get_athena()
    workflow_data = {
        "name": workflow.name,
        "description": workflow.description,
        "steps": workflow.steps,
        "user_id": current_user.id
    }
    
    workflow_id = athena.create_workflow(workflow_data)
    
    # Audit log
    await audit_log(db, current_user, "create_workflow", "workflow", workflow_id, request, 201)
    
    return {"workflow_id": workflow_id, "status": "created"}

@app.get("/api/v1/agents", response_model=List[AgentStatusResponse])
async def list_agents(
    current_user: User = Depends(get_current_active_user)
):
    """List all agents and their status"""
    athena = get_athena()
    
    agents = []
    for agent_id, agent in athena.agents.items():
        # Check permissions
        if f"agent:{agent.agent_type.value}" in current_user.permissions or current_user.is_superuser:
            agents.append(AgentStatusResponse(
                agent_id=agent_id,
                agent_type=agent.agent_type.value,
                status=agent.status.value,
                current_task=agent.current_task["id"] if agent.current_task else None,
                metrics=agent.metrics
            ))
            
    return agents

@app.get("/api/v1/agents/{agent_id}")
async def get_agent(
    agent_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get specific agent details"""
    athena = get_athena()
    
    if agent_id not in athena.agents:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    agent = athena.agents[agent_id]
    
    # Check permissions
    if f"agent:{agent.agent_type.value}" not in current_user.permissions and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Permission denied")
        
    return {
        "agent_id": agent_id,
        "agent_type": agent.agent_type.value,
        "status": agent.status.value,
        "current_task": agent.current_task,
        "metrics": agent.metrics,
        "capabilities": athena.agent_capabilities.get(agent_id, {})
    }

@app.post("/api/v1/rag/documents")
@rate_limit(calls=50, period=60)
async def upload_document(
    request: Request,
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload document to RAG system"""
    if "rag:write" not in current_user.permissions:
        raise HTTPException(status_code=403, detail="Permission denied")
        
    # Add to RAG system
    rag = ENTERPRISE_RAG
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not available")
        
    try:
        documents = rag.rag_pipeline.add_documents([file_path], [metadata] if metadata else None)
        
        # Audit log
        await audit_log(db, current_user, "upload_document", "document", documents[0].id, request, 201)
        
        return {
            "document_id": documents[0].id,
            "chunks": len(documents[0].chunks),
            "status": "indexed"
        }
    except Exception as e:
        api_errors_total.labels(type="rag_upload").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/rag/query")
@rate_limit(calls=100, period=60)
async def query_rag(
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Query the RAG system"""
    if "rag:read" not in current_user.permissions:
        raise HTTPException(status_code=403, detail="Permission denied")
        
    rag = ENTERPRISE_RAG
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not available")
        
    try:
        result = rag.query(query, k=k, filters=filters)
        return result
    except Exception as e:
        api_errors_total.labels(type="rag_query").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/consciousness/state")
async def get_consciousness_state(
    current_user: User = Depends(get_current_active_user)
):
    """Get current consciousness simulation state"""
    if "consciousness:read" not in current_user.permissions:
        raise HTTPException(status_code=403, detail="Permission denied")
        
    sim = CONSCIOUSNESS_SIMULATOR
    if not sim:
        raise HTTPException(status_code=503, detail="Consciousness simulator not available")
        
    state = sim.current_consciousness
    return {
        "awareness_level": state.awareness_level,
        "coherence": state.coherence,
        "entropy": state.entropy,
        "dimensions": state.dimensions,
        "timestamp": state.timestamp.isoformat()
    }

@app.post("/api/v1/consciousness/manipulate")
@rate_limit(calls=10, period=60)
async def manipulate_consciousness(
    operation: str,
    parameters: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Manipulate consciousness state"""
    if "consciousness:write" not in current_user.permissions:
        raise HTTPException(status_code=403, detail="Permission denied")
        
    sim = CONSCIOUSNESS_SIMULATOR
    if not sim:
        raise HTTPException(status_code=503, detail="Consciousness simulator not available")
        
    valid_operations = ["set_awareness", "set_dimension", "generate_thought", "shift_reality"]
    if operation not in valid_operations:
        raise HTTPException(status_code=400, detail=f"Invalid operation. Must be one of: {valid_operations}")
        
    try:
        if operation == "set_awareness":
            sim.set_consciousness_level(parameters.get("level", 0.5))
        elif operation == "set_dimension":
            sim.set_dimension(parameters.get("dimension"), parameters.get("value"))
        elif operation == "generate_thought":
            thought = sim.generate_thought(parameters.get("prompt"))
            return {"thought": thought}
        elif operation == "shift_reality":
            sim.reality_manipulation_agent.shift_reality(parameters.get("target"))
            
        return {"status": "success", "operation": operation}
    except Exception as e:
        api_errors_total.labels(type="consciousness_manipulation").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/metrics")
async def get_metrics(current_user: User = Depends(get_current_active_user)):
    """Get Prometheus metrics"""
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Admin access required")
        
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    athena = get_athena()
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "up",
            "database": "up" if engine else "down",
            "redis": "up" if redis_client.ping() else "down",
            "athena": "up" if athena else "down",
            "agents": len(athena.agents) if athena else 0
        }
    }
    
    # Check if any service is down
    if any(v == "down" for v in health_status["services"].values() if isinstance(v, str)):
        health_status["status"] = "degraded"
        
    return health_status

# GraphQL Schema
@strawberry.type
class Task:
    id: str
    type: str
    status: str
    agent_id: Optional[str]
    result: Optional[str]
    
@strawberry.type
class Agent:
    id: str
    type: str
    status: str
    current_task: Optional[str]
    
@strawberry.type
class Query:
    @strawberry.field
    async def task(self, task_id: str) -> Optional[Task]:
        athena = get_athena()
        if task_id in athena.active_tasks:
            t = athena.active_tasks[task_id]
            return Task(
                id=task_id,
                type=t.get("type"),
                status=t.get("status"),
                agent_id=t.get("agent_id"),
                result=None
            )
        return None
        
    @strawberry.field
    async def agents(self) -> List[Agent]:
        athena = get_athena()
        return [
            Agent(
                id=agent_id,
                type=agent.agent_type.value,
                status=agent.status.value,
                current_task=agent.current_task["id"] if agent.current_task else None
            )
            for agent_id, agent in athena.agents.items()
        ]

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def submit_task(self, type: str, payload: str) -> Task:
        athena = get_athena()
        task_data = {
            "type": type,
            "payload": json.loads(payload)
        }
        task_id = athena.submit_task(task_data)
        
        return Task(
            id=task_id,
            type=type,
            status="submitted",
            agent_id=None,
            result=None
        )

# Add GraphQL endpoint
schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/api/graphql")

# WebSocket support for real-time updates
from fastapi import WebSocket, WebSocketDisconnect
from typing import Set

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages
            # For now, just echo back
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Admin endpoints
@app.post("/api/v1/admin/users/{user_id}/permissions")
async def update_user_permissions(
    user_id: int,
    permissions: List[str],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update user permissions (admin only)"""
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Admin access required")
        
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    user.permissions = permissions
    db.commit()
    
    return {"status": "success", "permissions": permissions}

@app.get("/api/v1/admin/audit-logs")
async def get_audit_logs(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get audit logs (admin only)"""
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Admin access required")
        
    logs = db.query(AuditLog).offset(skip).limit(limit).all()
    return logs

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    api_errors_total.labels(type="http_error").inc()
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    api_errors_total.labels(type="internal_error").inc()
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
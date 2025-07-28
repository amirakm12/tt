"""
Triage Agent
Handles initial request processing, classification, and routing
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json
from collections import defaultdict
import hashlib

from core.config import SystemConfig
from ai.rag_engine import RAGEngine
from ai.speculative_decoder import SpeculativeDecoder

logger = logging.getLogger(__name__)

class RequestType(Enum):
    """Types of requests the system can handle."""
    QUESTION = "question"
    COMMAND = "command"
    ANALYSIS = "analysis"
    RESEARCH = "research"
    MONITORING = "monitoring"
    SYSTEM_CONTROL = "system_control"
    UNKNOWN = "unknown"

class Priority(Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class RequestStatus(Enum):
    """Request processing status."""
    RECEIVED = "received"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"

@dataclass
class TriageResult:
    """Result of request triage."""
    request_id: str
    request_type: RequestType
    priority: Priority
    complexity_score: float
    confidence: float
    routing_destination: str
    estimated_processing_time: float
    required_capabilities: List[str]
    metadata: Dict[str, Any]

@dataclass
class ProcessedRequest:
    """Processed request with response."""
    request_id: str
    original_request: str
    triage_result: TriageResult
    response: str
    processing_time: float
    status: RequestStatus
    timestamp: float
    metadata: Dict[str, Any]

class TriageAgent:
    """Agent responsible for request triage and initial processing."""
    
    def __init__(self, config: SystemConfig, rag_engine: RAGEngine, speculative_decoder: SpeculativeDecoder):
        self.config = config
        self.rag_engine = rag_engine
        self.speculative_decoder = speculative_decoder
        self.is_running = False
        
        # Request processing
        self.request_queue = asyncio.Queue()
        self.processed_requests = {}
        self.active_requests = {}
        
        # Classification patterns
        self.classification_patterns = self._initialize_classification_patterns()
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'requests_by_type': defaultdict(int),
            'requests_by_priority': defaultdict(int),
            'average_triage_time': 0.0,
            'average_processing_time': 0.0,
            'success_rate': 0.0,
            'escalation_rate': 0.0
        }
        
        # Processing history for learning
        self.processing_history = []
        self.max_history_size = 10000
        
        logger.info("Triage Agent initialized")
    
    def _initialize_classification_patterns(self) -> Dict[RequestType, List[str]]:
        """Initialize patterns for request classification."""
        return {
            RequestType.QUESTION: [
                r'\bwhat\b', r'\bhow\b', r'\bwhy\b', r'\bwhen\b', r'\bwhere\b', r'\bwho\b',
                r'\?', r'\bexplain\b', r'\btell me\b', r'\bdefine\b'
            ],
            RequestType.COMMAND: [
                r'\bstart\b', r'\bstop\b', r'\brestart\b', r'\brun\b', r'\bexecute\b',
                r'\bshutdown\b', r'\bpause\b', r'\bresume\b', r'\bconfigure\b'
            ],
            RequestType.ANALYSIS: [
                r'\banalyze\b', r'\bcompare\b', r'\bevaluate\b', r'\bassess\b',
                r'\breport\b', r'\bsummarize\b', r'\bstatistics\b', r'\bmetrics\b'
            ],
            RequestType.RESEARCH: [
                r'\bresearch\b', r'\binvestigate\b', r'\bfind\b', r'\bsearch\b',
                r'\blook up\b', r'\bgather\b', r'\bcollect\b', r'\bexplore\b'
            ],
            RequestType.MONITORING: [
                r'\bmonitor\b', r'\bwatch\b', r'\btrack\b', r'\bobserve\b',
                r'\bstatus\b', r'\bhealth\b', r'\bperformance\b', r'\balert\b'
            ],
            RequestType.SYSTEM_CONTROL: [
                r'\bsystem\b', r'\bkernel\b', r'\bdriver\b', r'\bservice\b',
                r'\bprocess\b', r'\bmemory\b', r'\bcpu\b', r'\bdisk\b'
            ]
        }
    
    async def initialize(self):
        """Initialize the triage agent."""
        logger.info("Initializing Triage Agent...")
        
        try:
            # Load historical data if available
            await self._load_processing_history()
            
            # Initialize classification models if needed
            await self._initialize_classification_models()
            
            logger.info("Triage Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Triage Agent: {e}")
            raise
    
    async def start(self):
        """Start the triage agent."""
        logger.info("Starting Triage Agent...")
        
        try:
            # Start background tasks
            self.background_tasks = {
                'request_processor': asyncio.create_task(self._request_processing_loop()),
                'metrics_collector': asyncio.create_task(self._metrics_collection_loop()),
                'history_manager': asyncio.create_task(self._history_management_loop()),
                'learning_optimizer': asyncio.create_task(self._learning_optimization_loop())
            }
            
            self.is_running = True
            logger.info("Triage Agent started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Triage Agent: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the triage agent."""
        logger.info("Shutting down Triage Agent...")
        
        self.is_running = False
        
        # Cancel background tasks
        for task_name, task in self.background_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Cancelled {task_name}")
        
        # Save processing history
        await self._save_processing_history()
        
        logger.info("Triage Agent shutdown complete")
    
    async def process_request(self, request: str, metadata: Dict[str, Any] = None) -> ProcessedRequest:
        """Process a request through triage and routing."""
        start_time = time.time()
        request_id = self._generate_request_id(request)
        
        logger.info(f"Processing request {request_id}: {request[:100]}...")
        
        try:
            # Perform triage
            triage_start = time.time()
            triage_result = await self._perform_triage(request, metadata or {})
            triage_time = time.time() - triage_start
            
            # Add to active requests
            self.active_requests[request_id] = {
                'request': request,
                'triage_result': triage_result,
                'start_time': start_time,
                'metadata': metadata or {}
            }
            
            # Route and process request
            processing_start = time.time()
            response = await self._route_and_process(request, triage_result)
            processing_time = time.time() - processing_start
            
            # Create processed request
            processed_request = ProcessedRequest(
                request_id=request_id,
                original_request=request,
                triage_result=triage_result,
                response=response,
                processing_time=time.time() - start_time,
                status=RequestStatus.COMPLETED,
                timestamp=time.time(),
                metadata={
                    'triage_time': triage_time,
                    'processing_time': processing_time,
                    **(metadata or {})
                }
            )
            
            # Store processed request
            self.processed_requests[request_id] = processed_request
            
            # Remove from active requests
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            
            # Update metrics
            self._update_metrics(processed_request)
            
            # Add to processing history
            self.processing_history.append(processed_request)
            if len(self.processing_history) > self.max_history_size:
                self.processing_history = self.processing_history[-self.max_history_size:]
            
            logger.info(f"Completed request {request_id} in {processed_request.processing_time:.3f}s")
            return processed_request
            
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            
            # Create failed request
            failed_request = ProcessedRequest(
                request_id=request_id,
                original_request=request,
                triage_result=TriageResult(
                    request_id=request_id,
                    request_type=RequestType.UNKNOWN,
                    priority=Priority.NORMAL,
                    complexity_score=0.0,
                    confidence=0.0,
                    routing_destination="error",
                    estimated_processing_time=0.0,
                    required_capabilities=[],
                    metadata={}
                ),
                response=f"Error processing request: {str(e)}",
                processing_time=time.time() - start_time,
                status=RequestStatus.FAILED,
                timestamp=time.time(),
                metadata={'error': str(e), **(metadata or {})}
            )
            
            self.processed_requests[request_id] = failed_request
            
            # Remove from active requests
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            
            return failed_request
    
    async def _perform_triage(self, request: str, metadata: Dict[str, Any]) -> TriageResult:
        """Perform triage analysis on a request."""
        request_id = self._generate_request_id(request)
        
        # Classify request type
        request_type = self._classify_request_type(request)
        
        # Determine priority
        priority = self._determine_priority(request, request_type, metadata)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(request, request_type)
        
        # Determine routing destination
        routing_destination = self._determine_routing_destination(request_type, complexity_score)
        
        # Estimate processing time
        estimated_time = self._estimate_processing_time(request_type, complexity_score)
        
        # Determine required capabilities
        required_capabilities = self._determine_required_capabilities(request, request_type)
        
        # Calculate confidence in triage
        confidence = self._calculate_triage_confidence(request, request_type, complexity_score)
        
        return TriageResult(
            request_id=request_id,
            request_type=request_type,
            priority=priority,
            complexity_score=complexity_score,
            confidence=confidence,
            routing_destination=routing_destination,
            estimated_processing_time=estimated_time,
            required_capabilities=required_capabilities,
            metadata={
                'request_length': len(request),
                'word_count': len(request.split()),
                'has_code': bool(re.search(r'```|`[^`]+`', request)),
                'has_urls': bool(re.search(r'https?://', request)),
                'urgency_keywords': self._count_urgency_keywords(request)
            }
        )
    
    def _classify_request_type(self, request: str) -> RequestType:
        """Classify the type of request."""
        request_lower = request.lower()
        type_scores = {}
        
        # Score each request type based on pattern matches
        for req_type, patterns in self.classification_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, request_lower))
                score += matches
            type_scores[req_type] = score
        
        # Return type with highest score, or UNKNOWN if no matches
        if max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        else:
            return RequestType.UNKNOWN
    
    def _determine_priority(self, request: str, request_type: RequestType, metadata: Dict[str, Any]) -> Priority:
        """Determine request priority."""
        priority_score = 0
        
        # Base priority by request type
        type_priorities = {
            RequestType.SYSTEM_CONTROL: 3,
            RequestType.MONITORING: 2,
            RequestType.COMMAND: 2,
            RequestType.ANALYSIS: 1,
            RequestType.RESEARCH: 1,
            RequestType.QUESTION: 0,
            RequestType.UNKNOWN: 0
        }
        priority_score += type_priorities.get(request_type, 0)
        
        # Check for urgency keywords
        urgency_keywords = [
            r'\burgent\b', r'\bcritical\b', r'\bemergency\b', r'\basap\b',
            r'\bimmediately\b', r'\bhigh priority\b', r'\bcrash\b', r'\bfail\b'
        ]
        
        for keyword in urgency_keywords:
            if re.search(keyword, request.lower()):
                priority_score += 2
                break
        
        # Check metadata for priority hints
        if metadata.get('priority'):
            try:
                metadata_priority = Priority[metadata['priority'].upper()]
                priority_score = max(priority_score, metadata_priority.value - 1)
            except (KeyError, ValueError):
                pass
        
        # Convert score to priority
        if priority_score >= 4:
            return Priority.CRITICAL
        elif priority_score >= 3:
            return Priority.URGENT
        elif priority_score >= 2:
            return Priority.HIGH
        elif priority_score >= 1:
            return Priority.NORMAL
        else:
            return Priority.LOW
    
    def _calculate_complexity_score(self, request: str, request_type: RequestType) -> float:
        """Calculate complexity score for the request."""
        complexity = 0.0
        
        # Base complexity by type
        type_complexity = {
            RequestType.SYSTEM_CONTROL: 0.8,
            RequestType.ANALYSIS: 0.7,
            RequestType.RESEARCH: 0.6,
            RequestType.MONITORING: 0.4,
            RequestType.COMMAND: 0.3,
            RequestType.QUESTION: 0.2,
            RequestType.UNKNOWN: 0.5
        }
        complexity += type_complexity.get(request_type, 0.5)
        
        # Adjust based on request characteristics
        word_count = len(request.split())
        if word_count > 100:
            complexity += 0.2
        elif word_count > 50:
            complexity += 0.1
        
        # Check for complex patterns
        if re.search(r'```|`[^`]+`', request):  # Code blocks
            complexity += 0.2
        
        if re.search(r'\bmultiple\b|\bseveral\b|\bmany\b', request.lower()):
            complexity += 0.1
        
        if re.search(r'\bcomplex\b|\badvanced\b|\bdetailed\b', request.lower()):
            complexity += 0.2
        
        return min(1.0, complexity)
    
    def _determine_routing_destination(self, request_type: RequestType, complexity_score: float) -> str:
        """Determine where to route the request."""
        # High complexity requests go to research agent
        if complexity_score > 0.7:
            return "research_agent"
        
        # Route based on request type
        routing_map = {
            RequestType.QUESTION: "rag_engine",
            RequestType.COMMAND: "orchestration_agent",
            RequestType.ANALYSIS: "research_agent",
            RequestType.RESEARCH: "research_agent",
            RequestType.MONITORING: "system_monitor",
            RequestType.SYSTEM_CONTROL: "kernel_manager",
            RequestType.UNKNOWN: "rag_engine"  # Default fallback
        }
        
        return routing_map.get(request_type, "rag_engine")
    
    def _estimate_processing_time(self, request_type: RequestType, complexity_score: float) -> float:
        """Estimate processing time for the request."""
        # Base time estimates (in seconds)
        base_times = {
            RequestType.QUESTION: 2.0,
            RequestType.COMMAND: 5.0,
            RequestType.ANALYSIS: 15.0,
            RequestType.RESEARCH: 30.0,
            RequestType.MONITORING: 3.0,
            RequestType.SYSTEM_CONTROL: 10.0,
            RequestType.UNKNOWN: 5.0
        }
        
        base_time = base_times.get(request_type, 5.0)
        
        # Adjust based on complexity
        complexity_multiplier = 1.0 + (complexity_score * 2.0)
        
        return base_time * complexity_multiplier
    
    def _determine_required_capabilities(self, request: str, request_type: RequestType) -> List[str]:
        """Determine what capabilities are required to handle the request."""
        capabilities = []
        
        # Base capabilities by type
        type_capabilities = {
            RequestType.QUESTION: ["rag", "language_model"],
            RequestType.COMMAND: ["system_control", "orchestration"],
            RequestType.ANALYSIS: ["data_analysis", "rag", "language_model"],
            RequestType.RESEARCH: ["rag", "web_search", "language_model"],
            RequestType.MONITORING: ["system_monitoring", "sensors"],
            RequestType.SYSTEM_CONTROL: ["kernel_access", "system_control"],
            RequestType.UNKNOWN: ["language_model"]
        }
        
        capabilities.extend(type_capabilities.get(request_type, []))
        
        # Check for specific capability requirements
        if re.search(r'\bfile\b|\bdocument\b', request.lower()):
            capabilities.append("file_access")
        
        if re.search(r'\bnetwork\b|\binternet\b|\bweb\b', request.lower()):
            capabilities.append("network_access")
        
        if re.search(r'\bdatabase\b|\bsql\b', request.lower()):
            capabilities.append("database_access")
        
        if re.search(r'\bimage\b|\bphoto\b|\bvideo\b', request.lower()):
            capabilities.append("media_processing")
        
        return list(set(capabilities))  # Remove duplicates
    
    def _calculate_triage_confidence(self, request: str, request_type: RequestType, complexity_score: float) -> float:
        """Calculate confidence in the triage result."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for clear request types
        if request_type != RequestType.UNKNOWN:
            confidence += 0.3
        
        # Adjust based on request clarity
        word_count = len(request.split())
        if 10 <= word_count <= 100:  # Optimal length
            confidence += 0.1
        elif word_count < 5 or word_count > 200:  # Too short or too long
            confidence -= 0.2
        
        # Check for clear intent indicators
        clear_indicators = [
            r'\bplease\b', r'\bcan you\b', r'\bi need\b', r'\bhelp me\b',
            r'\bshow me\b', r'\btell me\b', r'\bexplain\b'
        ]
        
        for indicator in clear_indicators:
            if re.search(indicator, request.lower()):
                confidence += 0.1
                break
        
        return min(1.0, max(0.0, confidence))
    
    def _count_urgency_keywords(self, request: str) -> int:
        """Count urgency-related keywords in the request."""
        urgency_keywords = [
            'urgent', 'critical', 'emergency', 'asap', 'immediately',
            'quickly', 'fast', 'now', 'help', 'problem', 'issue',
            'error', 'fail', 'crash', 'down', 'broken'
        ]
        
        count = 0
        request_lower = request.lower()
        for keyword in urgency_keywords:
            count += len(re.findall(rf'\b{keyword}\b', request_lower))
        
        return count
    
    async def _route_and_process(self, request: str, triage_result: TriageResult) -> str:
        """Route request to appropriate component and get response."""
        destination = triage_result.routing_destination
        
        try:
            if destination == "rag_engine":
                # Use RAG engine for knowledge-based queries
                rag_result = await self.rag_engine.query(request)
                return rag_result.generated_response
                
            elif destination == "research_agent":
                # For complex research tasks, use a more sophisticated approach
                # This would typically involve the research agent
                return await self._handle_research_request(request, triage_result)
                
            elif destination == "orchestration_agent":
                # For command and orchestration requests
                return await self._handle_orchestration_request(request, triage_result)
                
            elif destination == "system_monitor":
                # For monitoring requests
                return await self._handle_monitoring_request(request, triage_result)
                
            elif destination == "kernel_manager":
                # For system control requests
                return await self._handle_system_control_request(request, triage_result)
                
            else:
                # Fallback to RAG engine
                rag_result = await self.rag_engine.query(request)
                return rag_result.generated_response
                
        except Exception as e:
            logger.error(f"Error routing request to {destination}: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    async def _handle_research_request(self, request: str, triage_result: TriageResult) -> str:
        """Handle research-type requests."""
        # Use speculative decoder for complex reasoning
        from ai.speculative_decoder import DecodingRequest, SpeculationStrategy
        
        decoding_request = DecodingRequest(
            prompt=f"Research and provide a comprehensive answer to: {request}",
            max_tokens=500,
            temperature=0.3,
            strategy=SpeculationStrategy.TREE_ATTENTION
        )
        
        result = await self.speculative_decoder.decode(decoding_request)
        return result.text
    
    async def _handle_orchestration_request(self, request: str, triage_result: TriageResult) -> str:
        """Handle orchestration and command requests."""
        # This would typically interface with the orchestration agent
        return f"Command request received: {request}. This would be processed by the orchestration system."
    
    async def _handle_monitoring_request(self, request: str, triage_result: TriageResult) -> str:
        """Handle monitoring requests."""
        # This would interface with system monitoring components
        return f"Monitoring request: {request}. Current system status would be provided here."
    
    async def _handle_system_control_request(self, request: str, triage_result: TriageResult) -> str:
        """Handle system control requests."""
        # This would interface with kernel manager
        return f"System control request: {request}. This would require appropriate permissions and would be handled by the kernel manager."
    
    def _generate_request_id(self, request: str) -> str:
        """Generate unique request ID."""
        content = f"{request}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _update_metrics(self, processed_request: ProcessedRequest):
        """Update performance metrics."""
        self.metrics['total_requests'] += 1
        self.metrics['requests_by_type'][processed_request.triage_result.request_type.value] += 1
        self.metrics['requests_by_priority'][processed_request.triage_result.priority.value] += 1
        
        # Update average processing time
        total_time = (self.metrics['average_processing_time'] * (self.metrics['total_requests'] - 1) + 
                     processed_request.processing_time)
        self.metrics['average_processing_time'] = total_time / self.metrics['total_requests']
        
        # Update success rate
        successful_requests = sum(1 for req in self.processed_requests.values() 
                                if req.status == RequestStatus.COMPLETED)
        self.metrics['success_rate'] = successful_requests / self.metrics['total_requests']
    
    async def _request_processing_loop(self):
        """Background loop for processing queued requests."""
        while self.is_running:
            try:
                # This could handle queued requests if needed
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in request processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _metrics_collection_loop(self):
        """Background loop for metrics collection."""
        while self.is_running:
            try:
                if self.metrics['total_requests'] > 0:
                    logger.info(f"Triage Agent Metrics - "
                              f"Total Requests: {self.metrics['total_requests']}, "
                              f"Success Rate: {self.metrics['success_rate']:.2%}, "
                              f"Avg Processing Time: {self.metrics['average_processing_time']:.3f}s")
                
                await asyncio.sleep(300)  # Log every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(300)
    
    async def _history_management_loop(self):
        """Background loop for managing processing history."""
        while self.is_running:
            try:
                # Clean up old processed requests
                current_time = time.time()
                old_requests = [
                    req_id for req_id, req in self.processed_requests.items()
                    if current_time - req.timestamp > 3600  # 1 hour
                ]
                
                for req_id in old_requests:
                    del self.processed_requests[req_id]
                
                if old_requests:
                    logger.info(f"Cleaned up {len(old_requests)} old processed requests")
                
                await asyncio.sleep(1800)  # Clean every 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in history management: {e}")
                await asyncio.sleep(1800)
    
    async def _learning_optimization_loop(self):
        """Background loop for learning and optimization."""
        while self.is_running:
            try:
                # Analyze processing history to improve triage
                if len(self.processing_history) > 100:
                    await self._analyze_and_optimize()
                
                await asyncio.sleep(3600)  # Optimize every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in learning optimization: {e}")
                await asyncio.sleep(3600)
    
    async def _analyze_and_optimize(self):
        """Analyze processing history and optimize triage."""
        # This could implement machine learning to improve classification
        # For now, just log some basic analysis
        
        recent_requests = self.processing_history[-100:]
        
        # Analyze accuracy of complexity scoring
        complexity_errors = []
        for req in recent_requests:
            estimated_time = req.triage_result.estimated_processing_time
            actual_time = req.processing_time
            error = abs(estimated_time - actual_time) / max(actual_time, 0.1)
            complexity_errors.append(error)
        
        avg_complexity_error = sum(complexity_errors) / len(complexity_errors)
        logger.info(f"Average complexity estimation error: {avg_complexity_error:.2%}")
    
    async def _initialize_classification_models(self):
        """Initialize any ML models for classification."""
        # Placeholder for ML model initialization
        pass
    
    async def _load_processing_history(self):
        """Load historical processing data."""
        # Placeholder for loading historical data
        pass
    
    async def _save_processing_history(self):
        """Save processing history to persistent storage."""
        # Placeholder for saving historical data
        pass
    
    # Public API methods
    
    async def health_check(self) -> str:
        """Perform health check."""
        try:
            # Test basic functionality
            test_request = "What is the system status?"
            test_result = await self._perform_triage(test_request, {})
            
            if test_result.confidence > 0.0:
                return "healthy"
            else:
                return "unhealthy"
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return "unhealthy"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get triage agent statistics."""
        return {
            'metrics': self.metrics.copy(),
            'active_requests': len(self.active_requests),
            'processed_requests': len(self.processed_requests),
            'history_size': len(self.processing_history)
        }
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific request."""
        if request_id in self.processed_requests:
            req = self.processed_requests[request_id]
            return {
                'request_id': req.request_id,
                'status': req.status.value,
                'processing_time': req.processing_time,
                'timestamp': req.timestamp,
                'request_type': req.triage_result.request_type.value,
                'priority': req.triage_result.priority.value
            }
        elif request_id in self.active_requests:
            req = self.active_requests[request_id]
            return {
                'request_id': request_id,
                'status': 'processing',
                'processing_time': time.time() - req['start_time'],
                'request_type': req['triage_result'].request_type.value,
                'priority': req['triage_result'].priority.value
            }
        else:
            return None
    
    async def restart(self):
        """Restart the triage agent."""
        logger.info("Restarting Triage Agent...")
        await self.shutdown()
        await asyncio.sleep(1)
        await self.initialize()
        await self.start()
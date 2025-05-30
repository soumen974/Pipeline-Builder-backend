from fastapi import FastAPI, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class Node(BaseModel):
    id: str
    type: str
    position: Dict[str, float]
    data: Dict[str, Any]

class Edge(BaseModel):
    id: str
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None

class PipelineData(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class PipelineResponse(BaseModel):
    num_nodes: int
    num_edges: int
    is_dag: bool
    message: str
    execution_order: Optional[List[str]] = None
    node_types: Optional[Dict[str, int]] = None

@app.get('/')
def read_root():
    return {'Ping': 'Pong', 'By': 'Soumen Bhunia'}

# GET endpoint with Query parameter
@app.get('/pipelines/parse', response_model=PipelineResponse)
def parse_pipeline_get(pipeline: str = Query(..., description="JSON string of pipeline data")):
    try:
        # Parse the pipeline data from JSON string
        pipeline_data = json.loads(pipeline)
        return process_pipeline(pipeline_data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing pipeline: {str(e)}")

# POST endpoint with JSON body (preferred method)
@app.post('/pipelines/parse', response_model=PipelineResponse)
def parse_pipeline_post(pipeline_data: PipelineData):
    try:
        return process_pipeline(pipeline_data.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing pipeline: {str(e)}")

def process_pipeline(pipeline_data: dict) -> dict:
    """Process pipeline data and return analysis results"""
    nodes = pipeline_data.get("nodes", [])
    edges = pipeline_data.get("edges", [])
    
    # Validate pipeline structure
    if not nodes:
        raise HTTPException(status_code=400, detail="Pipeline must contain at least one node")
    
    # Count nodes and edges
    num_nodes = len(nodes)
    num_edges = len(edges)
    
    # Check if the graph is a DAG
    is_dag = check_if_dag(nodes, edges)
    
    # Get execution order if it's a DAG
    execution_order = get_execution_order(nodes, edges) if is_dag else None
    
    # Count node types
    node_types = {}
    for node in nodes:
        node_type = node.get("type", "unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    # Create success message
    message = f"Pipeline analyzed successfully. {'Valid DAG structure.' if is_dag else 'Contains cycles - not a valid DAG.'}"
    
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "is_dag": is_dag,
        "message": message,
        "execution_order": execution_order,
        "node_types": node_types
    }

def check_if_dag(nodes, edges):
    """
    Check if a graph is a Directed Acyclic Graph (DAG)
    """
    # Create a dictionary to store the graph
    graph = {node["id"]: [] for node in nodes}
    
    # Add edges to the graph
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in graph and target in graph:
            graph[source].append(target)
    
    # Set to keep track of visited nodes
    visited = set()
    # Set to keep track of nodes in the current recursion stack
    rec_stack = set()
    
    # DFS function to detect cycles
    def is_cyclic(node):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if is_cyclic(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    # Check for cycles starting from each unvisited node
    for node_id in graph:
        if node_id not in visited:
            if is_cyclic(node_id):
                return False  # Contains a cycle, not a DAG
    
    return True  # No cycles, is a DAG

def get_execution_order(nodes, edges):
    """
    Get topological sort order for DAG execution
    """
    if not check_if_dag(nodes, edges):
        return None
    
    # Create adjacency list and in-degree count
    graph = {node["id"]: [] for node in nodes}
    in_degree = {node["id"]: 0 for node in nodes}
    
    # Build graph and calculate in-degrees
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in graph and target in graph:
            graph[source].append(target)
            in_degree[target] += 1
    
    # Kahn's algorithm for topological sorting
    queue = [node_id for node_id in in_degree if in_degree[node_id] == 0]
    execution_order = []
    
    while queue:
        current = queue.pop(0)
        execution_order.append(current)
        
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return execution_order if len(execution_order) == len(nodes) else None

# Health check endpoint
@app.get('/health')
def health_check():
    return {"status": "healthy", "service": "Pipeline Parser API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from langgraph.graph import StateGraph, END
from swastikA.app.schemas import WorkflowState
from swastikA.app.variation_node import VariationRecreateNode
from swastikA.app.manim_node import ManimNode
from swastikA.app.analysis_node import AnalysisNode
from swastikA.app.preprocessing_node import PreprocessingNode

async def workflow_agent() -> StateGraph:
    """Create and configure the Kolam processing workflow.
    
    The workflow follows these steps:
    1. preprocessing (required)
    2. manim and analysis (run in parallel after preprocessing)
    3. variation (runs after analysis completes)
    
    Returns:
        Configured StateGraph ready for execution
    """
    # Initialize the workflow with our state model
    workflow = StateGraph(WorkflowState)
    
    # Add all nodes
    workflow.add_node("preprocessing", PreprocessingNode())
    workflow.add_node("manim", ManimNode())
    workflow.add_node("analysis", AnalysisNode())
    workflow.add_node("variation", VariationRecreateNode())
    
    # Define the workflow edges
    workflow.set_entry_point("preprocessing")
    
    # After preprocessing, run manim and analysis in parallel
    workflow.add_edge("preprocessing", "manim")
    workflow.add_edge("manim", "analysis")
    
    # After analysis completes, run variation
    workflow.add_edge("analysis", "variation")
    workflow.add_edge("variation", END)
    
    # Compile the workflow
    return workflow.compile()

    


# test_workflow.py
import asyncio
from pathlib import Path
from swastikA.app.schemas import WorkflowState
from swastikA.app.workflow_agent import workflow_agent

async def test_workflow_with_image(image_path: str):
    """Test the workflow with a specific image."""
    # Initialize the workflow
    workflow = await workflow_agent()
    
    # Create initial state
    initial_state = WorkflowState(
        original_image_path=image_path,
        is_completed=True,
        is_in_frame=True,
        file_prefix="test_kolam",
        output_folder_path="swastikA/media",
    )
    
    try:
        # Run the workflow
        print(f"Starting workflow with image: {image_path}")
        result = await workflow.ainvoke(initial_state)
        print("Workflow completed successfully!")
       
        return result
        
    except Exception as e:
        print(f"Error in workflow: {str(e)}")
        raise


if __name__ == "__main__":
    # Update this path to your image
    image_path = "image.jpg"  # Replace with your actual image path
    
    # Run the test
    result = asyncio.run(test_workflow_with_image(image_path))
    breakpoint()
    # Save the result to a text file
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"workflow_result_{timestamp}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(str(result))
    breakpoint()
    # Load the result
    # loaded_result = load_workflow_result("path_to_saved_file.json.gz")

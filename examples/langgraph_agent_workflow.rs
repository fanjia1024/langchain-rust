use langchain_rust::langgraph::{function_node, StateGraph, MessagesState, END, START};
use langchain_rust::schemas::messages::Message;
use std::collections::HashMap;

/// Complex agent workflow example for LangGraph
///
/// This example demonstrates:
/// 1. Multiple nodes in a workflow
/// 2. Conditional edges for routing
/// 3. State updates and merging
/// 4. Streaming execution
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Node 1: Process input
    let process_input = function_node("process_input", |state: &MessagesState| async move {
        let mut update = HashMap::new();
        
        // Get the last message
        if let Some(last_msg) = state.messages.last() {
            let processed = format!("Processed: {}", last_msg.content);
            update.insert(
                "messages".to_string(),
                serde_json::to_value(vec![Message::new_ai_message(&processed)])?,
            );
        }
        
        Ok(update)
    });

    // Node 2: Decide next action (condition node)
    let decide_action = function_node("decide_action", |state: &MessagesState| async move {
        let mut update = HashMap::new();
        
        // Simple decision logic: if message contains "help", route to help node
        // Otherwise, route to response node
        let needs_help = state.messages.iter().any(|msg| {
            msg.content.to_lowercase().contains("help")
        });
        
        let action = if needs_help {
            "help"
        } else {
            "respond"
        };
        
        update.insert("action".to_string(), serde_json::to_value(action)?);
        Ok(update)
    });

    // Node 3: Provide help
    let provide_help = function_node("provide_help", |_state: &MessagesState| async move {
        let mut update = HashMap::new();
        update.insert(
            "messages".to_string(),
            serde_json::to_value(vec![Message::new_ai_message(
                "I can help you with various tasks. Just ask me anything!"
            )])?,
        );
        Ok(update)
    });

    // Node 4: Generate response
    let generate_response = function_node("generate_response", |state: &MessagesState| async move {
        let mut update = HashMap::new();
        
        // Simple echo response
        if let Some(last_msg) = state.messages.last() {
            let response = format!("You said: {}", last_msg.content);
            update.insert(
                "messages".to_string(),
                serde_json::to_value(vec![Message::new_ai_message(&response)])?,
            );
        }
        
        Ok(update)
    });

    // Build the graph
    let mut graph = StateGraph::<MessagesState>::new();
    
    // Add nodes
    graph.add_node("process_input", process_input)?;
    graph.add_node("decide_action", decide_action)?;
    graph.add_node("provide_help", provide_help)?;
    graph.add_node("generate_response", generate_response)?;
    
    // Add edges
    graph.add_edge(START, "process_input");
    graph.add_edge("process_input", "decide_action");
    
    // Conditional edge from decide_action
    let mut mapping = HashMap::new();
    mapping.insert("help".to_string(), "provide_help".to_string());
    mapping.insert("respond".to_string(), "generate_response".to_string());
    
    graph.add_conditional_edges("decide_action", |state: &MessagesState| async move {
        // Extract action from state
        let state_json = serde_json::to_value(state)
            .map_err(|e| langchain_rust::langgraph::LangGraphError::SerializationError(e))?;
        
        if let Some(action) = state_json.get("action").and_then(|v| v.as_str()) {
            Ok(action.to_string())
        } else {
            Ok("respond".to_string()) // Default
        }
    }, mapping);
    
    graph.add_edge("provide_help", END);
    graph.add_edge("generate_response", END);

    // Compile the graph
    let compiled = graph.compile()?;

    // Example 1: Regular invocation
    println!("=== Example 1: Regular invocation ===");
    let initial_state = MessagesState::with_messages(vec![
        Message::new_human_message("Hello, how are you?")
    ]);
    let final_state = compiled.invoke(initial_state).await?;
    
    println!("Final messages:");
    for message in &final_state.messages {
        println!("  {}: {}", message.message_type.to_string(), message.content);
    }

    // Example 2: Help request
    println!("\n=== Example 2: Help request ===");
    let initial_state = MessagesState::with_messages(vec![
        Message::new_human_message("I need help")
    ]);
    let final_state = compiled.invoke(initial_state).await?;
    
    println!("Final messages:");
    for message in &final_state.messages {
        println!("  {}: {}", message.message_type.to_string(), message.content);
    }

    // Example 3: Streaming execution
    println!("\n=== Example 3: Streaming execution ===");
    use futures::StreamExt;
    let initial_state = MessagesState::with_messages(vec![
        Message::new_human_message("Test streaming")
    ]);
    let mut stream = compiled.stream(initial_state);
    
    while let Some(event) = stream.next().await {
        match event {
            langchain_rust::langgraph::StreamEvent::NodeStart { node, .. } => {
                println!("  → Starting node: {}", node);
            }
            langchain_rust::langgraph::StreamEvent::NodeEnd { node, .. } => {
                println!("  ← Completed node: {}", node);
            }
            langchain_rust::langgraph::StreamEvent::GraphEnd { .. } => {
                println!("  ✓ Graph completed");
            }
            langchain_rust::langgraph::StreamEvent::Error { error } => {
                eprintln!("  ✗ Error: {:?}", error);
            }
        }
    }

    Ok(())
}

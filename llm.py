"""
LLM integration for the OSINT tool.
Handles token counting and AWS Bedrock inference.
"""
import tiktoken

def count_tokens(text):
    """
    Counts the number of tokens in the given text using the tiktoken library.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Number of tokens in the text
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

def generate_conversation(bedrock_client, model_id, messages, guardrail_config):
    """
    Generate a conversation response from Bedrock.
    
    Args:
        bedrock_client: AWS Bedrock client
        model_id: Model ID to use
        messages: Messages for the conversation
        guardrail_config: Guardrail configuration
        
    Returns:
        Response from Bedrock
    """
    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        guardrailConfig=guardrail_config,
        inferenceConfig={"temperature": 0.5},
        additionalModelRequestFields={"top_k": 5}
    )
    return response

def bedrock_inference(bedrock_client, system_prompt, model_id, guardrail_config):
    """
    Get inference response from AWS Bedrock.
    
    Args:
        bedrock_client: AWS Bedrock client
        system_prompt: Prompt to send to the model
        model_id: Model ID to use
        guardrail_config: Guardrail configuration
        
    Returns:
        Tuple of (time taken, response message)
    """
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "guardContent": {
                            "text": {
                                "text": system_prompt
                            }
                        }
                    }
                ]
            }
        ]

        # Count tokens in the input prompt
        system_prompt_tokens = count_tokens(system_prompt)
        print(f"Number of system prompt tokens: {system_prompt_tokens}")
        input_text = system_prompt
        input_tokens = count_tokens(input_text)
        print(f"Number of input tokens: {input_tokens}")

        response = generate_conversation(
            bedrock_client, model_id, messages, guardrail_config)
        
        output_message = response.get('output', {}).get('message', "")

        time = int(response['metrics']['latencyMs']) / 1000
        return time, output_message
    except Exception as e:
        print(f"A client error occurred: {e}")
        return 0, "Error"

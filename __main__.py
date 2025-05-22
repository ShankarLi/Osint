"""
Main entry point for the OSINT tool.
This module coordinates the workflow of the application.
"""
import os
import boto3
import argparse
import prompt_catalog as pcl

from milvus_operations import MilvusOperations
from milvus_processor import MilvusProcessor
from content_fetcher import ContentFetcher
from document_processor import DocumentProcessor
from config import ConfigManager
from llm import bedrock_inference, count_tokens
from datetime import datetime


def parse_arguments():
    """
    Parse command-line arguments for company name and input link file path.
    """
    parser = argparse.ArgumentParser(description="Generate a company report using open-source research.")
    parser.add_argument("--company_name", required=True, help="Name of the company.")
    parser.add_argument("--individual_name", required=True, help="Name of the individual.")
    parser.add_argument("--input_links", required=True, help="Path to the .docx file containing trusted URLs.")
    parser.add_argument("--template_path", required=True, help="template path for the report.")
    return parser.parse_args()


def process_urls(bedrock_client, model_id, guardrail_config, milvus_ops, config_manager, url_list, input_file, entity, prompt, template_path):
    """
    Process a list of URLs, scrape their content, and pass it to the LLM for generating the report.
    Also, push the extracted content to Milvus vector database.
    """
    max_tokens = config_manager.get('MAX_TOKENS')
    template_content = DocumentProcessor.read_template(template_path)
    
    # Initialize the MilvusProcessor
    milvus_processor = MilvusProcessor(
        milvus_ops=milvus_ops,
        collection_name=config_manager.get('COLLECTION_NAME'),
        embedding_model=config_manager.get_embedding_model(),
        batch_size=config_manager.get('BATCH_SIZE'),
        max_text_length=config_manager.get('MAX_TEXT_LENGTH')
    )
    
    # Ensure collection exists
    milvus_processor.ensure_collection_exists(config_manager.get('FIELDS'))
    
    # Initialize content fetcher
    content_fetcher = ContentFetcher()
    
    # Process URLs
    milvus_processor.process_content(urls=url_list, content_fetcher=content_fetcher)
    
    # Create search query and retrieve content from Milvus
    milvus_query = pcl.milvus_query_template.format(entity_name=entity, template_content=template_content)
    search_results = milvus_processor.search(milvus_query, top_k=config_manager.get('TOP_K'))
    
    if isinstance(search_results, list):
        aggregated_content = " ".join(search_results)
    else:
        aggregated_content = search_results
    
    # Check token limit
    if count_tokens(aggregated_content) > max_tokens:
        print("Aggregated content exceeds token limit. Truncating...")
        aggregated_content = aggregated_content[:max_tokens]
    
    # Generate report if content was retrieved
    if aggregated_content:
        generate_report(bedrock_client, model_id, guardrail_config, prompt, entity, template_content, aggregated_content, input_file)
    else:
        print("No content was retrieved from the Milvus database")
    
    # Disconnect from Milvus
    milvus_ops.disconnect_from_milvus()


def generate_report(bedrock_client, model_id, guardrail_config, system_prompt, entity_name, template_content, aggregated_content, input_file_name):
    """
    Generate the report using the LLM and save it to a file.
    """
    print("Passing aggregated content to Bedrock inference...")
    system_prompt = system_prompt.format(entity_name=entity_name, template_content=template_content, aggregated_content=aggregated_content)
    _, response = bedrock_inference(bedrock_client, system_prompt, model_id, guardrail_config)
    output_message = response['content'][0]['text']
    
    # Save the report
    DocumentProcessor.save_report(output_message, input_file_name)


def validate_args(args):
    """
    Determine the entity name and system prompt based on the provided arguments.
    """
    if args.company_name != 'none':
        return args.company_name, pcl.company_system_prompt
    elif args.individual_name != 'none':
        return args.individual_name, pcl.individual_system_prompt
    elif args.company_name == 'none' and args.individual_name == 'none':
        raise ValueError("Please provide either a company name or an individual name.")
    elif args.company_name != 'none' and args.individual_name != 'none':
        raise ValueError("Invalid input. Please provide either a company name or an individual name.")


if __name__ == "__main__":
    # Initialize configuration manager
    config_manager = ConfigManager('constants.yaml')
    
    # Initialize Milvus operations
    milvus_ops = MilvusOperations(
        host=config_manager.get('HOST'),
        port=config_manager.get('PORT'),
        timeout=config_manager.get('TIMEOUT'),
        db_name=config_manager.get('DATABASE')
    )
    milvus_ops.connect_to_milvus()
    
    # Parse command line arguments
    args = parse_arguments()
    entity_name, system_prompt = validate_args(args)
    docx_file_path = args.input_links
    template_path = args.template_path
    
    # Initialize Bedrock client
    bedrock_client = boto3.client(service_name='bedrock-runtime')
    model_id = config_manager.get('MODEL_ID')
    guardrail_config = config_manager.get('GUARDRAIL_CONFIG')
    
    # Process input file
    if os.path.exists(docx_file_path):
        urls = DocumentProcessor.extract_urls(docx_file_path)
        print(f"Found {len(urls)} URLs in the document.")
        process_urls(bedrock_client, model_id, guardrail_config, milvus_ops, config_manager, urls, docx_file_path, entity_name, system_prompt, template_path)
    else:
        print(f"File not found: {docx_file_path}")
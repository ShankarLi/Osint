# OSINT Research Tool

A powerful Open Source Intelligence (OSINT) tool for generating comprehensive research reports on companies and individuals using trusted web sources.

## Overview

The OSINT Research Tool automates the process of collecting, analyzing, and compiling information from trusted web sources. It scrapes content from specified URLs, processes the information using vector databases (Milvus) for efficient retrieval, and generates comprehensive reports using AWS Bedrock AI models.

## Features

- **Web Scraping**: Automatically extracts content from trusted URLs
- **Vector Database Integration**: Uses Milvus to store and efficiently query web content
- **AI-Powered Report Generation**: Leverages AWS Bedrock LLMs to analyze data and create coherent reports
- **Support for Different Entity Types**: Generate reports on companies or individuals
- **Template-Based Reports**: Uses customizable templates for consistent report formatting

## Requirements

- Python 3.8+
- AWS account with Bedrock access
- Milvus vector database
- Required Python packages:
  - pymilvus
  - requests
  - beautifulsoup4
  - boto3
  - tiktoken
  - python-docx

## Project Structure

```
osint/
├── __main__.py             # Entry point for the application
├── config.py               # Configuration management
├── constants.py            # Constants used throughout the application
├── constants.yaml          # YAML configuration file
├── content_fetcher.py      # Web content scraping functionality
├── document_processor.py   # Document reading and output generation
├── input_files/            # Directory for input files with URLs
├── llm.py                  # AWS Bedrock LLM integration
├── milvus_operations.py    # Core Milvus database operations
├── milvus_processor.py     # Milvus processing and querying
├── output_files/           # Directory for generated reports
├── prompt_catalog.py       # Prompt templates for AI inference
└── test.py                 # Test functionality
```

## Usage

```bash
python -m osint --company_name "Company Name" --individual_name "none" --input_links "path/to/urls.docx" --template_path "path/to/template.docx"
```

or for individual research:

```bash
python -m osint --company_name "none" --individual_name "Person Name" --input_links "path/to/urls.docx" --template_path "path/to/template.docx"
```

### Parameters

- `--company_name`: Name of the company to research (use "none" if researching an individual)
- `--individual_name`: Name of the individual to research (use "none" if researching a company)
- `--input_links`: Path to a .docx file containing trusted URLs for research
- `--template_path`: Path to a template file for the report format

## Setup

1. Clone the repository
2. Configure AWS credentials for Bedrock access
3. Set up a Milvus instance (local or cloud)
4. Update the `constants.yaml` file with your configuration
5. Create input files with trusted URLs in the `input_files` directory
6. Create report templates in .docx format

## How It Works

1. The tool reads URLs from a specified input document
2. Content is fetched from each URL and preprocessed
3. Text is embedded and stored in Milvus vector database
4. Relevant information is retrieved based on the search query
5. AWS Bedrock LLM generates a comprehensive report
6. The report is saved as a document in the output directory

## Disclaimer

This tool should only be used for legitimate research purposes with public information from trusted sources. Always respect privacy rights and terms of service of websites when using this tool.

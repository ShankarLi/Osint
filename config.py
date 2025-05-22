"""
Configuration manager for the OSINT tool.
Loads and provides access to configuration settings from YAML file.
"""
import yaml
from sentence_transformers import SentenceTransformer

class ConfigManager:
    """Manages configuration settings for the OSINT application."""
    
    def __init__(self, config_path='constants.yaml'):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(self.config['EMBEDDING_MODEL'])
    
    def get(self, key, default=None):
        """
        Get a configuration value.
        
        Args:
            key: The configuration key to retrieve
            default: Default value if key is not found
            
        Returns:
            The configuration value
        """
        return self.config.get(key, default)
    
    def get_embedding_model(self):
        """Get the initialized embedding model."""
        return self.embedding_model

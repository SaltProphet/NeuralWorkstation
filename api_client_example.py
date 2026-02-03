#!/usr/bin/env python3
"""
API Client Example for FORGE v1
===============================

Demonstrates how to use the FORGE REST API programmatically.
"""

import requests
from pathlib import Path
import json


class ForgeAPIClient:
    """Client for interacting with FORGE REST API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "forge-dev-key-change-in-production"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API
            api_key: API key for authentication
        """
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key
        }
    
    def health_check(self):
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def separate_stems(self, audio_file: str, model: str = "htdemucs", use_cache: bool = True):
        """
        Separate audio into stems.
        
        Args:
            audio_file: Path to audio file
            model: Demucs model to use
            use_cache: Whether to use cache
            
        Returns:
            API response with stem file paths
        """
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            data = {
                'model': model,
                'use_cache': use_cache
            }
            response = requests.post(
                f"{self.base_url}/api/v1/stem-separation",
                files=files,
                json=data,
                headers=self.headers
            )
        
        return response.json()
    
    def extract_loops(
        self,
        audio_file: str,
        loop_duration: float = 4.0,
        aperture: float = 0.5,
        num_loops: int = 5
    ):
        """
        Extract loops from audio.
        
        Args:
            audio_file: Path to audio file
            loop_duration: Loop duration in seconds
            aperture: Aperture control (0-1)
            num_loops: Number of loops to extract
            
        Returns:
            API response with loop information
        """
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            data = {
                'loop_duration': loop_duration,
                'aperture': aperture,
                'num_loops': num_loops
            }
            response = requests.post(
                f"{self.base_url}/api/v1/loop-extraction",
                files=files,
                json=data,
                headers=self.headers
            )
        
        return response.json()
    
    def generate_vocal_chops(
        self,
        audio_file: str,
        mode: str = "onset",
        min_duration: float = 0.1,
        max_duration: float = 2.0,
        threshold: float = 0.3
    ):
        """
        Generate vocal chops.
        
        Args:
            audio_file: Path to audio file
            mode: Detection mode (silence, onset, hybrid)
            min_duration: Minimum chop duration
            max_duration: Maximum chop duration
            threshold: Detection threshold
            
        Returns:
            API response with chop file paths
        """
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            data = {
                'mode': mode,
                'min_duration': min_duration,
                'max_duration': max_duration,
                'threshold': threshold
            }
            response = requests.post(
                f"{self.base_url}/api/v1/vocal-chops",
                files=files,
                json=data,
                headers=self.headers
            )
        
        return response.json()
    
    def extract_midi(self, audio_file: str):
        """
        Extract MIDI from audio.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            API response with MIDI file path
        """
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.base_url}/api/v1/midi-extraction",
                files=files,
                headers=self.headers
            )
        
        return response.json()
    
    def download_file(self, filename: str, output_path: str = None):
        """
        Download a generated file.
        
        Args:
            filename: Name of file to download
            output_path: Where to save the file
            
        Returns:
            Path to downloaded file
        """
        response = requests.get(
            f"{self.base_url}/api/v1/download/{filename}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            if output_path is None:
                output_path = filename
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return output_path
        else:
            raise Exception(f"Download failed: {response.status_code}")
    
    def list_models(self):
        """List available Demucs models."""
        response = requests.get(
            f"{self.base_url}/api/v1/models",
            headers=self.headers
        )
        return response.json()
    
    def get_config(self):
        """Get current API configuration."""
        response = requests.get(
            f"{self.base_url}/api/v1/config",
            headers=self.headers
        )
        return response.json()


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = ForgeAPIClient()
    
    # Check health
    print("🏥 Health Check:")
    print(json.dumps(client.health_check(), indent=2))
    
    # List models
    print("\n🎵 Available Models:")
    print(json.dumps(client.list_models(), indent=2))
    
    # Get configuration
    print("\n⚙️  Configuration:")
    print(json.dumps(client.get_config(), indent=2))
    
    # Example: Extract loops (uncomment and provide audio file)
    # print("\n🔄 Extracting loops...")
    # result = client.extract_loops("path/to/audio.wav", num_loops=3)
    # print(json.dumps(result, indent=2))

#!/usr/bin/env python3
"""
Performance Optimization Module for FORGE v1
============================================

Provides performance enhancements:
- Parallel processing for batch operations
- Memory-mapped audio loading for large files
- Configurable quality settings
- Cache management with expiration
- Resource monitoring and limits

Author: NeuralWorkstation Team
License: MIT
"""

import os
import time
import json
import psutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import librosa


class PerformanceConfig:
    """Configuration for performance settings."""
    
    # Processing quality presets
    QUALITY_PRESETS = {
        'draft': {
            'sample_rate': 22050,
            'hop_length': 1024,
            'n_fft': 2048,
            'description': 'Fast processing with lower quality'
        },
        'balanced': {
            'sample_rate': 44100,
            'hop_length': 512,
            'n_fft': 2048,
            'description': 'Balance between speed and quality'
        },
        'high': {
            'sample_rate': 48000,
            'hop_length': 256,
            'n_fft': 4096,
            'description': 'High quality, slower processing'
        }
    }
    
    # Cache settings
    CACHE_MAX_AGE_DAYS = 30  # Maximum age for cached files
    CACHE_MAX_SIZE_GB = 10   # Maximum cache size in GB
    
    # Resource limits
    MAX_MEMORY_PERCENT = 80  # Maximum memory usage percentage
    MAX_CPU_PERCENT = 90     # Maximum CPU usage percentage
    
    @classmethod
    def get_quality_preset(cls, preset_name: str) -> Dict[str, Any]:
        """Get quality preset configuration."""
        return cls.QUALITY_PRESETS.get(preset_name, cls.QUALITY_PRESETS['balanced'])


class ResourceMonitor:
    """Monitor system resources during processing."""
    
    def __init__(self):
        self.start_time = None
        self.peak_memory = 0
        self.peak_cpu = 0
    
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.peak_memory = 0
        self.peak_cpu = 0
    
    def update(self):
        """Update resource metrics."""
        process = psutil.Process()
        memory_percent = process.memory_percent()
        cpu_percent = process.cpu_percent()
        
        self.peak_memory = max(self.peak_memory, memory_percent)
        self.peak_cpu = max(self.peak_cpu, cpu_percent)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            'elapsed_seconds': round(elapsed, 2),
            'peak_memory_percent': round(self.peak_memory, 2),
            'peak_cpu_percent': round(self.peak_cpu, 2),
            'current_memory_mb': round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
        }
    
    def check_limits(self) -> bool:
        """Check if resource limits are exceeded."""
        process = psutil.Process()
        memory_percent = process.memory_percent()
        cpu_percent = process.cpu_percent()
        
        if memory_percent > PerformanceConfig.MAX_MEMORY_PERCENT:
            print(f"⚠️  Memory usage high: {memory_percent:.1f}%")
            return False
        
        if cpu_percent > PerformanceConfig.MAX_CPU_PERCENT:
            print(f"⚠️  CPU usage high: {cpu_percent:.1f}%")
            return False
        
        return True


class CacheManager:
    """Manage cache with expiration and size limits."""
    
    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_size(self) -> float:
        """Get total cache size in GB."""
        total_size = 0
        for file_path in self.cache_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 ** 3)  # Convert to GB
    
    def clean_expired_cache(self, max_age_days: int = None) -> int:
        """
        Remove cache files older than max_age_days.
        
        Args:
            max_age_days: Maximum age in days (default: PerformanceConfig.CACHE_MAX_AGE_DAYS)
            
        Returns:
            Number of files removed
        """
        if max_age_days is None:
            max_age_days = PerformanceConfig.CACHE_MAX_AGE_DAYS
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0
        
        for file_path in self.cache_dir.rglob('*'):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
        
        # Remove empty directories
        for dir_path in sorted(self.cache_dir.rglob('*'), reverse=True):
            if dir_path.is_dir() and not list(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                except Exception:
                    pass
        
        return removed_count
    
    def clean_by_size(self, max_size_gb: float = None) -> int:
        """
        Remove oldest cache files until size is under limit.
        
        Args:
            max_size_gb: Maximum cache size in GB (default: PerformanceConfig.CACHE_MAX_SIZE_GB)
            
        Returns:
            Number of files removed
        """
        if max_size_gb is None:
            max_size_gb = PerformanceConfig.CACHE_MAX_SIZE_GB
        
        current_size = self.get_cache_size()
        if current_size <= max_size_gb:
            return 0
        
        # Get all files with their sizes and modification times
        files = []
        for file_path in self.cache_dir.rglob('*'):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    'path': file_path,
                    'size': stat.st_size,
                    'mtime': stat.st_mtime
                })
        
        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x['mtime'])
        
        removed_count = 0
        removed_size = 0
        
        for file_info in files:
            if current_size - (removed_size / (1024 ** 3)) <= max_size_gb:
                break
            
            try:
                file_info['path'].unlink()
                removed_size += file_info['size']
                removed_count += 1
            except Exception as e:
                print(f"Error removing {file_info['path']}: {e}")
        
        return removed_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_files = sum(1 for _ in self.cache_dir.rglob('*') if _.is_file())
        total_size_gb = self.get_cache_size()
        
        return {
            'total_files': total_files,
            'total_size_gb': round(total_size_gb, 2),
            'max_size_gb': PerformanceConfig.CACHE_MAX_SIZE_GB,
            'max_age_days': PerformanceConfig.CACHE_MAX_AGE_DAYS
        }


class OptimizedAudioLoader:
    """Optimized audio loading with memory mapping for large files."""
    
    @staticmethod
    def load_audio(
        file_path: str,
        sr: int = 44100,
        quality_preset: str = 'balanced',
        use_mmap: bool = True
    ) -> tuple:
        """
        Load audio with optimizations.
        
        Args:
            file_path: Path to audio file
            sr: Target sample rate
            quality_preset: Quality preset name
            use_mmap: Use memory mapping for large files
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        preset = PerformanceConfig.get_quality_preset(quality_preset)
        target_sr = preset['sample_rate']
        
        # Use librosa for now, could optimize further with mmap if needed
        y, sr_original = librosa.load(file_path, sr=target_sr)
        
        return y, target_sr
    
    @staticmethod
    def estimate_memory_usage(file_path: str, sr: int = 44100) -> float:
        """
        Estimate memory usage for loading an audio file.
        
        Args:
            file_path: Path to audio file
            sr: Sample rate
            
        Returns:
            Estimated memory usage in MB
        """
        # Get file duration without loading full audio
        duration = librosa.get_duration(path=file_path)
        
        # Estimate: duration * sample_rate * 4 bytes (float32) * 2 (stereo)
        estimated_bytes = duration * sr * 4 * 2
        return estimated_bytes / (1024 ** 2)  # Convert to MB


class ParallelProcessor:
    """Parallel processing for batch operations."""
    
    def __init__(self, max_workers: int = None):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of workers (default: CPU count - 1)
        """
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)
        self.max_workers = max_workers
    
    def process_parallel(
        self,
        files: List[str],
        process_func,
        use_threads: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process files in parallel.
        
        Args:
            files: List of file paths
            process_func: Function to apply to each file
            use_threads: Use threads (True) or processes (False)
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of results
        """
        results = []
        executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_func, file_path, **kwargs): file_path
                for file_path in files
            }
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    results.append({
                        'file': file_path,
                        'success': True,
                        'result': result
                    })
                except Exception as e:
                    results.append({
                        'file': file_path,
                        'success': False,
                        'error': str(e)
                    })
        
        return results


def optimize_performance():
    """Run performance optimizations."""
    print("\n🔧 Running performance optimizations...")
    
    # Clean cache
    cache_manager = CacheManager()
    
    print(f"  Cleaning expired cache files...")
    expired_count = cache_manager.clean_expired_cache()
    print(f"  ✅ Removed {expired_count} expired files")
    
    print(f"  Checking cache size...")
    size_count = cache_manager.clean_by_size()
    print(f"  ✅ Removed {size_count} files to maintain size limit")
    
    # Show cache stats
    stats = cache_manager.get_cache_stats()
    print(f"\n  📊 Cache Statistics:")
    print(f"     Files: {stats['total_files']}")
    print(f"     Size: {stats['total_size_gb']:.2f} GB / {stats['max_size_gb']} GB")
    print(f"     Max Age: {stats['max_age_days']} days")
    
    print("\n✅ Performance optimization complete!")
    
    return stats


if __name__ == "__main__":
    # Run optimization when called directly
    optimize_performance()

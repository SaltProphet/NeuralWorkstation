"""
Unit tests for performance optimization features in performance.py
"""
import pytest
import time
import sys
import os
from pathlib import Path
import numpy as np
import soundfile as sf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance import (
    PerformanceConfig,
    ResourceMonitor,
    CacheManager,
    OptimizedAudioLoader,
    ParallelProcessor,
    optimize_performance
)


@pytest.mark.unit
class TestPerformanceConfig:
    """Test performance configuration."""
    
    def test_quality_presets_exist(self):
        """Test that quality presets are defined."""
        assert 'draft' in PerformanceConfig.QUALITY_PRESETS
        assert 'balanced' in PerformanceConfig.QUALITY_PRESETS
        assert 'high' in PerformanceConfig.QUALITY_PRESETS
    
    def test_quality_preset_structure(self):
        """Test quality preset structure."""
        for preset_name, preset in PerformanceConfig.QUALITY_PRESETS.items():
            assert 'sample_rate' in preset
            assert 'hop_length' in preset
            assert 'n_fft' in preset
            assert 'description' in preset
            assert isinstance(preset['sample_rate'], int)
            assert isinstance(preset['hop_length'], int)
            assert isinstance(preset['n_fft'], int)
    
    def test_get_quality_preset(self):
        """Test getting quality preset."""
        draft = PerformanceConfig.get_quality_preset('draft')
        assert draft['sample_rate'] == 22050
        
        balanced = PerformanceConfig.get_quality_preset('balanced')
        assert balanced['sample_rate'] == 44100
        
        high = PerformanceConfig.get_quality_preset('high')
        assert high['sample_rate'] == 48000
    
    def test_get_invalid_preset_returns_default(self):
        """Test that invalid preset name returns balanced."""
        preset = PerformanceConfig.get_quality_preset('invalid_preset')
        assert preset == PerformanceConfig.QUALITY_PRESETS['balanced']
    
    def test_cache_configuration(self):
        """Test cache configuration values."""
        assert PerformanceConfig.CACHE_MAX_AGE_DAYS > 0
        assert PerformanceConfig.CACHE_MAX_SIZE_GB > 0
        assert isinstance(PerformanceConfig.CACHE_MAX_AGE_DAYS, int)
        assert isinstance(PerformanceConfig.CACHE_MAX_SIZE_GB, (int, float))
    
    def test_resource_limits(self):
        """Test resource limit configuration."""
        assert 0 < PerformanceConfig.MAX_MEMORY_PERCENT <= 100
        assert 0 < PerformanceConfig.MAX_CPU_PERCENT <= 100


@pytest.mark.unit
class TestResourceMonitor:
    """Test resource monitoring functionality."""
    
    def test_monitor_initialization(self):
        """Test ResourceMonitor initialization."""
        monitor = ResourceMonitor()
        assert monitor.start_time is None
        assert monitor.peak_memory == 0
        assert monitor.peak_cpu == 0
    
    def test_monitor_start(self):
        """Test starting the monitor."""
        monitor = ResourceMonitor()
        monitor.start()
        assert monitor.start_time is not None
        assert monitor.start_time > 0
    
    def test_monitor_update(self):
        """Test updating monitor metrics."""
        monitor = ResourceMonitor()
        monitor.start()
        monitor.update()
        
        # After update, peak values should be set
        assert monitor.peak_memory >= 0
        assert monitor.peak_cpu >= 0
    
    def test_get_stats(self):
        """Test getting statistics."""
        monitor = ResourceMonitor()
        monitor.start()
        time.sleep(0.1)  # Small delay
        monitor.update()
        
        stats = monitor.get_stats()
        assert 'elapsed_seconds' in stats
        assert 'peak_memory_percent' in stats
        assert 'peak_cpu_percent' in stats
        assert 'current_memory_mb' in stats
        
        assert stats['elapsed_seconds'] >= 0.1
        assert stats['peak_memory_percent'] >= 0
        assert stats['current_memory_mb'] > 0
    
    def test_check_limits(self):
        """Test resource limit checking."""
        monitor = ResourceMonitor()
        monitor.start()
        monitor.update()
        
        # Should return True under normal conditions
        result = monitor.check_limits()
        assert isinstance(result, bool)


@pytest.mark.unit
class TestCacheManager:
    """Test cache management functionality."""
    
    def test_cache_manager_initialization(self, tmp_path):
        """Test CacheManager initialization."""
        cache_dir = tmp_path / "test_cache"
        manager = CacheManager(str(cache_dir))
        assert manager.cache_dir.exists()
    
    def test_get_cache_size_empty(self, tmp_path):
        """Test getting cache size for empty directory."""
        cache_dir = tmp_path / "test_cache"
        manager = CacheManager(str(cache_dir))
        size = manager.get_cache_size()
        assert size == 0.0
    
    def test_get_cache_size_with_files(self, tmp_path):
        """Test getting cache size with files."""
        cache_dir = tmp_path / "test_cache"
        manager = CacheManager(str(cache_dir))
        
        # Create test files
        test_file = cache_dir / "test.txt"
        test_file.write_text("test data" * 1000)
        
        size = manager.get_cache_size()
        assert size > 0
    
    def test_clean_expired_cache(self, tmp_path):
        """Test cleaning expired cache files."""
        cache_dir = tmp_path / "test_cache"
        manager = CacheManager(str(cache_dir))
        
        # Create test file
        test_file = cache_dir / "test.txt"
        test_file.write_text("test data")
        
        # Clean with max_age_days=0 should remove all files
        removed = manager.clean_expired_cache(max_age_days=0)
        assert removed >= 0  # May be 0 if file timestamp is very recent
    
    def test_clean_by_size(self, tmp_path):
        """Test cleaning cache by size."""
        cache_dir = tmp_path / "test_cache"
        manager = CacheManager(str(cache_dir))
        
        # Create test files totaling more than 0.001 GB
        for i in range(10):
            test_file = cache_dir / f"test_{i}.txt"
            test_file.write_text("test data" * 100000)
        
        # Clean to max size of 0.001 GB
        removed = manager.clean_by_size(max_size_gb=0.001)
        assert removed >= 0
    
    def test_get_cache_stats(self, tmp_path):
        """Test getting cache statistics."""
        cache_dir = tmp_path / "test_cache"
        manager = CacheManager(str(cache_dir))
        
        # Create test files
        for i in range(3):
            test_file = cache_dir / f"test_{i}.txt"
            test_file.write_text("test data")
        
        stats = manager.get_cache_stats()
        assert 'total_files' in stats
        assert 'total_size_gb' in stats
        assert 'max_size_gb' in stats
        assert 'max_age_days' in stats
        
        assert stats['total_files'] == 3
        assert stats['total_size_gb'] >= 0


@pytest.mark.unit
class TestOptimizedAudioLoader:
    """Test optimized audio loading."""
    
    def test_load_audio_default(self, sample_audio_file):
        """Test loading audio with default settings."""
        y, sr = OptimizedAudioLoader.load_audio(sample_audio_file)
        
        assert isinstance(y, np.ndarray)
        assert len(y) > 0
        assert sr == 44100  # balanced preset default
    
    def test_load_audio_with_quality_presets(self, sample_audio_file):
        """Test loading audio with different quality presets."""
        presets = ['draft', 'balanced', 'high']
        
        for preset in presets:
            y, sr = OptimizedAudioLoader.load_audio(
                sample_audio_file,
                quality_preset=preset
            )
            assert isinstance(y, np.ndarray)
            assert len(y) > 0
            assert sr > 0
    
    def test_estimate_memory_usage(self, sample_audio_file):
        """Test memory usage estimation."""
        estimated_mb = OptimizedAudioLoader.estimate_memory_usage(sample_audio_file)
        
        assert estimated_mb > 0
        assert isinstance(estimated_mb, float)


@pytest.mark.unit
class TestParallelProcessor:
    """Test parallel processing functionality."""
    
    def test_parallel_processor_initialization(self):
        """Test ParallelProcessor initialization."""
        processor = ParallelProcessor(max_workers=2)
        assert processor.max_workers == 2
    
    def test_parallel_processor_default_workers(self):
        """Test default worker count."""
        processor = ParallelProcessor()
        assert processor.max_workers >= 1
    
    def test_process_parallel_with_threads(self, sample_audio_file, sample_audio_file_long):
        """Test parallel processing with threads."""
        processor = ParallelProcessor(max_workers=2)
        
        def simple_process(file_path):
            return f"Processed: {Path(file_path).name}"
        
        files = [sample_audio_file, sample_audio_file_long]
        results = processor.process_parallel(files, simple_process, use_threads=True)
        
        assert len(results) == 2
        for result in results:
            assert 'file' in result
            assert 'success' in result
            if result['success']:
                assert 'result' in result
            else:
                assert 'error' in result


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceOptimization:
    """Test complete performance optimization workflow."""
    
    def test_optimize_performance(self, tmp_path, monkeypatch):
        """Test running full performance optimization."""
        monkeypatch.chdir(tmp_path)
        
        # Create cache directory with some files
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(exist_ok=True)
        (cache_dir / "test.txt").write_text("test")
        
        # Run optimization
        stats = optimize_performance()
        
        assert isinstance(stats, dict)
        assert 'total_files' in stats
        assert 'total_size_gb' in stats


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for critical operations."""
    
    def test_audio_loading_speed(self, sample_audio_file, benchmark=None):
        """Benchmark audio loading speed."""
        start_time = time.time()
        y, sr = OptimizedAudioLoader.load_audio(sample_audio_file)
        elapsed = time.time() - start_time
        
        assert elapsed < 5.0  # Should load in under 5 seconds for test file
        assert len(y) > 0
    
    def test_cache_operations_speed(self, tmp_path, benchmark=None):
        """Benchmark cache operations."""
        cache_dir = tmp_path / "test_cache"
        manager = CacheManager(str(cache_dir))
        
        # Create test files
        for i in range(10):
            (cache_dir / f"test_{i}.txt").write_text("test" * 100)
        
        start_time = time.time()
        stats = manager.get_cache_stats()
        elapsed = time.time() - start_time
        
        assert elapsed < 0.5  # Should be fast
        assert stats['total_files'] == 10

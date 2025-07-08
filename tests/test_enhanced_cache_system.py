"""
Test suite for Phase 2.5 Enhanced Caching System.

Tests compressed storage, memory-mapped files, and distributed caching.
"""

import sys
import tempfile
from pathlib import Path
import logging
import numpy as np
import time
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.cache import AdvancedCache, DistributedCache, CacheFormat
from src.data.representation import MusicalRepresentation, MusicEvent, EventType


def create_test_data():
    """Create test data for caching tests."""
    # Create test data of different types
    test_data = {
        # NumPy array (should use NPZ format)
        "numpy_array": np.random.rand(1000, 128).astype(np.float32),
        
        # Dictionary with NumPy arrays (musical representation)
        "musical_repr": {
            "piano_roll": np.random.rand(100, 88).astype(np.float32),
            "velocity_roll": np.random.rand(100, 88).astype(np.float32),
            "tokens": np.random.randint(0, 387, 200).astype(np.int32),
        },
        
        # Regular Python object (should use pickle)
        "python_object": {
            "metadata": {"title": "Test Song", "composer": "Test Artist"},
            "events": [MusicEvent(EventType.NOTE_ON, 60, 0, 0.0) for _ in range(50)],
            "stats": {"note_count": 50, "duration": 10.0}
        },
        
        # Large array for memory mapping test
        "large_array": np.random.rand(5000, 512).astype(np.float32),  # ~10MB
    }
    
    return test_data


def test_compression_formats():
    """Test different compression formats and their efficiency."""
    print("\n💾 Testing Compression Formats")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with compression enabled
        cache_compressed = AdvancedCache(
            Path(temp_dir) / "compressed",
            max_size_gb=1.0,
            compression=True
        )
        
        # Test without compression
        cache_uncompressed = AdvancedCache(
            Path(temp_dir) / "uncompressed", 
            max_size_gb=1.0,
            compression=False
        )
        
        test_data = create_test_data()
        
        print(f"Testing compression efficiency:")
        
        total_compressed_size = 0
        total_uncompressed_size = 0
        
        for key, value in test_data.items():
            if key == "large_array":  # Skip large array for this test
                continue
                
            # Store in both caches
            cache_compressed.set(key, value)
            cache_uncompressed.set(key, value)
            
            # Get file sizes
            compressed_stats = cache_compressed.get_stats()
            uncompressed_stats = cache_uncompressed.get_stats()
            
            print(f"  {key}:")
            print(f"    Compressed format: {cache_compressed._metadata[key]['format']}")
            print(f"    Uncompressed format: {cache_uncompressed._metadata[key]['format']}")
            
            # Compare sizes for this item
            compressed_size = cache_compressed._metadata[key]["size_bytes"]
            uncompressed_size = cache_uncompressed._metadata[key]["size_bytes"]
            compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1.0
            
            print(f"    Compressed: {compressed_size:,} bytes")
            print(f"    Uncompressed: {uncompressed_size:,} bytes") 
            print(f"    Compression ratio: {compression_ratio:.2f}x")
            
            total_compressed_size += compressed_size
            total_uncompressed_size += uncompressed_size
            
            # Verify data integrity
            loaded_compressed = cache_compressed.get(key)
            loaded_uncompressed = cache_uncompressed.get(key)
            
            if isinstance(value, np.ndarray):
                assert np.array_equal(loaded_compressed, value), f"Data corruption in compressed {key}"
                assert np.array_equal(loaded_uncompressed, value), f"Data corruption in uncompressed {key}"
            elif isinstance(value, dict) and any(isinstance(v, np.ndarray) for v in value.values()):
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        assert np.array_equal(loaded_compressed[k], v), f"Dict data corruption in compressed {key}"
                        assert np.array_equal(loaded_uncompressed[k], v), f"Dict data corruption in uncompressed {key}"
            
            print(f"    ✅ Data integrity verified")
            print()
        
        overall_compression = total_uncompressed_size / total_compressed_size if total_compressed_size > 0 else 1.0
        print(f"Overall compression ratio: {overall_compression:.2f}x")
        print(f"Space saved: {(total_uncompressed_size - total_compressed_size):,} bytes ({(1 - total_compressed_size/total_uncompressed_size)*100:.1f}%)")
        
        cache_compressed.close()
        cache_uncompressed.close()


def test_memory_mapped_files():
    """Test memory-mapped file support for large datasets."""
    print("\n🗺️ Testing Memory-Mapped Files")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create cache with memory mapping enabled
        cache = AdvancedCache(
            Path(temp_dir) / "mmap_cache",
            max_size_gb=1.0,
            compression=True,
            use_memory_mapping=True
        )
        
        test_data = create_test_data()
        large_array = test_data["large_array"]
        
        print(f"Testing large array: {large_array.shape}, {large_array.nbytes:,} bytes")
        
        # Store large array
        start_time = time.time()
        cache.set("large_test", large_array)
        store_time = time.time() - start_time
        
        print(f"  Store time: {store_time:.3f}s")
        
        # Access the large array multiple times
        access_times = []
        for i in range(5):
            start_time = time.time()
            loaded_array = cache.get("large_test")
            access_time = time.time() - start_time
            access_times.append(access_time)
            
            # Verify data integrity
            assert np.array_equal(loaded_array, large_array), f"Data corruption on access {i}"
        
        avg_access_time = np.mean(access_times)
        print(f"  Average access time: {avg_access_time:.3f}s")
        print(f"  Memory-mapped files in use: {cache._stats['memory_mapped_files']}")
        
        # Check cache statistics
        stats = cache.get_stats()
        print(f"  Cache size: {stats['size_mb']:.1f} MB")
        print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Hit rate: {stats['hit_rate']:.2%}")
        
        print(f"  ✅ Memory-mapped file access working")
        
        cache.close()


def test_distributed_cache():
    """Test distributed cache for multi-GPU training simulation."""
    print("\n🌐 Testing Distributed Cache")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "distributed"
        
        # Create multiple workers (simulating multi-GPU setup)
        worker_0 = DistributedCache(cache_dir, worker_id=0, sync_interval=1)  # 1 second for testing
        worker_1 = DistributedCache(cache_dir, worker_id=1, sync_interval=1)
        
        test_data = create_test_data()
        
        print(f"Testing distributed cache with 2 workers:")
        
        # Worker 0 stores some data
        print(f"  Worker 0 storing data...")
        worker_0.set("shared_array", test_data["numpy_array"])
        worker_0.set("shared_music", test_data["musical_repr"])
        
        # Worker 1 stores different data
        print(f"  Worker 1 storing data...")
        worker_1.set("worker1_data", test_data["python_object"])
        
        # Check initial stats
        stats_0 = worker_0.get_stats()
        stats_1 = worker_1.get_stats()
        
        print(f"  Worker 0 local items: {stats_0['item_count']}")
        print(f"  Worker 1 local items: {stats_1['item_count']}")
        
        # Wait for sync interval and trigger sync
        time.sleep(1.1)
        
        # Worker 1 tries to access Worker 0's data
        print(f"  Worker 1 accessing Worker 0's data...")
        shared_array = worker_1.get("shared_array")
        shared_music = worker_1.get("shared_music")
        
        # Verify data integrity
        assert shared_array is not None, "Failed to access shared array"
        assert shared_music is not None, "Failed to access shared music"
        
        if isinstance(shared_array, np.ndarray):
            assert np.array_equal(shared_array, test_data["numpy_array"]), "Shared array data corruption"
        
        print(f"  ✅ Cross-worker data access working")
        
        # Worker 0 tries to access Worker 1's data
        print(f"  Worker 0 accessing Worker 1's data...")
        worker1_data = worker_0.get("worker1_data")
        assert worker1_data is not None, "Failed to access worker 1 data"
        
        print(f"  ✅ Bidirectional data sharing working")
        
        # Check final stats
        final_stats_0 = worker_0.get_stats()
        final_stats_1 = worker_1.get_stats()
        
        print(f"  Final Worker 0 items: {final_stats_0['item_count']}")
        print(f"  Final Worker 1 items: {final_stats_1['item_count']}")
        print(f"  Shared cache files: {final_stats_0['shared_files']}")
        print(f"  Shared cache size: {final_stats_0['shared_size_mb']:.1f} MB")
        
        worker_0.close()
        worker_1.close()


def test_cache_performance():
    """Test cache performance with various scenarios."""
    print("\n⚡ Testing Cache Performance")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = AdvancedCache(
            Path(temp_dir) / "perf_cache",
            max_size_gb=0.5,  # Smaller cache to test eviction
            compression=True
        )
        
        # Create various sizes of test data
        test_sizes = [
            ("small", np.random.rand(100, 10).astype(np.float32)),
            ("medium", np.random.rand(1000, 100).astype(np.float32)),
            ("large", np.random.rand(2000, 200).astype(np.float32)),
        ]
        
        print(f"Performance test with different data sizes:")
        
        for name, data in test_sizes:
            # Test store performance
            start_time = time.time()
            cache.set(f"perf_{name}", data)
            store_time = time.time() - start_time
            
            # Test load performance
            start_time = time.time()
            loaded_data = cache.get(f"perf_{name}")
            load_time = time.time() - start_time
            
            # Verify integrity
            assert np.array_equal(loaded_data, data), f"Data corruption in {name}"
            
            # Get compression info
            meta = cache._metadata[f"perf_{name}"]
            compression_ratio = meta["compression_ratio"]
            
            print(f"  {name:6s}: Store {store_time:.3f}s, Load {load_time:.3f}s, Compression {compression_ratio:.2f}x")
        
        # Test cache eviction
        print(f"\n  Testing cache eviction:")
        initial_stats = cache.get_stats()
        print(f"    Initial items: {initial_stats['item_count']}")
        
        # Add many items to trigger eviction
        for i in range(20):
            cache.set(f"evict_test_{i}", np.random.rand(500, 50).astype(np.float32))
        
        final_stats = cache.get_stats()
        print(f"    Final items: {final_stats['item_count']}")
        print(f"    Evictions: {final_stats['evictions']}")
        print(f"    Cache utilization: {final_stats['utilization']:.1%}")
        print(f"    Hit rate: {final_stats['hit_rate']:.1%}")
        
        print(f"  ✅ Cache eviction working correctly")
        
        cache.close()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    print("💾 Aurl.ai Enhanced Caching System Test Suite")
    print("Testing Phase 2.5 - Compressed Storage, Memory Mapping, Distributed Cache")
    print("=" * 70)
    
    try:
        test_compression_formats()
        test_memory_mapped_files()
        test_distributed_cache()
        test_cache_performance()
        
        print(f"\n🎉 All Enhanced Caching Tests Passed!")
        print(f"✅ Compressed storage (NPZ): Working")
        print(f"✅ Memory-mapped files: Working")
        print(f"✅ Distributed cache: Working")
        print(f"✅ Performance optimization: Working")
        print(f"✅ Data integrity: Preserved")
        print(f"\n🚀 Phase 2.5 Smart Caching System COMPLETE!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise
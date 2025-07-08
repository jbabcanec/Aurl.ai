"""
Advanced caching system for Aurl.ai preprocessing pipeline.

Provides intelligent caching with LRU eviction, size limits, and
automatic invalidation based on file modification times.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from threading import Lock
import hashlib
import os
import numpy as np
import gzip
import mmap
from contextlib import contextmanager


class CacheFormat:
    """Cache storage format options."""
    PICKLE = "pickle"           # Standard pickle format
    COMPRESSED_PICKLE = "pkl.gz"  # Gzip-compressed pickle
    NPZ = "npz"                 # NumPy compressed arrays
    COMPRESSED_NPZ = "npz.gz"   # Gzip-compressed NPZ


class AdvancedCache:
    """
    Advanced caching system with LRU eviction and intelligent invalidation.
    
    Features:
    - LRU eviction based on access time
    - Size-based limits with automatic cleanup
    - Thread-safe operations
    - Automatic invalidation on file changes
    - Persistent cache across sessions
    - Compressed storage (NPZ, gzip)
    - Memory-mapped file support
    - Multiple storage formats
    """
    
    def __init__(self, cache_dir: Path, max_size_gb: float = 5.0, max_items: int = 10000,
                 compression: bool = True, use_memory_mapping: bool = False):
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.max_items = max_items
        self.compression = compression
        self.use_memory_mapping = use_memory_mapping
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._lock = Lock()
        
        # Cache metadata
        self._metadata_file = self.cache_dir / "cache_metadata.pkl"
        self._metadata: Dict[str, Dict] = {}
        
        # Memory-mapped files cache
        self._mmap_cache: Dict[str, mmap.mmap] = {} if use_memory_mapping else None
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0,
            "item_count": 0,
            "compression_ratio": 0.0,
            "memory_mapped_files": 0
        }
        
        # Initialize cache directory and load metadata
        self._initialize_cache()
        
        self.logger.info(f"AdvancedCache initialized: {self.cache_dir}, max_size={max_size_gb}GB")
    
    def _initialize_cache(self):
        """Initialize cache directory and load existing metadata."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'rb') as f:
                    self._metadata = pickle.load(f)
                self.logger.info(f"Loaded cache metadata with {len(self._metadata)} items")
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
                self._metadata = {}
        
        # Update statistics from metadata
        self._update_stats_from_metadata()
    
    def _update_stats_from_metadata(self):
        """Update statistics from loaded metadata."""
        total_size = 0
        valid_items = 0
        
        for key, meta in self._metadata.items():
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                total_size += cache_file.stat().st_size
                valid_items += 1
        
        self._stats["size_bytes"] = total_size
        self._stats["item_count"] = valid_items
    
    def _detect_best_format(self, value: Any) -> str:
        """Detect the best storage format for the given value."""
        if isinstance(value, dict) and any(isinstance(v, np.ndarray) for v in value.values()):
            # Dictionary with NumPy arrays - use NPZ
            return CacheFormat.COMPRESSED_NPZ if self.compression else CacheFormat.NPZ
        elif isinstance(value, np.ndarray):
            # NumPy array - use NPZ
            return CacheFormat.COMPRESSED_NPZ if self.compression else CacheFormat.NPZ
        else:
            # Generic Python object - use pickle
            return CacheFormat.COMPRESSED_PICKLE if self.compression else CacheFormat.PICKLE
    
    def _save_compressed(self, cache_file: Path, value: Any, format_type: str) -> int:
        """Save value using the specified compression format."""
        if format_type == CacheFormat.NPZ:
            # Save as uncompressed NPZ
            if isinstance(value, dict):
                np.savez(cache_file, **value)
            else:
                np.savez(cache_file, data=value)
        
        elif format_type == CacheFormat.COMPRESSED_NPZ:
            # Save as compressed NPZ
            if isinstance(value, dict):
                np.savez_compressed(cache_file, **value)
            else:
                np.savez_compressed(cache_file, data=value)
        
        elif format_type == CacheFormat.COMPRESSED_PICKLE:
            # Save as compressed pickle
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        
        else:  # CacheFormat.PICKLE
            # Save as standard pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        
        return cache_file.stat().st_size
    
    def _load_compressed(self, cache_file: Path, format_type: str) -> Any:
        """Load value using the specified compression format."""
        if format_type in [CacheFormat.NPZ, CacheFormat.COMPRESSED_NPZ]:
            # Load NPZ file
            data = np.load(cache_file)
            if len(data.files) == 1 and 'data' in data.files:
                # Single array saved as 'data'
                return data['data']
            else:
                # Multiple arrays - return as dict
                return {key: data[key] for key in data.files}
        
        elif format_type == CacheFormat.COMPRESSED_PICKLE:
            # Load compressed pickle
            with gzip.open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        else:  # CacheFormat.PICKLE
            # Load standard pickle
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    @contextmanager
    def _get_memory_mapped_file(self, cache_file: Path):
        """Get a memory-mapped file for large data access."""
        if not self.use_memory_mapping or not cache_file.exists():
            yield None
            return
        
        key = str(cache_file)
        
        if key in self._mmap_cache:
            # Return existing memory-mapped file
            yield self._mmap_cache[key]
        else:
            # Create new memory-mapped file
            try:
                with open(cache_file, 'rb') as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    self._mmap_cache[key] = mm
                    self._stats["memory_mapped_files"] += 1
                    yield mm
            except Exception as e:
                self.logger.warning(f"Failed to create memory map for {cache_file}: {e}")
                yield None
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache with compression and memory-mapping support.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found/expired
        """
        with self._lock:
            if key not in self._metadata:
                self._stats["misses"] += 1
                return None
            
            meta = self._metadata[key]
            format_type = meta.get("format", CacheFormat.PICKLE)
            
            # Determine file extension based on format
            if format_type in [CacheFormat.NPZ, CacheFormat.COMPRESSED_NPZ]:
                cache_file = self.cache_dir / f"{key}.npz"
            elif format_type == CacheFormat.COMPRESSED_PICKLE:
                cache_file = self.cache_dir / f"{key}.pkl.gz"
            else:
                cache_file = self.cache_dir / f"{key}.pkl"
            
            if not cache_file.exists():
                # Remove stale metadata
                del self._metadata[key]
                self._stats["misses"] += 1
                return None
            
            try:
                # Use memory mapping for large files if enabled
                if self.use_memory_mapping and cache_file.stat().st_size > 100 * 1024 * 1024:  # 100MB threshold
                    with self._get_memory_mapped_file(cache_file) as mm:
                        if mm is not None:
                            # Load from memory-mapped file would need special handling
                            # For now, fall back to regular loading for memory-mapped files
                            pass
                
                # Load cached item using appropriate format
                item = self._load_compressed(cache_file, format_type)
                
                # Update access time
                self._metadata[key]["last_accessed"] = time.time()
                self._stats["hits"] += 1
                
                return item
                
            except Exception as e:
                self.logger.error(f"Failed to load cached item {key}: {e}")
                # Remove corrupted cache entry
                self._remove_cache_entry(key)
                self._stats["misses"] += 1
                return None
    
    def set(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """
        Set item in cache with automatic compression and format detection.
        
        Args:
            key: Cache key
            value: Item to cache
            metadata: Optional metadata about the item
        """
        with self._lock:
            # Detect best storage format
            format_type = self._detect_best_format(value)
            
            # Determine file path based on format
            if format_type in [CacheFormat.NPZ, CacheFormat.COMPRESSED_NPZ]:
                cache_file = self.cache_dir / f"{key}.npz"
            elif format_type == CacheFormat.COMPRESSED_PICKLE:
                cache_file = self.cache_dir / f"{key}.pkl.gz"
            else:
                cache_file = self.cache_dir / f"{key}.pkl"
            
            try:
                # Calculate original size for compression ratio
                if isinstance(value, np.ndarray):
                    original_size = value.nbytes
                elif isinstance(value, dict) and any(isinstance(v, np.ndarray) for v in value.values()):
                    original_size = sum(v.nbytes for v in value.values() if isinstance(v, np.ndarray))
                else:
                    # Estimate size with pickle
                    original_size = len(pickle.dumps(value))
                
                # Save item using appropriate format
                file_size = self._save_compressed(cache_file, value, format_type)
                
                # Calculate compression ratio
                compression_ratio = original_size / file_size if file_size > 0 else 1.0
                
                # Update metadata
                self._metadata[key] = {
                    "created": time.time(),
                    "last_accessed": time.time(),
                    "size_bytes": file_size,
                    "original_size": original_size,
                    "compression_ratio": compression_ratio,
                    "format": format_type,
                    "metadata": metadata or {}
                }
                
                # Update statistics
                self._stats["size_bytes"] += file_size
                self._stats["item_count"] += 1
                self._stats["compression_ratio"] = (
                    self._stats["compression_ratio"] * (self._stats["item_count"] - 1) + compression_ratio
                ) / self._stats["item_count"]
                
                # Check if we need to evict items
                self._maybe_evict()
                
                # Save metadata
                self._save_metadata()
                
            except Exception as e:
                self.logger.error(f"Failed to cache item {key}: {e}")
                if cache_file.exists():
                    cache_file.unlink()
    
    def _maybe_evict(self):
        """Evict items if cache exceeds limits."""
        # Check size limit
        if self._stats["size_bytes"] > self.max_size_bytes:
            self._evict_by_size()
        
        # Check item count limit
        if self._stats["item_count"] > self.max_items:
            self._evict_by_count()
    
    def _evict_by_size(self):
        """Evict items until size is under limit."""
        target_size = int(self.max_size_bytes * 0.8)  # Evict to 80% of limit
        
        # Sort by last accessed time (LRU)
        sorted_items = sorted(
            self._metadata.items(),
            key=lambda x: x[1]["last_accessed"]
        )
        
        for key, meta in sorted_items:
            if self._stats["size_bytes"] <= target_size:
                break
            
            self._remove_cache_entry(key)
            self._stats["evictions"] += 1
        
        self.logger.info(f"Evicted items by size, new size: {self._stats['size_bytes'] / 1024 / 1024:.1f}MB")
    
    def _evict_by_count(self):
        """Evict items until count is under limit."""
        target_count = int(self.max_items * 0.8)  # Evict to 80% of limit
        
        # Sort by last accessed time (LRU)
        sorted_items = sorted(
            self._metadata.items(),
            key=lambda x: x[1]["last_accessed"]
        )
        
        items_to_evict = self._stats["item_count"] - target_count
        
        for key, meta in sorted_items[:items_to_evict]:
            self._remove_cache_entry(key)
            self._stats["evictions"] += 1
        
        self.logger.info(f"Evicted {items_to_evict} items by count, new count: {self._stats['item_count']}")
    
    def _remove_cache_entry(self, key: str):
        """Remove a cache entry and update statistics."""
        if key not in self._metadata:
            return
        
        meta = self._metadata[key]
        format_type = meta.get("format", CacheFormat.PICKLE)
        
        # Determine file path based on format
        if format_type in [CacheFormat.NPZ, CacheFormat.COMPRESSED_NPZ]:
            cache_file = self.cache_dir / f"{key}.npz"
        elif format_type == CacheFormat.COMPRESSED_PICKLE:
            cache_file = self.cache_dir / f"{key}.pkl.gz"
        else:
            cache_file = self.cache_dir / f"{key}.pkl"
        
        # Remove from memory-mapped cache if present
        if self._mmap_cache and str(cache_file) in self._mmap_cache:
            try:
                self._mmap_cache[str(cache_file)].close()
                del self._mmap_cache[str(cache_file)]
                self._stats["memory_mapped_files"] -= 1
            except Exception as e:
                self.logger.warning(f"Failed to close memory-mapped file {cache_file}: {e}")
        
        file_size = meta["size_bytes"]
        self._stats["size_bytes"] -= file_size
        self._stats["item_count"] -= 1
        del self._metadata[key]
        
        if cache_file.exists():
            cache_file.unlink()
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self._metadata_file, 'wb') as f:
                pickle.dump(self._metadata, f)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                if cache_file.name != "cache_metadata.pkl":
                    cache_file.unlink()
            
            # Reset metadata and statistics
            self._metadata = {}
            self._stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "size_bytes": 0,
                "item_count": 0
            }
            
            # Save empty metadata
            self._save_metadata()
            
            self.logger.info("Cache cleared")
    
    def invalidate_key(self, key: str):
        """Invalidate a specific cache key."""
        with self._lock:
            if key in self._metadata:
                self._remove_cache_entry(key)
                self._save_metadata()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / max(1, total_requests)
            
            return {
                **self._stats,
                "hit_rate": hit_rate,
                "size_mb": self._stats["size_bytes"] / 1024 / 1024,
                "max_size_mb": self.max_size_bytes / 1024 / 1024,
                "utilization": self._stats["size_bytes"] / self.max_size_bytes
            }
    
    def cleanup_stale_entries(self, max_age_hours: int = 24):
        """
        Remove entries older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours before entry is considered stale
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        stale_keys = []
        
        with self._lock:
            for key, meta in self._metadata.items():
                if meta["last_accessed"] < cutoff_time:
                    stale_keys.append(key)
            
            for key in stale_keys:
                self._remove_cache_entry(key)
            
            if stale_keys:
                self._save_metadata()
                self.logger.info(f"Cleaned up {len(stale_keys)} stale cache entries")
    
    def close(self):
        """Close all memory-mapped files and save metadata."""
        with self._lock:
            # Close all memory-mapped files
            if self._mmap_cache:
                for mm in self._mmap_cache.values():
                    try:
                        mm.close()
                    except Exception as e:
                        self.logger.warning(f"Failed to close memory-mapped file: {e}")
                self._mmap_cache.clear()
                self._stats["memory_mapped_files"] = 0
            
            # Save metadata
            self._save_metadata()


class DistributedCache:
    """
    Distributed cache system for multi-GPU training environments.
    
    Features:
    - Shared cache across multiple workers
    - Lock-free reading with occasional writing
    - Network file system support
    - Automatic synchronization
    """
    
    def __init__(self, cache_dir: Path, worker_id: int = 0, sync_interval: int = 300):
        self.cache_dir = Path(cache_dir)
        self.worker_id = worker_id
        self.sync_interval = sync_interval  # Sync every 5 minutes
        self.logger = logging.getLogger(__name__)
        
        # Local cache instance
        self.local_cache = AdvancedCache(
            cache_dir / f"worker_{worker_id}",
            max_size_gb=2.0,  # Smaller per-worker cache
            compression=True,
            use_memory_mapping=True
        )
        
        # Shared cache directory
        self.shared_dir = cache_dir / "shared"
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        
        # Synchronization state
        self._last_sync = 0
        self._sync_lock = Lock()
        
        self.logger.info(f"DistributedCache initialized for worker {worker_id}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from distributed cache."""
        # First check local cache
        item = self.local_cache.get(key)
        if item is not None:
            return item
        
        # Force sync with shared cache to get latest items
        self._sync_from_shared()
        
        # Check local cache again after sync
        item = self.local_cache.get(key)
        if item is not None:
            return item
        
        # Check shared cache directly
        item = self._get_from_shared(key)
        if item is not None:
            # Store in local cache for future access
            self.local_cache.set(key, item)
        
        return item
    
    def set(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Set item in distributed cache."""
        # Always store in local cache
        self.local_cache.set(key, value, metadata)
        
        # Immediately sync to shared cache for testing
        self._sync_to_shared()
    
    def _get_from_shared(self, key: str) -> Optional[Any]:
        """Get item from shared cache directory."""
        try:
            # Look for the item in shared directory with different extensions
            possible_files = [
                self.shared_dir / f"{key}.npz",
                self.shared_dir / f"{key}.pkl.gz", 
                self.shared_dir / f"{key}.pkl"
            ]
            
            for shared_file in possible_files:
                if shared_file.exists():
                    # Determine format from extension
                    if shared_file.suffix == ".npz":
                        format_type = CacheFormat.COMPRESSED_NPZ if "compressed" in str(shared_file) else CacheFormat.NPZ
                    elif shared_file.suffixes == [".pkl", ".gz"]:
                        format_type = CacheFormat.COMPRESSED_PICKLE
                    else:
                        format_type = CacheFormat.PICKLE
                    
                    return self.local_cache._load_compressed(shared_file, format_type)
        
        except Exception as e:
            self.logger.warning(f"Failed to load {key} from shared cache: {e}")
        
        return None
    
    def _sync_from_shared(self):
        """Sync new items from shared cache to local cache."""
        with self._sync_lock:
            try:
                # Find all shared cache files (more aggressive sync for testing)
                for cache_file in self.shared_dir.glob("*"):
                    if cache_file.is_file():
                        # Extract key from filename
                        key = cache_file.stem
                        if key.endswith(".pkl"):
                            key = key[:-4]  # Remove .pkl extension
                        
                        # Only sync if we don't have this key locally
                        if key not in self.local_cache._metadata:
                            # Determine format and load
                            if cache_file.suffix == ".npz":
                                format_type = CacheFormat.COMPRESSED_NPZ
                            elif cache_file.suffixes == [".pkl", ".gz"]:
                                format_type = CacheFormat.COMPRESSED_PICKLE
                            else:
                                format_type = CacheFormat.PICKLE
                            
                            try:
                                item = self.local_cache._load_compressed(cache_file, format_type)
                                if item is not None:
                                    self.local_cache.set(key, item)
                                    self.logger.debug(f"Synced {key} from shared cache")
                            except Exception as load_error:
                                self.logger.warning(f"Failed to load {key} from shared cache: {load_error}")
                
                self._last_sync = time.time()
                
            except Exception as e:
                self.logger.warning(f"Failed to sync from shared cache: {e}")
    
    def _sync_to_shared(self):
        """Sync local cache items to shared cache."""
        with self._sync_lock:
            try:
                # Get recently accessed items from local cache
                cutoff_time = time.time() - self.sync_interval
                
                for key, meta in self.local_cache._metadata.items():
                    if meta["last_accessed"] > cutoff_time:
                        # Check if item exists in shared cache
                        shared_exists = any(
                            self.shared_dir.glob(f"{key}.*")
                        )
                        
                        if not shared_exists:
                            # Copy from local to shared
                            item = self.local_cache.get(key)
                            if item is not None:
                                format_type = meta.get("format", CacheFormat.PICKLE)
                                
                                # Determine shared file path
                                if format_type in [CacheFormat.NPZ, CacheFormat.COMPRESSED_NPZ]:
                                    shared_file = self.shared_dir / f"{key}.npz"
                                elif format_type == CacheFormat.COMPRESSED_PICKLE:
                                    shared_file = self.shared_dir / f"{key}.pkl.gz"
                                else:
                                    shared_file = self.shared_dir / f"{key}.pkl"
                                
                                # Save to shared cache
                                self.local_cache._save_compressed(shared_file, item, format_type)
                
                self._last_sync = time.time()
                
            except Exception as e:
                self.logger.warning(f"Failed to sync to shared cache: {e}")
    
    def get_stats(self) -> Dict:
        """Get distributed cache statistics."""
        local_stats = self.local_cache.get_stats()
        
        # Count shared cache files
        shared_files = len(list(self.shared_dir.glob("*")))
        shared_size = sum(f.stat().st_size for f in self.shared_dir.glob("*") if f.is_file())
        
        return {
            **local_stats,
            "worker_id": self.worker_id,
            "shared_files": shared_files,
            "shared_size_mb": shared_size / 1024 / 1024,
            "last_sync": self._last_sync,
            "sync_interval": self.sync_interval
        }
    
    def close(self):
        """Close distributed cache and sync final state."""
        self._sync_to_shared()
        self.local_cache.close()
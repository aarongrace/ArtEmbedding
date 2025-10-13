import threading
from typing import Dict, List, Optional
import torch

from embed_model import forward_images
from model_services import get_seen_list, load_PIL_image, get_random_image_id, USER_NAME, add_to_seen_list

class EmbeddingCache:
    """
    Manages a cache of pre-computed embeddings with automatic batch refilling.
    When cache drops to LOW_THRESHOLD, automatically fetches BATCH_SIZE more.
    """
    
    BATCH_SIZE = 8
    LOW_THRESHOLD = 4
    
    def __init__(self):
        """Initialize embedding cache with automatic prefetching."""
        # Cache storage: {image_id: embedding_vector}
        self.cache: Dict[str, List[float]] = {}
        self.cache_lock = threading.Lock()

        self._prefill_cache(num_items=1)
        
        # Background processing
        self.prefetch_thread = None
        self.should_stop = threading.Event()
        
        # Initialize and start background thread
        self._start_prefetch_thread()
        
    def _start_prefetch_thread(self):
        """Start background thread for prefetching embeddings."""
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        print("Embedding cache prefetch thread started")
        
    def _prefetch_worker(self):
        """Background worker that maintains cache at desired level."""
        while not self.should_stop.is_set():
            try:
                # Check if we need to refill
                with self.cache_lock:
                    cache_size = len(self.cache)
                    
                if cache_size <= self.LOW_THRESHOLD:
                    print(f"Cache at {cache_size}, refilling with {self.BATCH_SIZE} embeddings...")
                    self._batch_compute_embeddings(self.BATCH_SIZE)
                else:
                    # Sleep a bit before checking again
                    self.should_stop.wait(timeout=5.0)
                    
            except Exception as e:
                print(f"Error in prefetch worker: {e}")
                import traceback
                traceback.print_exc()
                self.should_stop.wait(timeout=5.0)
                
    def _batch_compute_embeddings(self, num: int = BATCH_SIZE):
        """
        Compute embeddings for a batch of images.
        
        Args:
            num: Number of embeddings to compute (default: BATCH_SIZE)
        """
        # Get random image IDs using the existing function, excluding already cached ones
        try:
            with self.cache_lock:
                cached_ids = list(self.cache.keys())
            
            # Get random IDs that aren't already cached
            batch_ids = get_random_image_id(num, exclude=cached_ids)
            
            # Handle single ID case
            if isinstance(batch_ids, str):
                batch_ids = [batch_ids]
                
        except ValueError as e:
            print(f"No more images available to cache: {e}")
            return
        except Exception as e:
            print(f"Error getting random image IDs: {e}")
            return
        
        # Load images
        images = []
        valid_ids = []
        for image_id in batch_ids:
            try:
                img = load_PIL_image(image_id)
                images.append(img)
                valid_ids.append(image_id)
            except Exception as e:
                print(f"Error loading image {image_id}: {e}")
        
        if not images:
            return
        
        # Batch forward pass
        try:
            embeddings = forward_images(images)
            print(f"Forward pass completed on {len(images)} images")
            with self.cache_lock:
                for img_id, emb in zip(valid_ids, embeddings):
                        self.cache[img_id] = emb
            
        except Exception as e:
            print(f"Error during forward pass: {e}")
            return
            
        except Exception as e:
            print(f"Error computing batch embeddings: {e}")
            import traceback
            traceback.print_exc()

    def _prefill_cache(self, num_items: int = 1):
        """
        Manually trigger cache prefilling.
        Useful for warming up the cache on startup.
        
        Args:
            num_items: Number of items to prefill (default: 1)
        """
        print(f"Prefilling cache with {num_items} item(s)...")
        self._batch_compute_embeddings(num_items)
        print(f"Cache prefill complete. Stats: {self.get_cache_stats()}")
    
    def get_embedding(self) -> Optional[tuple[str, List[float]]]:
        """
        Get next embedding from cache.
        If cache is empty, compute more embeddings and return one.
        
        Returns:
            Tuple of (image_id, embedding_vector) or None if no images available
        """
        with self.cache_lock:
            # Try cache first
            if self.cache:
                image_id, embedding = self.cache.popitem()
                add_to_seen_list(image_id)
                print(f"Cache hit. Returning {image_id}. Remaining: {len(self.cache)}")
                return (image_id, embedding)
        
        # Cache is empty - compute more
        print(f"Cache empty, computing {self.BATCH_SIZE} embeddings...")
        self._batch_compute_embeddings(self.BATCH_SIZE)
        
        # Now try again
        with self.cache_lock:
            if self.cache:
                image_id, embedding = self.cache.popitem()
                add_to_seen_list(image_id)
                print(f"Returning {image_id}. Remaining: {len(self.cache)}")
                return (image_id, embedding)
        
        print("No images available")
        return None
    
    def get_cache_stats(self) -> Dict:
        """Get current cache statistics."""
        with self.cache_lock:
            return {
                "cached": len(self.cache),
                "seen": len(get_seen_list(USER_NAME)),
                "threshold": self.LOW_THRESHOLD,
                "batch_size": self.BATCH_SIZE,
                "user_id": USER_NAME
            }
    
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        with self.cache_lock:
            self.cache.clear()
        print("Cache cleared")
    
    def shutdown(self):
        """Gracefully shutdown the cache."""
        print("Shutting down embedding cache...")
        self.should_stop.set()
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=5.0)
        print("Cache shutdown complete")


# Singleton instance
_cache_instance: Optional[EmbeddingCache] = None
_cache_lock = threading.Lock()

def get_cache() -> EmbeddingCache:
    """Get or create the singleton cache instance."""
    global _cache_instance
    
    with _cache_lock:
        if _cache_instance is None:
            _cache_instance = EmbeddingCache()
        return _cache_instance

def reset_cache():
    """Reset the singleton cache instance (useful for testing or user switching)."""
    global _cache_instance
    
    with _cache_lock:
        if _cache_instance is not None:
            _cache_instance.shutdown()
            _cache_instance = None
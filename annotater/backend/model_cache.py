import threading
from typing import Dict, List, Optional
from collections import deque

from embed_model import forward_images, backward_single_image
from model_services import get_seen_list, load_PIL_image, get_random_image_id, USER_NAME, add_to_seen_list


one_by_one_mode = False
class EmbeddingCache:
    """
    Manages a cache of pre-computed embeddings with automatic batch refilling.
    When cache drops to LOW_THRESHOLD, automatically fetches BATCH_SIZE more.
    """
    
    BATCH_SIZE = 8
    LOW_THRESHOLD = 4
    if one_by_one_mode:
        BATCH_SIZE = 1
        LOW_THRESHOLD = 1

    
    def __init__(self):
        """Initialize embedding cache with automatic prefetching."""
        # Cache storage: {image_id: embedding_vector}
        self.cache: deque[tuple[str, List[float]]] = deque()
        self.cache_lock = threading.Lock()

        self.backprop_queue: deque[tuple[str, List[float]]] = deque()
        self.backprop_queue_lock = threading.Lock()
        
        # Background processing
        self.model_thread = None
        self.should_stop = threading.Event()


        self.fetch_large_images = True
        self.fetch_test_images = False
        
        # Initialize and start background thread
        self._start_thread()
        
    def _start_thread(self):
        """Start background thread for prefetching embeddings."""
        self.model_thread = threading.Thread(target=self._model_worker, daemon=True)
        self.model_thread.start()
        print("Embedding cache prefetch thread started")
        
    def _model_worker(self):
        """Background worker that maintains cache at desired level."""
        while not self.should_stop.is_set():
            try:
                # Check if we need to refill, higher priority than backprop
                with self.cache_lock:
                    cache_size = len(self.cache)
                if cache_size == 0:
                    print("Cache empty, refilling immediately...")
                    self._batch_compute_embeddings(2)
                elif cache_size <= self.LOW_THRESHOLD:
                    print(f"Cache at {cache_size}, refilling with {self.BATCH_SIZE} embeddings...")
                    self._batch_compute_embeddings(self.BATCH_SIZE)
                
                item = None
                with self.backprop_queue_lock:
                    if self.backprop_queue:
                        print(f"Processing backprop queue with {len(self.backprop_queue)} items...")
                        item = self.backprop_queue.popleft()
                if item:
                    self._send_backprop_image(item)

                self.should_stop.wait(timeout=1.0)
                    
            except Exception as e:
                print(f"Error in prefetch worker: {e}")
                import traceback
                traceback.print_exc()
                self.should_stop.wait(timeout=5.0)
                
    def _send_backprop_image(self, item):
        image_id, vector = item
        image = load_PIL_image(image_id)
        backward_single_image(image, vector, image_id=image_id)
        print(f"Back prop completed for {image_id}")

    
    def _batch_compute_embeddings(self, num: int = BATCH_SIZE):
        """
        Compute embeddings for a batch of images.
        
        Args:
            num: Number of embeddings to compute (default: BATCH_SIZE)
        """
        # Get random image IDs using the existing function, excluding already cached ones
        try:
            with self.cache_lock:
                cached_ids = [img_id for img_id, _ in self.cache]
            
            # Get random IDs that aren't already cached
            batch_ids = get_random_image_id(num, exclude=cached_ids, 
                                           exclude_test=not self.fetch_test_images, 
                                           exclude_train=self.fetch_test_images,
                                           prioritize_large_files=self.fetch_large_images)
            
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
            with self.cache_lock:
                for img_id, emb in zip(valid_ids, embeddings):
                        self.cache.append((img_id, emb))
            
        except Exception as e:
            print(f"Error during forward pass: {e}")
            return
            
    
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
                image_id, embedding = self.cache.popleft()
                add_to_seen_list(image_id)
                formatted_embedding = [float(f"{v:.3f}") for v in embedding]
                print(f"Cache hit. Returning {image_id} with {formatted_embedding}. Remaining: {len(self.cache)}")
                return (image_id, embedding)
        
        # Cache is empty - compute more
        print(f"Cache empty, computing {self.BATCH_SIZE} embeddings...")
        self._batch_compute_embeddings(self.BATCH_SIZE)
        
        # Now try again
        with self.cache_lock:
            if self.cache:
                image_id, embedding = self.cache.popleft()
                add_to_seen_list(image_id)
                formatted_embedding = [float(f"{v:.3f}") for v in embedding]
                print(f"Returning {image_id} with {formatted_embedding}. Remaining: {len(self.cache)}")
                return (image_id, embedding)
        
        print("No images available")
        return None
    
    def add_to_backprop_queue(self, image_id: str, vector: List[float]):
        """Add an image ID to the backpropagation queue."""
        with self.backprop_queue_lock:
            self.backprop_queue.append((image_id, vector))
            print(f"Added {image_id} to backprop queue with vector: {vector}")

    def set_fetch_large_images(self, value: bool):
        """Set whether to fetch larger images."""
        self.fetch_large_images = value
        print(f"Set fetch_large_images to {value}")
        self.cache.clear()  # Clear cache to respect new setting
    
    def set_fetch_test_images(self, value: bool):
        """Set whether to fetch test images."""
        self.fetch_test_images = value
        print(f"Set fetch_test_images to {value}")
        self.cache.clear()  # Clear cache to respect new setting

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
        if self.model_thread:
            self.model_thread.join(timeout=5.0)
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
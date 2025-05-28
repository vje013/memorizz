

class SemanticCache:
    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        # When a key is placed into the cache, check if the key exists in the memory provider
        if self.memory_provider.get(key) is not None:
            # If the key exists in the memory provider, update the value in the cache
            self.cache[key] = value
        else:
            # If the key does not exist in the memory provider, add the key to the memory provider
            self.memory_provider.add(key, value)
        self.cache[key] = value
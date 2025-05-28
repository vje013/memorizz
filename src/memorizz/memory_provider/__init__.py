from .base import MemoryProvider
from .mongodb import MongoDBProvider
from .memory_type import MemoryType

# Conditional import for PostgreSQL to avoid dependency issues
try:
    from .postgresql import PostgreSQLProvider, PostgreSQLConfig
    _postgresql_available = True
except ImportError:
    _postgresql_available = False
    PostgreSQLProvider = None
    PostgreSQLConfig = None

__all__ = [
    'MemoryProvider',
    'MongoDBProvider',
    'MemoryType'
]

# Only add PostgreSQL classes to __all__ if they're available
if _postgresql_available:
    __all__.extend(['PostgreSQLProvider', 'PostgreSQLConfig'])
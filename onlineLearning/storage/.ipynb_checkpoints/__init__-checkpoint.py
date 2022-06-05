"""Storage classes
"""

from onlineLearning.storage.model import MemoryModelStorage,RedisModelStorage
from onlineLearning.storage.history import (MemoryHistoryStorage,
                                            History,
                                            RedisHistoryStorage,
                                            DiskCacheHistoryStorage)
from onlineLearning.storage.action import MemoryActionStorage, Action
from onlineLearning.storage.recommendation import Recommendation
from onlineLearning.storage.yaml_dataset import YAMLDataSet
from onlineLearning.storage.json_dataset import JSONDataSet
from onlineLearning.storage.dbengine import dbEngine

__all__ = [
    'MemoryModelStorage',
    'RedisModelStorage',
    'MemoryHistoryStorage',
    'History',
    'RedisHistoryStorage',
    'DiskCacheHistoryStorage',
    'MemoryActionStorage',
    'Action',
    'Recommendation',
    'YAMLDataSet',
    'JSONDataSet',
    'dbEngine'
]
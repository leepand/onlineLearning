"""This module contains functionality for defining data and workflows within onlineLearning.
This module is primarily intended for internal use within onlineLearning. However, it is documented
and made public for advanced use cases where existing Environment and Experiment creation
functionality is not sufficient. That being said, a good understanding of the patterns
in onlineLearning.pipes can help one understand how to best take advantage what onlineLearning has to offer.
"""

from onlineLearning.pipes.filters import Flatten

__all__ = [
    "Flatten"
]
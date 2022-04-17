import logging as __logging

_logger = __logging.getLogger("metrix")
_logger.addHandler(__logging.StreamHandler())
_logger.setLevel(__logging.INFO)

"""Parser registry. Importing this package registers all built-in parsers."""

from chatmem.parsers import instagram  # noqa: F401  (registers InstagramParser)
from chatmem.parsers import whatsapp  # noqa: F401  (registers WhatsAppParser)
from chatmem.parsers.base import PARSERS, Parser, select_parser

__all__ = ["PARSERS", "Parser", "select_parser"]

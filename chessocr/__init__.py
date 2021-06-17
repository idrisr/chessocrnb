__version__ = "0.0.3"
from fastai.data.all import URLs
from pathlib import Path
from pkgutil import get_data

__all__ = ['URLs', 'boards_url', 'pieces_url']

URLs.chess_small = "https://chess-screenshots.s3.amazonaws.com/chess-small.tgz"
URLs.website = "https://chess-screenshots.s3.amazonaws.com/websites.tgz"

boards_url = Path(__file__).parent/"img/boards"
pieces_url = Path(__file__).parent/"img/pieces"

import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

INDEX_INPUT_PATH = Path("SEC/data/filing_index.parquet")
SECTIONS_OUTPUT_PATH = Path("SEC/data/sections.parquet")

REGEX_ITEM_1A_START = re.compile(
    r"(?:item\s+1a[\.\:\-\s]+risk\s+factors|item\s+1a\b)", re.IGNORECASE
)#item 1a. risk factors
#item 1a: risk factors
#item 1a - risk factors
#item 1a risk factors

REGEX_ITEM_1A_END = re.compile(
    r"(?:item\s+1b[\.\:\-\s]+unresolved\s+staff\s+comments|item\s+1b\b|item\s+2[\.\:\-\s]+properties)", re.IGNORECASE
)
#ITEM 1B. Unresolved Staff Comments
#Item 1B.
#ITEM 2. Properties

REGEX_ITEM_7_START = re.compile(
    r"(?:item\s+7[\.\:\-\s]+management[\’\'\`s]*\s+discussion|item\s+7\b(?!\s*a))", re.IGNORECASE
)
#ITEM 7. Management's Discussion
#Item 7: Management’s Discussion
#Item 7 - Management Discussion

REGEX_ITEM_7_END = re.compile(
    r"(?:item\s+7a[\.\:\-\s]+quantitative|item\s+7a\b|item\s+8[\.\:\-\s]+financial\s+statements)", re.IGNORECASE
)
#ITEM 7A. Quantitative
#Item 7A.
#ITEM 8. Financial Statements

def clean_html_to_text(html_content: bytes) -> str:
    soup = BeautifulSoup(html_content, "lxml")

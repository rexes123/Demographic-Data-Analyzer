import numpy as np
# Syetem-specific parameters and functions
import sys
import pandas as pd
import matplotlib.pyplot as pltf
from decimal import Decimal
import sqlite3

# Parsing HTML tables from the web
html_url = "https://www.basketball-reference.com/leagues/NBA_2022_per_game.html"
# print(html_url)
nba_tables = pd.read_html(html_url)
# print(nba_tables)
len(nba_tables)
print(nba_tables)
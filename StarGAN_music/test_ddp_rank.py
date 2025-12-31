import os
import sys
print(f"LOCAL_RANK env: {os.environ.get('LOCAL_RANK', 'NOT SET')}")
print(f"RANK env: {os.environ.get('RANK', 'NOT SET')}")
print(f"WORLD_SIZE env: {os.environ.get('WORLD_SIZE', 'NOT SET')}")
print(f"sys.argv: {sys.argv}")

# .github/update_version.py
import sys
import re

version = sys.argv[1]
with open('setup.py', 'r+') as f:
    content = f.read()
    content_new = re.sub(r'version=".*",', f'version="{version}",', content)
    f.seek(0)
    f.write(content_new)
    f.truncate()
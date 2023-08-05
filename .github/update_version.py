# .github/update_version.py
import re


def increment_version(version):
    major, minor, patch = map(int, version.split("."))
    patch += 1
    return f"{major}.{minor}.{patch}"


with open("setup.py", "r+") as f:
    content = f.read()
    version_match = re.search(r'version="(.*)",', content)
    if version_match:
        old_version = version_match.group(1)
        new_version = increment_version(old_version)
        content_new = re.sub(old_version, new_version, content)
        f.seek(0)
        f.write(content_new)
        f.truncate()

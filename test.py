from pathlib import Path
path = Path(__file__).resolve().parent
print(path)
target = path/"test.txt"

a = target.read_text(encoding="utf-8")
print(a)
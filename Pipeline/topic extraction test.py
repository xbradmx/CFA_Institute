"""Quick debug script to see what BeautifulSoup outputs around section headers."""

from pathlib import Path
from bs4 import BeautifulSoup

f = Path("data/Filings by company/ACLS/10-K")
html = sorted(f.iterdir())[0]
print(f"File: {html.name}\n")

with open(html, "r", encoding="utf-8") as fh:
    soup = BeautifulSoup(fh.read(), "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    text = soup.get_text(separator="\n")

lines = text.split("\n")

print("=== Around Risk Factors (lines 1260-1275) ===")
for i in range(1260, min(1275, len(lines))):
    print(f"  {i}: {repr(lines[i][:120])}")

print("\n=== Around 1B / Unresolved (lines 1435-1445) ===")
for i in range(1435, min(1445, len(lines))):
    print(f"  {i}: {repr(lines[i][:120])}")

print("\n=== Around MD&A (lines 1760-1800) ===")
for i in range(1760, min(1800, len(lines))):
    print(f"  {i}: {repr(lines[i][:120])}")

print("\n=== Around Quantitative (lines 3098-3110) ===")
for i in range(3098, min(3110, len(lines))):
    print(f"  {i}: {repr(lines[i][:120])}")
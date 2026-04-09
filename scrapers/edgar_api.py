import requests
import json
import time
import csv
import os

from dotenv import load_dotenv

load_dotenv()
EDGAR_USER_AGENT = os.environ.get("EDGAR_USER_AGENT")

headers = {"User-Agent": EDGAR_USER_AGENT}

print("Fetching company list...")
tickers = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers).json()
print(f"Total companies: {len(tickers)}")

companies = []
errors = 0

print("Querying for SIC 3400-3599 (this will take ~19 minutes)...\n")

for i, (key, company) in enumerate(tickers.items()):
    cik = str(company["cik_str"]).zfill(10)
    
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        
        sic = int(data.get("sic", 0))
        if 3400 <= sic <= 3599:
            exchanges = data.get("exchanges", [])
            companies.append({
                "name": data.get("name"),
                "ticker": company.get("ticker"),
                "sic": sic,
                "sicDescription": data.get("sicDescription"),
                "exchanges": [e for e in exchanges if e is not None],
                "category": data.get("category", ""),
                "website": data.get("website", "")
            })
            print(f"  FOUND: {company.get('ticker'):8s} | SIC {sic} | {data.get('sicDescription', '')}")
    except:
        errors += 1
    
    time.sleep(0.11)
    
    if (i + 1) % 500 == 0:
        print(f"  ...processed {i+1}/{len(tickers)} | found {len(companies)} so far | {errors} errors")

# Summary
all_count = len(companies)
major = [c for c in companies if any(e in ["NYSE", "Nasdaq", "NASDAQ"] for e in c["exchanges"])]
major_no_computers = [c for c in major if not (3570 <= c["sic"] <= 3579)]
all_no_computers = [c for c in companies if not (3570 <= c["sic"] <= 3579)]

print(f"\n{'='*60}")
print(f"SIC 3400-3599 total: {all_count}")
print(f"NYSE/NASDAQ only: {len(major)}")
print(f"Excluding computers (3570-3579):")
print(f"  Total: {len(all_no_computers)}")
print(f"  NYSE/NASDAQ: {len(major_no_computers)}")
print(f"{'='*60}")

# Breakdown by sub-group
groups = {
    "3400-3409 Fabricated Metal Products": (3400, 3409),
    "3410-3419 Metal Cans & Shipping Containers": (3410, 3419),
    "3420-3429 Cutlery, Hardware & Tools": (3420, 3429),
    "3430-3439 Heating Equipment & Plumbing": (3430, 3439),
    "3440-3449 Fabricated Structural Metal": (3440, 3449),
    "3450-3459 Screw Products, Bolts, Rivets": (3450, 3459),
    "3460-3469 Metal Forgings & Stampings": (3460, 3469),
    "3470-3479 Coating & Engraving": (3470, 3479),
    "3480-3489 Ordnance & Accessories": (3480, 3489),
    "3490-3499 Misc Fabricated Metal Products": (3490, 3499),
    "3510-3519 Engines & Turbines": (3510, 3519),
    "3520-3529 Farm & Garden Machinery": (3520, 3529),
    "3530-3539 Construction & Mining Equipment": (3530, 3539),
    "3540-3549 Metalworking Machinery": (3540, 3549),
    "3550-3559 Special Industry Machinery": (3550, 3559),
    "3560-3569 General Industrial Machinery": (3560, 3569),
    "3570-3579 Computer & Office Equipment (EXCLUDE)": (3570, 3579),
    "3580-3589 Refrigeration & Service Machinery": (3580, 3589),
    "3590-3599 Misc Industrial Machinery": (3590, 3599),
}

print(f"\nBreakdown (all / major exchange):")
print(f"{'-'*60}")
for name, (lo, hi) in groups.items():
    all_ct = len([c for c in companies if lo <= c["sic"] <= hi])
    maj_ct = len([c for c in major if lo <= c["sic"] <= hi])
    if all_ct > 0:
        print(f"  {name}: {all_ct} / {maj_ct}")

# Save to CSV
print(f"\nSaving to capital_goods_companies.csv...")
with open("capital_goods_companies.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ticker", "name", "sic", "sicDescription", "exchanges", "category", "website"])
    for c in sorted(companies, key=lambda x: x["sic"]):
        writer.writerow([
            c["ticker"], c["name"], c["sic"], c["sicDescription"],
            "|".join(c["exchanges"]), c["category"], c["website"]
        ])

print("\nDone!")
print(f"\nRECOMMENDATION: Use the NYSE/NASDAQ companies excluding 3570-3579 computers.")
print(f"This gives you {len(major_no_computers)} companies in US Capital Goods.")
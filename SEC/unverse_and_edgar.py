import os
import time
from pathlib import Path
from typing import List, Dict
import requests
import pandas as pd
from tqdm import tqdm

USER_AGENT = "bh2991@columbia.edu"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

RAW_HTML_DIR = Path("SEC/data/raw_filings")
INDEX_OUTPUT_PATH = Path("SEC/data/raw_filings/filing_index.parquet")
RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

def get_sample_ciks(limit: int = 5) -> List[str]:
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    ciks = [str(item["cik_str"]).zfill(10) for item in list(data.values())[:limit]]
    return ciks

def fetch_company_10k_metadata(cik: str, start_year: int = 2015, end_year: int = 2024) -> List[Dict]:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return []
    data = resp.json()
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return []
    records = []
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    acceptance_dts = recent.get("acceptanceDateTime", [])
    accession_nums = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    report_dates = recent.get("reportDate", [])

    for i, form in enumerate(forms):
        if form != "10-K":
            continue

        filing_date = filing_dates[i]
        filing_year = int(filing_date[:4])


        if not (start_year <= filing_year <= end_year):
            continue
        acc_num_raw = accession_nums[i]
        acc_num_no_hyphen = acc_num_raw.replace("-", "")
        primary_doc = primary_docs[i]
        print(primary_doc)
        cik_int = str(int(cik))
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_num_no_hyphen}/{primary_doc}"
        records.append({
            "cik": cik,
            "form": form,
            "filing_date": filing_date,
            "report_date": report_dates[i],
            "acceptance_datetime": acceptance_dts[i],  # Point-in-Time  (eg 2024-02-01T16:30:12.000Z)
            "accession_number": acc_num_raw,
            "doc_url": doc_url,
            "local_path": str(RAW_HTML_DIR / f"{cik}_{filing_date}_{acc_num_no_hyphen}.htm")
        })
    return records
def download_filings(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    根据元数据下载 10-K 原始 HTML 文件，严格遵守 SEC 10 requests/sec 限流。
    """
    valid_records = []
    for row in tqdm(metadata_df.to_dict(orient="records"), desc="Downloading 10-K Filings"):
        local_path = Path(row["local_path"])

        # cached
        if local_path.exists() and local_path.stat().st_size > 1000:
            row["download_status"] = "cached"
            valid_records.append(row)
            continue

        try:
            time.sleep(0.12)  # 控制请求频率 <= 8.3 次/秒，留出安全边际
            resp = requests.get(row["doc_url"], headers=HEADERS, timeout=30)
            resp.raise_for_status()

            with open(local_path, "wb") as f:
                f.write(resp.content)

            row["download_status"] = "success"
            valid_records.append(row)
        except Exception as e:
            print(f"\n[Error] Failed to download {row['cik']} ({row['filing_date']}): {e}")
            row["download_status"] = "failed"

    return pd.DataFrame(valid_records)





if __name__ == "__main__":
    # 1. 获取目标公司 CIK 列表（test for 5 companies Pipeline）
    target_ciks = get_sample_ciks(limit=5)
    print(f"Target CIKs: {target_ciks}")

    # 2. 拉取 Point-in-Time 索引元数据
    all_meta = []
    for cik in tqdm(target_ciks, desc="Fetching EDGAR Metadata"):
        time.sleep(0.12)
        all_meta.extend(fetch_company_10k_metadata(cik, start_year=2015, end_year=2024))

    meta_df = pd.DataFrame(all_meta)

    # 3. download the article

    if not meta_df.empty:
        final_index_df = download_filings(meta_df)
        final_index_df.to_parquet(INDEX_OUTPUT_PATH, index=False)
        print(f"Saved Point-in-Time index to {INDEX_OUTPUT_PATH}")

import os
import requests
import json
import time
import threading
import random
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from datetime import datetime
import sys
from dotenv import load_dotenv

load_dotenv()

class InferenceProgressTracker:
    """Thread-safe progress tracker for GPL inference"""

    def __init__(self, total_items: int, api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.total = total_items
        self.completed = 0
        self.failed = 0
        self.retried = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.last_update = 0
        self._api_url = api_url or os.getenv("GPT_API_URL")
        self._api_key = api_key or os.getenv("GPT_API_KEY")

    def update(self, status: str, gpl_id: str, details: str = "", retry_count: int = 0):
        with self.lock:
            if status == "success":
                self.completed += 1
            elif status == "failed":
                self.failed += 1
            elif status == "retry":
                self.retried += 1

            # Update progress every item or every 5 seconds
            now = time.time()
            processed = self.completed + self.failed

            if processed % 1 == 0 or now - self.last_update > 5:
                self.print_progress(gpl_id, details, retry_count)
                self.last_update = now

    def print_progress(self, current_gpl: str = "", details: str = "", retry_count: int = 0):
        elapsed = time.time() - self.start_time
        processed = self.completed + self.failed
        remaining = self.total - processed

        if processed > 0 and elapsed > 0:
            rate = processed / elapsed
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
        else:
            rate = 0
            eta_minutes = 0

        percentage = (processed / self.total) * 100 if self.total > 0 else 0

        # Progress bar
        bar_length = 30
        filled = int(bar_length * processed / self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)

        retry_info = f" (retry {retry_count})" if retry_count > 0 else ""

        print(f"\r🤖 [{bar}] {percentage:5.1f}% | "
              f"✅ {self.completed} | ❌ {self.failed} | 🔄 {self.retried} | "
              f"🚀 {rate:.1f}/s | ⏱️ ETA: {eta_minutes:.1f}m | 🎯 {current_gpl}{retry_info}",
              end="", flush=True)

        if details:
            print(f" | {details}", end="", flush=True)

    def final_report(self):
        elapsed = time.time() - self.start_time
        # Prevent division by zero
        elapsed = max(elapsed, 0.001)  # Minimum 1ms to avoid division by zero
        
        print(f"\n\n🎉 INFERENCE COMPLETE!")
        print(f"{'='*60}")
        print(f"⏰ Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"✅ Successful: {self.completed}")
        print(f"❌ Failed: {self.failed}")
        print(f"🔄 Total retries: {self.retried}")
        print(f"📈 Success rate: {(self.completed/(self.completed+self.failed)*100):.1f}%" if (self.completed+self.failed) > 0 else "N/A")
        print(f"🚀 Average rate: {(self.completed+self.failed)/elapsed:.1f} GPL/second")
        print(f"{'='*60}")


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff delay with jitter"""
    delay = min(base_delay * (2 ** attempt), max_delay)
    # Add jitter (±25%)
    jitter = delay * 0.25 * (2 * random.random() - 1)
    return max(0, delay + jitter)

def build_prompt(gpl_record: dict) -> str:
    """Build prompt for GPL inference with proper string escaping"""
    gpl_id = gpl_record.get("gpl_id", "")
    metadata = gpl_record.get("metadata", {})
    table = gpl_record.get("table", {})
    columns = table.get("columns", [])
    sample_rows = table.get("sample_rows", [])
    column_descriptions = metadata.get("column_descriptions", {})
    
    # Safely build strings with proper escaping
    if column_descriptions:
        column_desc_parts = []
        for k, v in column_descriptions.items():
            # Escape any problematic characters in the description
            safe_key = str(k).replace("{", "{{").replace("}", "}}")
            safe_value = str(v).replace("{", "{{").replace("}", "}}")
            column_desc_parts.append(f"**{safe_key}**: {safe_value}")
        column_desc_str = "\n".join(column_desc_parts)
    else:
        column_desc_str = "No column descriptions available."
    
    # Build metadata string with escaping
    metadata_parts = []
    for k, v in metadata.items():
        if k != "column_descriptions":
            safe_key = str(k).replace("{", "{{").replace("}", "}}")
            safe_value = str(v).replace("{", "{{").replace("}", "}}")
            metadata_parts.append(f"- {safe_key}: {safe_value}")
    metadata_str = "\n".join(metadata_parts)

    # Build table with proper escaping
    header_row = "\t".join(str(col) for col in columns)
    row_strs = []
    for row in sample_rows[:10]:
        row_cells = []
        for col in columns:
            cell_value = str(row.get(col, "")).replace("{", "{{").replace("}", "}}")
            row_cells.append(cell_value)
        row_strs.append("\t".join(row_cells))
    table_str = "\n".join([header_row] + row_strs)

    # Use string concatenation instead of f-string to avoid format issues
    prompt = """

    You are a biomedical data curation assistant.

    You are evaluating a GEO platform (GPL) to determine the best mapping strategy for **gene identifiers**, where the goal is to compare gene expression levels across disease vs control samples.

    Your job is to read the metadata, column descriptions, and example values, and decide:

    1. Does the platform have probes that can be mapped to protein-coding genes?

    ## Mapping Priority System

    When evaluating columns for gene mapping, use this **strict priority order** (highest to lowest):

    ### Priority 1: Stable Database IDs (Most Reliable)
    - **Entrez Gene ID** / **Gene ID** / **GENE_NUM** (numeric identifiers like "2514089")
    - **Ensembl Gene ID** (ENSG* identifiers)
    - **Vega Gene ID** (OTTHUMG* identifiers - historical Vega project, now part of Ensembl)
    - **RefSeq mRNA** (NM_*, NR_* accessions)
    - **UniProt/SwissProt** (P*, Q* accessions like "Q13485")

    ### Priority 2: Sequence-based IDs
    - **GenBank accessions** (clear accession patterns)
    - **RefSeq protein** (NP_*, XP_* accessions)

    ### Priority 3: Genomic Coordinates
    - **Chromosomal coordinates** (chr1:123-456 format)
    - **START/STOP positions** with chromosome info

    ### Priority 4: Gene Symbols (Least Reliable)
    - **Gene symbols/names** only if no higher priority options exist
    - These are prone to ambiguity and version changes

    ## Column Evaluation Process

    For each potential mapping column, evaluate:

    1. **Column name and description**: What type of identifier does it claim to contain?
    2. **Actual values**: Do the values match the expected pattern?
    3. **Population rate**: Are most values populated (not empty/null)?
    4. **Value format consistency**: Are values in expected format?

    ## Mapping Method Classification

    - **"direct"**: Platform provides direct gene symbols AND they are the best available option
    - **"accession_lookup"**: Platform provides database accessions (RefSeq, GenBank, UniProt, Entrez)
    - **"coordinate_lookup"**: Platform provides genomic coordinates
    - **"unknown"**: No reliable mapping method identified

    ## Analysis Instructions

    1. Identify ALL potential mapping columns by examining names, descriptions, and sample values  
    2. Rank identified columns by the priority system above  
    3. Select the highest priority column with good data quality

    **Output Format** (replace placeholders with actual data):


    ```json
    {
    "gpl_id": "<actual GPL ID>",
    "organism": "<actual organism>",
    "mapping_method": "<direct|accession_lookup|coordinate_lookup|unknown>",
    "description": "<explanation of mapping strategy and field chosen>",
    "fields_used_for_mapping": ["<list of column names used>"],
    "alternate_fields_for_mapping": ["<list of other viable mapping columns if available>"],
    "available_fields": ["<list of all column names>"],
    "mapping_priority_rationale": "<explanation of why this field was chosen>"
    }
    ```

    **Platform Data**:
    - Platform ID: """ + str(gpl_id) + """
    - Metadata:
    """ + metadata_str + """
    - Column Descriptions:
    """ + column_desc_str + """
    - Sample Table:
    """ + table_str + """

    **Final Instruction**: Return only valid JSON with actual data for the GPL, based on the analysis. Do NOT include the placeholder template, code blocks, or any non-JSON content."""

    return prompt.strip()


def call_openai_chat_api_with_retry(prompt: str, api_url: str, api_key: str, model: str = "gpt-4o", max_retries: int = 5) -> str:
    """Call OpenAI API with exponential backoff retry logic"""
    if not api_key:
        raise ValueError("API key is missing.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    last_exception = None

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=60)

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('retry-after', 60))
                delay = min(retry_after, exponential_backoff(attempt, base_delay=2.0))
                raise requests.exceptions.RequestException(f"Rate limited, retrying after {delay:.1f}s")

            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"]

            # Parse JSON response
            try:
                content = content.strip()
                if content.startswith("```json"):
                    content = content[len("```json"):].strip()
                if content.endswith("```"):
                    content = content[:-3].strip()
                return content
            except json.JSONDecodeError as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to parse GPT response: {e}\nRaw content:\n{content}")
                last_exception = e

        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                break

            # Calculate delay
            delay = exponential_backoff(attempt)
            time.sleep(delay)

    raise last_exception or Exception("Max retries exceeded")


def infer_probe_mapping_strategy_with_retry(gpl_record: dict, progress: InferenceProgressTracker) -> Optional[str]:
    """Process a single GPL with retry logic"""
    gpl_id = gpl_record.get("gpl_id", "unknown")
    start_time = time.time()

    try:
        prompt = build_prompt(gpl_record)
        
        # Check if API credentials are available
        if not progress._api_url or not progress._api_key:
            progress.update("failed", gpl_id, "Missing API URL or key")
            return None

        for attempt in range(5):  # Max 5 retries
            try:
                if attempt > 0:
                    progress.update("retry", gpl_id, f"Attempt {attempt + 1}", attempt)

                result = call_openai_chat_api_with_retry(
                    prompt, 
                    progress._api_url, 
                    progress._api_key, 
                    max_retries=3
                )

                elapsed = time.time() - start_time
                progress.update("success", gpl_id, f"Inferred in {elapsed:.1f}s")
                return result

            except Exception as e:
                error_msg = str(e)
                print(f"\n🔍 Debug - Attempt {attempt + 1} failed for {gpl_id}: {error_msg}")
                
                if attempt < 4:  # Will retry
                    delay = exponential_backoff(attempt, base_delay=1.0)
                    time.sleep(delay)
                    continue
                else:  # Final attempt failed
                    elapsed = time.time() - start_time
                    progress.update("failed", gpl_id, f"All retries failed: {error_msg[:50]}...")
                    return None

    except Exception as e:
        error_msg = str(e)
        print(f"\n🔍 Debug - Outer exception for {gpl_id}: {error_msg}")
        elapsed = time.time() - start_time
        progress.update("failed", gpl_id, f"Error: {error_msg[:30]}...")
        return None


def process_gpl_jsonl_parallel(gpl_dict: Dict[str, Dict], max_workers: int = 4):
    """
    Process GPL records file with parallel inference

    Args:
        gpl_dict: Dictionary of GPL records
        max_workers: Number of parallel workers (keep low to respect API limits)
    """

    # Load all GPL records
    gpl_records = [value for value in gpl_dict.values()]
    print(f"📋 Loaded {len(gpl_records)} GPL records")
    if not gpl_records:
        print("❌ No valid GPL records found!")
        return

    print(f"🚀 Starting GPL Inference Pipeline")
    print(f"{'='*60}")
    print(f"📋 Total GPLs to process: {len(gpl_records)}")
    print(f"👥 Workers: {max_workers}")
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Initialize progress tracker
    progress = InferenceProgressTracker(len(gpl_records))
    results = {}
    errors = {}
    
    # Process GPLs in parallel
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_gpl = {
                executor.submit(infer_probe_mapping_strategy_with_retry, record, progress): record.get("gpl_id", "unknown")
                for record in gpl_records
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_gpl):
                gpl_id = future_to_gpl[future]
                try:
                    result = future.result()
                    if result:
                        results[gpl_id] = result
                    else:
                        errors[gpl_id] = "Processing failed or no data"
                except Exception as e:
                    print(f"\n⚠️ Unexpected error for {gpl_id}: {e}")
                    errors[gpl_id] = str(e)

    except KeyboardInterrupt:
        print(f"\n\n⚠️ Process interrupted by user!")

    finally:
        # Final progress report
        progress.final_report()
        print(f"\n✨ Inference complete! Returning results.")
    
    return {
        'results': results,
        'errors': errors,
        'statistics': {
            'total_requested': len(gpl_records),
            'successful': len(results),
            'failed': len(errors),
            'success_rate_percent': (len(results) / len(gpl_records) * 100) if gpl_records else 0,
            'total_processing_time_seconds': time.time() - progress.start_time,
            'average_time_per_gpl_seconds': (time.time() - progress.start_time) / len(gpl_records) if gpl_records else 0,
            'gpls_per_second': len(results) / (time.time() - progress.start_time) if (time.time() - progress.start_time) > 0 else 0,
            'max_workers_used': max_workers,
            'processed_at': datetime.now().isoformat()
        }
    }


def infer_probe_mapping_strategy(gpl_record: dict, api_url: str = None, api_key: str = None) -> str:
    """Original function for single GPL inference - kept for compatibility"""
    prompt = build_prompt(gpl_record)
    print(f"\n--- PROMPT SENT TO GPT ---\n{prompt}\n")
    
    api_url = api_url or os.getenv("GPT_API_URL")
    api_key = api_key or os.getenv("GPT_API_KEY")
    
    result = call_openai_chat_api_with_retry(prompt, api_url, api_key)
    return result
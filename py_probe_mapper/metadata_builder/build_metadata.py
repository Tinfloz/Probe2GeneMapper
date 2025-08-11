import requests
import gzip
import time
import random
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import json
import os
from datetime import datetime

class ProgressTracker:
    """Thread-safe progress tracker with rich logging"""
    def __init__(self, total_items: int):
        self.total = total_items
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.last_update = 0

    def update(self, status: str, gpl_id: str, details: str = ""):
        with self.lock:
            if status == "success":
                self.completed += 1
            elif status == "failed":
                self.failed += 1
            elif status == "skipped":
                self.skipped += 1

            # Update progress every 5 items or every 10 seconds
            now = time.time()
            processed = self.completed + self.failed + self.skipped

            if processed % 5 == 0 or now - self.last_update > 10:
                self.print_progress(gpl_id, details)
                self.last_update = now

    def print_progress(self, current_gpl: str = "", details: str = ""):
        elapsed = time.time() - self.start_time
        processed = self.completed + self.failed + self.skipped
        remaining = self.total - processed

        if processed > 0:
            rate = processed / elapsed
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
        else:
            rate = 0
            eta_minutes = 0

        percentage = (processed / self.total) * 100

        # Progress bar
        bar_length = 30
        filled = int(bar_length * processed / self.total)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"\r📊 [{bar}] {percentage:5.1f}% | "
              f"✅ {self.completed} | ❌ {self.failed} | ⏭️ {self.skipped} | "
              f"🚀 {rate:.1f}/s | ⏱️ ETA: {eta_minutes:.1f}m | 🎯 {current_gpl}",
              end="", flush=True)

        if details:
            print(f" | {details}", end="", flush=True)

    def final_report(self):
        elapsed = time.time() - self.start_time
        print(f"\n\n🎉 PROCESSING COMPLETE!")
        print(f"{'='*60}")
        print(f"⏰ Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"✅ Successful: {self.completed}")
        print(f"❌ Failed: {self.failed}")
        print(f"⏭️ Skipped: {self.skipped}")
        print(f"📈 Success rate: {(self.completed/(self.completed+self.failed)*100):.1f}%" if (self.completed+self.failed) > 0 else "N/A")
        print(f"🚀 Average rate: {self.total/elapsed:.1f} GPL/second")
        print(f"{'='*60}")

class GPLDatasetBuilder:
    """Class to build a GPL dataset with parallel processing, reusable in other modules"""
    
    def __init__(self, max_rows: int = 5, max_workers: int = 8):
        """
        Initialize the GPLDatasetBuilder.
        
        Args:
            max_rows: Maximum number of rows to sample per GPL.
            max_workers: Number of parallel workers.
            output_path: Default output file path for results.
        """
        self.max_rows = max_rows
        self.max_workers = max_workers

    @staticmethod
    def _gpl_url(gpl_id: str) -> str:
        """Generate NCBI FTP URL for GPL ID"""
        numeric = gpl_id[3:]
        if len(numeric) <= 3:
            return f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/{gpl_id}/soft/{gpl_id}_family.soft.gz"
        prefix = numeric[:-3]
        return f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL{prefix}nnn/{gpl_id}/soft/{gpl_id}_family.soft.gz"

    def _count_table_rows(self, gpl_id: str) -> int:
        """Quick count of total rows in the table"""
        url = self._gpl_url(gpl_id)
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            in_table = False
            row_count = 0

            with gzip.open(response.raw, 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith("!platform_table_begin"):
                        in_table = True
                        continue
                    elif line.startswith("!platform_table_end"):
                        break
                    elif in_table:
                        if row_count == 0:
                            # Skip header row
                            row_count += 1
                            continue
                        row_count += 1

            return row_count - 1  # Subtract 1 for header

        except Exception as e:
            raise Exception(f"Row counting failed: {str(e)}")

    def _extract_specific_rows(self, gpl_id: str, target_row_numbers: List[int]) -> List[Dict[str, str]]:
        """Extract only specific row numbers from the table"""
        url = self._gpl_url(gpl_id)
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            in_table = False
            header = []
            current_row = 0
            target_set = set(target_row_numbers)  # For O(1) lookup
            extracted_rows = []

            with gzip.open(response.raw, 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith("!platform_table_begin"):
                        in_table = True
                        continue
                    elif line.startswith("!platform_table_end"):
                        break
                    elif in_table:
                        line = line.strip()
                        if not header:
                            # Store header
                            header = line.split("\t")
                        else:
                            # Check if this is a row we want
                            if current_row in target_set:
                                row_values = line.split("\t")
                                # Pad row_values to match header length
                                while len(row_values) < len(header):
                                    row_values.append("")
                                row = dict(zip(header, row_values))
                                extracted_rows.append(row)

                                # Early exit if we got all rows we need
                                if len(extracted_rows) == len(target_row_numbers):
                                    break

                            current_row += 1

            return extracted_rows

        except Exception as e:
            raise Exception(f"Row extraction failed: {str(e)}")

    def _extract_table_smart_random(self, gpl_id: str, max_rows: Optional[int] = None) -> List[Dict[str, str]]:
        """Extract random rows using smart two-pass approach"""
        max_rows = max_rows or self.max_rows
        try:
            # Pass 1: Count total rows (fast)
            total_rows = self._count_table_rows(gpl_id)

            if total_rows == 0:
                return []

            # Generate random row numbers
            if total_rows <= max_rows:
                # If table is small, take all rows
                target_rows = list(range(total_rows))
            else:
                # Generate random row numbers (0-indexed)
                target_rows = sorted(random.sample(range(total_rows), max_rows))

            # Pass 2: Extract only the target rows (fast)
            extracted_rows = self._extract_specific_rows(gpl_id, target_rows)

            return extracted_rows

        except Exception as e:
            raise Exception(f"Smart random extraction failed: {str(e)}")

    def _extract_metadata(self, gpl_id: str) -> Dict[str, str]:
        """Extract metadata from GPL file"""
        url = self._gpl_url(gpl_id)
        metadata = {}
        fallback_description_lines = []
        column_descriptions = {}

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            # First pass: collect all lines to find column descriptions
            lines = []
            with gzip.open(response.raw, 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    lines.append(line)
                    if line.startswith("!platform_table_begin"):
                        break

            # Extract metadata from lines
            for line in lines:
                if not line.startswith("!"):
                    continue
                if line.startswith("!platform_table_begin"):
                    break  # Stop at start of table

                line = line.strip()
                if " = " in line:
                    parts = line.split(" = ", 1)
                    key = parts[0].replace("!", "").strip().lower()
                    value = parts[1].strip()
                    metadata[key] = value

                elif line.lower().startswith("!platform_comment"):
                    fallback_description_lines.append(line.split("=", 1)[-1].strip())

            if "platform_description" not in metadata:
                metadata["platform_description"] = " ".join(fallback_description_lines)

            # Extract column descriptions from comments before table
            for i, line in enumerate(lines):
                if line.startswith("!platform_table_begin"):
                    j = i - 1
                    comment_block = []
                    while j >= 0 and lines[j].startswith("#"):
                        comment_block.insert(0, lines[j].strip("#").strip())
                        j -= 1
                    for desc_line in comment_block:
                        if "=" in desc_line:
                            col, desc = desc_line.split("=", 1)
                            column_descriptions[col.strip()] = desc.strip()
                    break

            metadata["column_descriptions"] = column_descriptions

            return metadata
        except Exception as e:
            raise Exception(f"Metadata extraction failed: {str(e)}")

    def process_single_gpl(self, gpl_id: str, progress: ProgressTracker) -> Optional[Dict]:
        """Process a single GPL ID"""
        start_time = time.time()

        try:
            # Extract data using smart random sampling
            metadata = self._extract_metadata(gpl_id)
            table_rows = self._extract_table_smart_random(gpl_id)

            if not table_rows:
                progress.update("skipped", gpl_id, "No table data")
                return None

            record = {
                "gpl_id": gpl_id,
                "metadata": {
                    "title": metadata.get("platform_title", ""),
                    "technology": metadata.get("platform_technology", ""),
                    "organism": metadata.get("platform_organism", ""),
                    "description": metadata.get("platform_description", ""),
                    "column_descriptions": metadata.get("column_descriptions", {})
                },
                "table": {
                    "columns": list(table_rows[0].keys()) if table_rows else [],
                    "sample_rows": table_rows
                }
            }

            elapsed = time.time() - start_time
            progress.update("success", gpl_id, f"{len(table_rows)} smart random rows, {elapsed:.1f}s")
            return record

        except Exception as e:
            elapsed = time.time() - start_time
            progress.update("failed", gpl_id, f"Error: {str(e)[:50]}...")
            return None

    def build_dataset(self, gpl_ids: List[str], max_workers: Optional[int] = None, max_rows: Optional[int] = None) -> Dict[str, any]:
        """
        Build GPL dataset using parallel processing with rich progress tracking.

        Args:
            gpl_ids: List of GPL IDs to process.
            output_path: Output JSONL file path (overrides default).
            max_workers: Number of parallel workers (overrides default).
            max_rows: Maximum rows to extract per GPL (overrides default).

        Returns:
            Dictionary with results, errors, and statistics.
        """
        max_workers = max_workers or self.max_workers
        max_rows = max_rows or self.max_rows

        print(f"🚀 Starting GPL Dataset Builder")
        print(f"{'='*60}")
        print(f"📋 Total GPLs to process: {len(gpl_ids)}")
        print(f"👥 Workers: {max_workers}")
        print(f"📄 Max rows per GPL: {max_rows}")
        print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # Initialize progress tracker
        progress = ProgressTracker(len(gpl_ids))

        # Process GPLs in parallel
        results = {}
        errors = {}
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_gpl = {
                    executor.submit(self.process_single_gpl, gpl_id, progress): gpl_id
                    for gpl_id in gpl_ids
                }

                # Process completed tasks as they finish
                for future in as_completed(future_to_gpl):
                    gpl_id = future_to_gpl[future]
                    try:
                        record = future.result()
                        if record:
                            results[gpl_id] = record
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

            print(f"\n✨ Run complete! Returning results.")

        return {
            'results': results,
            'errors': errors,
            'statistics': {
                'total_requested': len(gpl_ids),
                'successful': len(results),
                'failed': len(errors),
                'success_rate_percent': (len(results) / len(gpl_ids) * 100) if gpl_ids else 0,
                'total_processing_time_seconds': time.time() - progress.start_time,
                'average_time_per_gpl_seconds': (time.time() - progress.start_time) / len(gpl_ids) if gpl_ids else 0,
                'gpls_per_second': len(results) / (time.time() - progress.start_time) if (time.time() - progress.start_time) > 0 else 0,
                'max_workers_used': max_workers,
                'processed_at': datetime.now().isoformat()
            }
        }

#if __name__ == "__main__":
#sample_gpls = ["GPL97", "GPL570", "GPL96"]
#builder = GPLDatasetBuilder(max_rows=5, max_workers=2)
#res = builder.build_dataset(sample_gpls)
#print(res['results'])


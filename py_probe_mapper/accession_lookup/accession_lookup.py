import logging
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter, Retry
from threading import Lock, Thread
from pathlib import Path
from datetime import datetime
import zarr
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import gzip

# Fix for Zarr compressor compatibility
IS_ZARR_V3 = zarr.__version__.startswith('3')

def create_zarr_dataset_safe(group, name, data, dtype='<U50'):
    """Safely create Zarr dataset with version compatibility."""
    try:
        if IS_ZARR_V3:
            return group.create_array(name, data=data)
        else:
            return group.create_dataset(name, data=data, compressor='blosc')
    except Exception as e:
        if IS_ZARR_V3:
            return group.create_array(name, data=data)
        else:
            return group.create_dataset(name, data=data)

class EnhancedGPLMapperV2:
    """Enhanced GPL mapper with fallback columns and delimiter handling."""

    def __init__(self, entry: Dict[str, Any], log_level: str = "INFO",
                 max_workers: int = 3, rate_limit_delay: float = 0.5):
        self.entry = entry
        self.gpl_id = entry["gpl_id"]
        self.mapping_method = entry.get("mapping_method")
        self.primary_fields = entry.get("fields_used_for_mapping", [])
        self.fallback_fields = entry.get("alternate_fields_for_mapping", [])
        self.gpl_url = self._gpl_url(self.gpl_id)
        self.mapping = {}
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.mapping_stats = {
            "total_probes": 0,
            "mapped_probes": 0,
            "unmapped_probes": 0,
            "primary_field_success": 0,
            "fallback_field_success": 0,
            "empty_both_fields": 0,
            "delimiter_separated_found": 0,
            "field_usage_stats": Counter(),
            "id_type_distribution": Counter(),
            "mapping_source_distribution": Counter(),
            "failed_mappings": []
        }
        self._stats_lock = Lock()
        self.logger = self._setup_logging(log_level)
        self.session = self._create_session()
        self._last_request_time = {}
        self._request_lock = Lock()
        self.common_delimiters = [',', ';', '|', '/', ' // ', ' /// ', '\t']
        self.id_patterns = {
            "entrezgene": [re.compile(r"^\d+$")],
            "ensemblgene": [
                re.compile(r"^ENS[A-Z]*G\d{11}$", re.I),
                re.compile(r"^ENS[A-Z]*\d+$", re.I),
            ],
            "ensembltranscript": [re.compile(r"^ENS[A-Z]*T\d{11}$", re.I)],
            "ensemblprotein": [re.compile(r"^ENS[A-Z]*P\d{11}$", re.I)],
            "vega_gene": [re.compile(r"^(OTTHUMG|OTTMUSG|OTTHUME|OTTMUSE)\d{11}$", re.I)],
            "vega_transcript": [re.compile(r"^(OTTHUMT|OTTMUST)\d{11}$", re.I)],
            "refseq_mrna": [re.compile(r"^(NM|XM)_\d+(\.\d+)?$", re.I)],
            "refseq_ncrna": [re.compile(r"^(NR|XR)_\d+(\.\d+)?$", re.I)],
            "refseq_protein": [re.compile(r"^(NP|XP|YP)_\d+(\.\d+)?$", re.I)],
            "genbank": [re.compile(r"^[A-Z]{1,4}\d{5,8}(\.\d+)?$", re.I)],
            "uniprot": [re.compile(r"^([OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d{5})$", re.I)],
            "symbol": [re.compile(r"^[A-Z][A-Z0-9\-_\.]*$", re.I)],
            "affymetrix": [re.compile(r"^.+_(at|s_at|x_at|a_at)$", re.I)],
        }

    def _setup_logging(self, log_level: str) -> logging.Logger:
        logger = logging.getLogger(f"GPLMapperV2_{self.gpl_id}")
        logger.setLevel(getattr(logging, log_level.upper()))
        logger.handlers = []
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def _gpl_url(self, gpl_id: str) -> str:
        numeric = gpl_id[3:]
        if len(numeric) <= 3:
            return f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/{gpl_id}/soft/{gpl_id}_family.soft.gz"
        prefix = numeric[:-3]
        return f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL{prefix}nnn/{gpl_id}/soft/{gpl_id}_family.soft.gz"

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504, 408, 520, 521, 522, 524],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            'User-Agent': 'GPL-Mapper/2.0 (+https://github.com/your-repo)',
            'Connection': 'keep-alive'
        })
        return session

    def _update_stats(self, **kwargs):
        with self._stats_lock:
            for key, value in kwargs.items():
                if key in ["mapped_probes", "unmapped_probes", "total_probes",
                          "primary_field_success", "fallback_field_success",
                          "empty_both_fields", "delimiter_separated_found"]:
                    self.mapping_stats[key] += value
                elif key == "field_used":
                    self.mapping_stats["field_usage_stats"][value] += 1
                elif key == "mapping_source":
                    self.mapping_stats["mapping_source_distribution"][value] += 1
                elif key == "id_type":
                    self.mapping_stats["id_type_distribution"][value] += 1
                elif key == "failed_mapping":
                    self.mapping_stats["failed_mappings"].append(value)

    def extract_ids_from_value(self, value: str) -> List[str]:
        if not value or not value.strip():
            return []
        value = value.strip()
        if self._is_valid_single_id(value):
            return [value]
        best_ids = []
        best_score = 0
        for delimiter in self.common_delimiters:
            if delimiter in value:
                parts = [part.strip() for part in value.split(delimiter) if part.strip()]
                if len(parts) > 1:
                    valid_parts = [part for part in parts if self._is_valid_single_id(part)]
                    score = len(valid_parts) / len(parts)
                    if score > best_score and len(valid_parts) > 0:
                        best_score = score
                        best_ids = valid_parts
        if best_ids:
            self._update_stats(delimiter_separated_found=1)
            return best_ids
        if self._looks_like_id(value):
            return [value]
        return []

    def _is_valid_single_id(self, value: str) -> bool:
        if not value or len(value) < 2:
            return False
        for id_type, patterns in self.id_patterns.items():
            for pattern in patterns:
                if pattern.match(value):
                    return True
        if (value.isdigit() and len(value) < 8) or \
           re.match(r"^[A-Z][A-Z0-9\-_\.]*$", value, re.I):
            return True
        return False

    def _looks_like_id(self, value: str) -> bool:
        if not value or len(value) < 2:
            return False
        if any(word in value.lower() for word in ['unknown', 'null', 'na', 'n/a', 'none', 'empty']):
            return False
        if not re.search(r'[A-Za-z0-9]', value):
            return False
        return True

    def get_mapping_value_for_probe(self, row: Dict[str, str], probe_id: str) -> Tuple[str, str, str]:
        for field in self.primary_fields:
            if field in row:
                raw_value = row[field].strip()
                if raw_value:
                    ids = self.extract_ids_from_value(raw_value)
                    if ids:
                        self._update_stats(field_used=field, primary_field_success=1)
                        return ids[0], field, "primary"
        for field in self.fallback_fields:
            if field in row:
                raw_value = row[field].strip()
                if raw_value:
                    ids = self.extract_ids_from_value(raw_value)
                    if ids:
                        self._update_stats(field_used=field, fallback_field_success=1)
                        return ids[0], field, "fallback"
        self._update_stats(empty_both_fields=1)
        return "", "", "none"

    def detect_id_type(self, value: str) -> Tuple[str, float]:
        if not value or not value.strip():
            return "unknown", 0.0
        value = value.strip()
        clean_value = re.sub(r'(_at|_s_at|_x_at|_a_at)$', '', value)
        for id_type, patterns in self.id_patterns.items():
            for pattern in patterns:
                if pattern.match(clean_value):
                    confidence = 0.95 if id_type in ["entrezgene", "ensemblgene", "vega_gene"] else 0.8
                    return id_type, confidence
        if clean_value.isdigit() and len(clean_value) < 8:
            return "entrezgene", 0.7
        elif re.match(r"^[A-Z][A-Z0-9\-_]*$", clean_value, re.I):
            return "symbol", 0.6
        else:
            return "accession", 0.5

    def download_gpl_file(self) -> List[Dict[str, str]]:
        self.logger.info(f"Downloading GPL annotation for {self.gpl_id}")
        try:
            response = self.session.get(self.gpl_url, stream=True, timeout=120)
            response.raise_for_status()
            in_table = False
            header = None
            extracted_rows = []
            with gzip.open(response.raw, 'rt', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.rstrip('\n')
                    if line.startswith("!platform_table_begin"):
                        in_table = True
                        continue
                    elif line.startswith("!platform_table_end"):
                        break
                    elif in_table:
                        if not header:
                            header = line.split("\t")
                        else:
                            row_values = line.split("\t")
                            while len(row_values) < len(header):
                                row_values.append("")
                            row = dict(zip(header, row_values))
                            extracted_rows.append(row)
            if not extracted_rows:
                raise ValueError("No table data found in GPL soft family file.")
            self.logger.info(f"Successfully extracted {len(extracted_rows)} rows")
            return extracted_rows
        except Exception as e:
            self.logger.error(f"Failed to download/parse GPL file: {str(e)}")
            raise

    def _rate_limit(self, service: str):
        with self._request_lock:
            current_time = time.time()
            if service in self._last_request_time:
                time_since_last = current_time - self._last_request_time[service]
                if time_since_last < self.rate_limit_delay:
                    sleep_time = self.rate_limit_delay - time_since_last
                    time.sleep(sleep_time)
            self._last_request_time[service] = time.time()

    def query_vega_hgnc_worker(self, vega_id: str) -> Tuple[str, Optional[str]]:
        try:
            self._rate_limit("hgnc")
            url = f"https://rest.genenames.org/search/vega_id:{vega_id}"
            headers = {"Accept": "application/json"}
            response = self.session.get(url, headers=headers, timeout=90)
            if response.status_code == 200:
                data = response.json()
                docs = data.get("response", {}).get("docs", [])
                if docs:
                    symbol = docs[0].get("symbol")
                    if symbol:
                        self._update_stats(mapping_source="hgnc")
                        return vega_id, symbol
            self._update_stats(failed_mapping={
                "id": vega_id, "type": "vega_gene", "reason": "no_hgnc_mapping"
            })
            return vega_id, None
        except Exception as e:
            self._update_stats(failed_mapping={
                "id": vega_id, "type": "vega_gene", "reason": f"hgnc_api_error: {str(e)}"
            })
            return vega_id, None

    def batch_query_vega_concurrent(self, vega_ids: List[str]) -> Dict[str, str]:
        if not vega_ids:
            return {}
        self.logger.info(f"Processing {len(vega_ids)} VEGA gene IDs via HGNC")
        vega_results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_vega = {
                executor.submit(self.query_vega_hgnc_worker, vega_id): vega_id
                for vega_id in vega_ids
            }
            for future in as_completed(future_to_vega):
                try:
                    returned_id, symbol = future.result(timeout=120)
                    if symbol:
                        vega_results[returned_id] = symbol
                except Exception as e:
                    vega_id = future_to_vega[future]
                    self.logger.error(f"VEGA worker task failed for {vega_id}: {str(e)}")
        return vega_results

    def batch_query_mygene(self, id_batches: Dict[str, List[str]]) -> Dict[str, str]:
        url = "https://mygene.info/v3/query"
        gene_map = {}
        batch_size = 1000
        headers = {"Content-Type": "application/json"}
        scope_mapping = {
            "entrezgene": "entrezgene",
            "ensemblgene": "ensembl.gene",
            "ensembltranscript": "ensembl.transcript",
            "ensemblprotein": "ensembl.protein",
            "refseq_mrna": "refseq.rna",
            "refseq_ncrna": "refseq.rna",
            "refseq_protein": "refseq.protein",
            "genbank": "accession",
            "uniprot": "uniprot.Swiss-Prot",
            "symbol": "symbol",
            "affymetrix": "reporter"
        }
        for id_type, ids in id_batches.items():
            if not ids:
                continue
            scope = scope_mapping.get(id_type, "symbol")
            self.logger.info(f"Querying MyGene.info for {len(ids)} {id_type} IDs")
            for i in range(0, len(ids), batch_size):
                batch = ids[i:i+batch_size]
                try:
                    self._rate_limit("mygene")
                    data = {
                        "q": batch,
                        "scopes": scope,
                        "fields": "symbol,entrezgene",
                        "species": "human",
                        "size": 1
                    }
                    response = self.session.post(url, json=data, headers=headers, timeout=90)
                    if response.status_code == 200:
                        hits = response.json()
                        if isinstance(hits, list):
                            for hit in hits:
                                query_id = hit.get("query")
                                symbol = hit.get("symbol")
                                if query_id and symbol:
                                    gene_map[query_id] = symbol
                                    self._update_stats(mapping_source="mygene")
                except Exception as e:
                    self.logger.error(f"Error in MyGene.info batch query: {str(e)}")
                    continue
        return gene_map

    def process_with_fallback(self, rows: List[Dict[str, str]]) -> None:
        probe_col = "ID" if "ID" in rows[0] else "SPOT_ID"
        id_groups = defaultdict(list)
        probe_to_mapping_info = {}
        for row in rows:
            probe_id = row.get(probe_col, "").strip()
            if not probe_id:
                continue
            mapping_value, field_used, source_type = self.get_mapping_value_for_probe(row, probe_id)
            if mapping_value:
                clean_id = re.sub(r'(_at|_s_at|_x_at|_a_at)$', '', mapping_value.strip())
                id_type, confidence = self.detect_id_type(clean_id)
                if confidence >= 0.5:
                    id_groups[id_type].append(clean_id)
                    probe_to_mapping_info[probe_id] = {
                        'clean_id': clean_id,
                        'id_type': id_type,
                        'field_used': field_used,
                        'source_type': source_type,
                        'confidence': confidence
                    }
                    self._update_stats(id_type=id_type)
            else:
                probe_to_mapping_info[probe_id] = {
                    'clean_id': '',
                    'id_type': 'none',
                    'field_used': 'none',
                    'source_type': 'none',
                    'confidence': 0.0
                }
        self.mapping_stats["total_probes"] = len(probe_to_mapping_info)
        vega_results = {}
        if "vega_gene" in id_groups:
            unique_vega_ids = list(set(id_groups['vega_gene']))
            vega_results = self.batch_query_vega_concurrent(unique_vega_ids)
        mygene_groups = {k: list(set(v)) for k, v in id_groups.items() if not k.startswith("vega")}
        mygene_results = self.batch_query_mygene(mygene_groups) if mygene_groups else {}
        all_results = {**mygene_results, **vega_results}
        mapped_count = 0
        for probe_id, info in probe_to_mapping_info.items():
            clean_id = info['clean_id']
            if clean_id and clean_id in all_results:
                symbol = all_results[clean_id]
                self.mapping[probe_id] = symbol
                mapped_count += 1
            else:
                self.mapping[probe_id] = ""
        self.mapping_stats["mapped_probes"] = mapped_count
        self.mapping_stats["unmapped_probes"] = len(probe_to_mapping_info) - mapped_count

    def run(self) -> Dict[str, Any]:
        if self.mapping_method != "accession_lookup":
            self.logger.warning(f"Mapping method is '{self.mapping_method}', skipping.")
            return {"success": False, "reason": "unsupported_mapping_method"}
        try:
            if not self.primary_fields:
                self.logger.error("No primary mapping fields provided, aborting.")
                return {"success": False, "reason": "no_primary_fields"}
            rows = self.download_gpl_file()
            available_columns = set(rows[0].keys()) if rows else set()
            missing_primary = [f for f in self.primary_fields if f not in available_columns]
            missing_fallback = [f for f in self.fallback_fields if f not in available_columns]
            if missing_primary:
                self.logger.warning(f"Missing primary fields: {missing_primary}")
            if missing_fallback:
                self.logger.warning(f"Missing fallback fields: {missing_fallback}")
            if all(f not in available_columns for f in self.primary_fields + self.fallback_fields):
                self.logger.error("No mapping fields available in the data")
                return {"success": False, "reason": "no_fields_available"}
            self.process_with_fallback(rows)
            return {
                "success": True,
                "gpl_id": self.gpl_id,
                "mapping": dict(self.mapping),
                "stats": dict(self.mapping_stats),
                "available_columns": list(available_columns),
                "missing_primary_fields": missing_primary,
                "missing_fallback_fields": missing_fallback
            }
        except Exception as e:
            self.logger.error(f"GPL mapping failed: {str(e)}")
            return {"success": False, "reason": f"processing_error: {str(e)}"}

class AccessionLookupGPLProcessor:
    """Lean GPL processor for programmatic usage with asynchronous HuggingFace upload."""

    def __init__(self, gpl_records: List[Dict[str, Any]], 
                 zarr_path: str = "gpl_mappings.zarr",
                 max_workers: int = None):
        """
        Initialize the GPL processor for programmatic usage.

        Args:
            gpl_records: List of GPL record dictionaries.
            zarr_path: Local path for Zarr dataset storage.
            max_workers: Number of parallel workers.
        """
        self.gpl_records = gpl_records
        self.zarr_path = Path(zarr_path)
        self.hf_repo = "Tinfloz/probe-gene-map"
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("EnhancedGPLProcessor")

    def filter_records(self) -> List[Dict[str, Any]]:
        """Filter GPL records for processing."""
        records = [
            record for record in self.gpl_records
            if record.get("mapping_method") == "accession_lookup"
        ]
        self.logger.info(f"Found {len(records)} suitable GPL records for processing")
        return records

    def initialize_zarr_dataset(self) -> 'zarr.Group':
        """Initialize Zarr dataset with proper versioning."""
        if self.zarr_path.exists():
            self.logger.info(f"Opening existing Zarr dataset at {self.zarr_path}")
            root = zarr.open_group(str(self.zarr_path), mode='r+')
        else:
            self.logger.info(f"Creating new Zarr dataset at {self.zarr_path}")
            root = zarr.open_group(str(self.zarr_path), mode='w')
            metadata = root.create_group('metadata')
            metadata.attrs.update({
                'created': datetime.now().isoformat(),
                'description': 'GPL probe-to-gene mappings dataset with fallback support',
                'version': '2.0',
                'zarr_version': zarr.__version__
            })
            root.create_group('mappings')
        return root

    def process_single_gpl_enhanced(self, entry: Dict[str, Any]) -> Tuple[str, bool, Dict[str, str], Dict]:
        """Process a single GPL entry using enhanced mapper."""
        gpl_id = entry["gpl_id"]
        try:
            mapper = EnhancedGPLMapperV2(
                entry=entry,
                log_level="WARNING",
                max_workers=2,
                rate_limit_delay=0.5
            )
            results = mapper.run()
            return (gpl_id, results["success"], results.get("mapping", {}), 
                    results.get("stats", {"error": results.get("reason", "unknown")}))
        except Exception as e:
            self.logger.error(f"Failed to process {gpl_id}: {str(e)}")
            return gpl_id, False, {}, {"error": str(e)}

    def save_gpl_to_zarr_enhanced(self, zarr_root: 'zarr.Group', gpl_id: str,
                                 mapping: Dict[str, str], stats: Dict, entry: Dict[str, Any]):
        """Save GPL mapping to Zarr dataset with enhanced metadata."""
        mappings_group = zarr_root['mappings']
        gpl_group = mappings_group.create_group(gpl_id)
        if mapping:
            probe_ids = np.array(list(mapping.keys()), dtype='<U50')
            gene_symbols = np.array(list(mapping.values()), dtype='<U50')
        else:
            probe_ids = np.array([], dtype='<U50')
            gene_symbols = np.array([], dtype='<U50')
        create_zarr_dataset_safe(gpl_group, 'probe_ids', probe_ids)
        create_zarr_dataset_safe(gpl_group, 'gene_symbols', gene_symbols)
        enhanced_stats = {}
        for key, value in stats.items():
            if isinstance(value, (Counter, defaultdict)):
                enhanced_stats[key] = dict(value)
            elif isinstance(value, list):
                enhanced_stats[key] = [str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item for item in value]
            elif isinstance(value, (str, int, float, bool, type(None))):
                enhanced_stats[key] = value
            else:
                enhanced_stats[key] = str(value)
        enhanced_stats['primary_fields'] = entry.get('fields_used_for_mapping', [])
        enhanced_stats['fallback_fields'] = entry.get('alternate_fields_for_mapping', [])
        attrs_dict = {
            'gpl_id': str(gpl_id),
            'organism': str(entry.get('organism', '')),
            'description': str(entry.get('description', '')),
            'primary_fields': entry.get('fields_used_for_mapping', []),
            'fallback_fields': entry.get('alternate_fields_for_mapping', []),
            'processing_timestamp': datetime.now().isoformat(),
            'mapping_count': int(len(mapping)),
            'mapper_version': '2.0',
            'stats': enhanced_stats
        }
        gpl_group.attrs.update(attrs_dict)
        

    def process_all_enhanced(self):
        """Process all GPL records and initiate asynchronous HuggingFace upload."""
        records = self.filter_records()
        if not records:
            self.logger.info("No records to process")
            return {}
        zarr_root = self.initialize_zarr_dataset()
        new_mappings = {}
        successful = 0
        failed = 0
        total_mappings = 0
        field_usage_summary = Counter()
        actual_workers = min(self.max_workers, 3)
        self.logger.info(f"Processing {len(records)} GPLs with {actual_workers} workers")
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(self.process_single_gpl_enhanced, record): record
                      for record in records}
            with tqdm(total=len(futures), desc="Processing GPLs") as pbar:
                for future in as_completed(futures):
                    record = futures[future]
                    try:
                        gpl_id, success, mapping, stats = future.result(timeout=600)
                        if success:
                            self.save_gpl_to_zarr_enhanced(zarr_root, gpl_id, mapping, stats, record)
                            new_mappings[gpl_id] = mapping
                            successful += 1
                            total_mappings += len(mapping)
                            if 'field_usage_stats' in stats:
                                for field, count in stats['field_usage_stats'].items():
                                    field_usage_summary[field] += count
                        else:
                            failed += 1
                    except Exception as e:
                        self.logger.error(f"Processing failed for {record.get('gpl_id', 'unknown')}: {e}")
                        failed += 1
                    pbar.update(1)
                    pbar.set_postfix({'Success': successful, 'Failed': failed, 'Mappings': total_mappings})
        metadata = zarr_root['metadata']
        metadata.attrs.update({
            'last_updated': datetime.now().isoformat(),
            'total_gpls_processed': len(zarr_root['mappings']),
            'total_probe_mappings': total_mappings,
            'field_usage_summary': dict(field_usage_summary),
            'processing_summary': {
                'successful': successful,
                'failed': failed,
                'total_processed': len(records)
            }
        })
        # Start asynchronous upload to HuggingFace
        #upload_thread = Thread(target=self.push_to_huggingface)
        #upload_thread.start()
        self.logger.info("Processing complete")
        return new_mappings

    def get_gpl_mapping(self, gpl_id: str) -> Optional[Dict[str, str]]:
        """Get mapping for a specific GPL platform from local Zarr."""
        if self.zarr_path.exists():
            root = zarr.open_group(str(self.zarr_path), mode='r')
            if gpl_id in root['mappings']:
                gpl_group = root['mappings'][gpl_id]
                probe_ids = list(gpl_group['probe_ids'][:])
                gene_symbols = list(gpl_group['gene_symbols'][:])
                return dict(zip(probe_ids, gene_symbols))
        return None
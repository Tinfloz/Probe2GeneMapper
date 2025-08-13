import requests
import time
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter, Retry
import logging
from typing import Dict, List, Any, Optional, Tuple, override, Set
from collections import Counter, defaultdict
from threading import Lock, Thread
import re
import os
import gc
from py_probe_mapper.accession_lookup.accession_lookup import EnhancedGPLMapperV2, AccessionLookupGPLProcessor

class CoordinateLookupGPLMapperUtils(EnhancedGPLMapperV2):

    def __init__(self, entry: Dict[str, Any], log_level: str = "INFO",
                 max_workers: int = 3, rate_limit_delay: float = 0.5):
        
        super().__init__(entry, log_level, max_workers, rate_limit_delay)

        self.id_patterns = {
            "range_gb": [
                re.compile(r"^NC_\d+\.\d+$", re.I),
                re.compile(r"^NG_\d+\.\d+$", re.I),
                re.compile(r"^NT_\d+\.\d+$", re.I),
            ],
            "sequence": [re.compile(r'^[ACGTNRY]+$', re.I)],
            "ranges":[re.compile(r"^\d+$")],
            "chromosome":[re.compile(r"^chr(\d+|[XYM])$", re.I)],
            "spot_id":[re.compile(r"^chr[0-9XYM]+:\s*\d+-\d+$", re.I)]
        }
        self._tsv_path = os.path.join(os.path.dirname(__file__), '..', 'genome_utils', 'Homo_sapiens.GRCh38.genes.tsv')
        self.gene_db = None
        self._chromosome_intervals = {}
        self._load_gene_db()

    #load gene db
    def _load_gene_db(self) -> None:
        """Load and optimize the gene database for fast coordinate-based queries."""
        try:
            self.logger.info(f"Loading gene database from {self._tsv_path}")
            start_time = time.time()
            self.gene_db = pd.read_csv(
                self._tsv_path,
                sep='\t',
                dtype={
                    'chromosome': 'category',
                    'start': 'int32',        
                    'end': 'int32',
                    'gene_name': 'string',  
                    'gene_id': 'string'  
                },
                usecols=['chromosome', 'start', 'end', 'gene_name', 'gene_id'],  # Only load needed columns
                engine='c'
            )
            
            
            # Create optimized index for interval queries
            self.gene_db = self.gene_db.sort_values(['chromosome', 'start', 'end'])
            
            # Create chromosome groups for faster filtering
            self._precompute_chromosome_data()
            
            load_time = time.time() - start_time
            self.logger.info(f"Loaded {len(self.gene_db)} genes in {load_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Failed to load gene database: {str(e)}")
            self.gene_db = None

    def _precompute_chromosome_data(self) -> None:
        """Pre-compute chromosome-specific data structures for ultra-fast lookups."""
        self._chromosome_data = {}
        
        for chrom in self.gene_db['chromosome'].cat.categories:
            chr_data = self.gene_db[self.gene_db['chromosome'] == chrom].copy()
            if not chr_data.empty:
                # Convert to numpy arrays for vectorized operations
                self._chromosome_data[chrom] = {
                    'starts': chr_data['start'].values,
                    'ends': chr_data['end'].values,
                    'gene_names': chr_data['gene_name'].values,
                    'gene_ids': chr_data['gene_id'].values,
                    'chromosomes': chr_data['chromosome'].values
                }

    #look up overlapping genes
    def _find_overlapping_genes_ultra_vectorized(self, probe_lookups: Dict[str, Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """Ultra-optimized vectorized gene finding using numpy operations."""
        if self.gene_db is None:
            self.logger.error("Gene database not loaded")
            return {}
            
        results = {}
        
        # Group probes by chromosome for batch processing
        chr_probe_groups = defaultdict(list)
        
        for probe_id, coords in probe_lookups.items():
            if all(coords.get(k) for k in ['chromosome', 'range_start', 'range_end']):
                try:
                    chrom = str(coords['chromosome'])
                    start = int(coords['range_start'])
                    end = int(coords['range_end'])
                    chr_probe_groups[chrom].append((probe_id, start, end))
                except (ValueError, TypeError):
                    results[probe_id] = []
            else:
                results[probe_id] = []
        
        # Process each chromosome batch with numpy vectorization
        for chrom, probes in chr_probe_groups.items():
            if chrom not in self._chromosome_data:
                for probe_id, _, _ in probes:
                    results[probe_id] = []
                continue
            
            chr_data = self._chromosome_data[chrom]
            gene_starts = chr_data['starts']
            gene_ends = chr_data['ends']
            gene_names = chr_data['gene_names']
            gene_ids = chr_data['gene_ids']
            
            # Vectorized overlap detection for all probes at once
            for probe_id, probe_start, probe_end in probes:
                # Numpy boolean indexing for blazing fast overlap detection
                overlaps = (gene_starts <= probe_end) & (gene_ends >= probe_start)
                overlap_indices = np.where(overlaps)[0]
                
                if len(overlap_indices) > 0:
                    genes = []
                    for idx in overlap_indices:
                        gene_name = gene_names[idx] if pd.notna(gene_names[idx]) else None
                        gene_id = gene_ids[idx] if pd.notna(gene_ids[idx]) else None
                        
                        genes.append({
                            'gene_name': gene_name,
                            'gene_id': gene_id,
                            'chromosome': chrom,
                            'start': int(gene_starts[idx]),
                            'end': int(gene_ends[idx])
                        })
                    results[probe_id] = genes
                else:
                    results[probe_id] = []
        
        return results
    
    def _batch_process_probes(self, probe_lookups: Dict[str, Dict[str, str]], 
                             batch_size: int = 50000) -> Dict[str, List[Dict[str, str]]]:
        """Process probes in batches to manage memory efficiently for large datasets."""
        all_results = {}
        probe_items = list(probe_lookups.items())
        
        for i in range(0, len(probe_items), batch_size):
            batch = dict(probe_items[i:i + batch_size])
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(probe_items) + batch_size - 1)//batch_size}")
            
            batch_results = self._find_overlapping_genes_ultra_vectorized(batch)
            all_results.update(batch_results)
            
            if len(probe_lookups) > 1000000:
                gc.collect()
        return all_results

    #extract missing genes
    def _extract_missing_gene_names(self, gene_results: Dict[str, List[Dict[str, str]]]) -> Set[str]:
        """Extract unique gene IDs that are missing gene names."""
        missing_gene_ids = set()
        
        for genes in gene_results.values():
            for gene in genes:
                if gene.get('gene_id') and not gene.get('gene_name'):
                    # Extract ENSEMBL gene ID
                    gene_id = gene.get('gene_id')
                    if gene_id and gene_id.startswith('ENSG') and not gene.get('gene_name'):
                        missing_gene_ids.add(gene_id)
        
        return missing_gene_ids
    
    def _update_gene_names(self, gene_results: Dict[str, List[Dict[str, str]]], 
                          gene_name_mapping: Dict[str, str]) -> None:
        """Update gene results with fetched gene names."""
        for genes in gene_results.values():
            for gene in genes:
                gene_id = gene.get('gene_id')
                if gene_id in gene_name_mapping:
                    gene['gene_name'] = gene_name_mapping[gene_id]
                if gene.get('gene_name') is None:
                    gene['gene_name'] = ''

    def map_coordinates_to_genes_optimized(self, probe_lookups: Dict[str, Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """
        Main method to map probe coordinates to genes with blazing speed.
        
        Args:
            probe_lookups: Dictionary with probe_id as key and coordinate info as value
                          Expected format: {probe_id: {"chromosome": "17", "range_start": "123456", "range_end": "123567"}}
        
        Returns:
            Dictionary mapping probe_id to list of overlapping genes with their information
        """
        if not probe_lookups:
            return {}
        num_probes = len(probe_lookups)
        self.logger.info(f"Processing {num_probes:,} probe coordinates")
        start_time = time.time()
        # Use batching for very large datasets
        if num_probes > 100000:
            gene_results = self._batch_process_probes(probe_lookups, batch_size=50000)
        else:
            gene_results = self._find_overlapping_genes_ultra_vectorized(probe_lookups)
        overlap_time = time.time()
        self.logger.info(f"Found overlapping genes in {overlap_time - start_time:.2f} seconds")
        # Extract missing gene names more efficiently
        missing_gene_ids = self._extract_missing_gene_names(gene_results)
        if missing_gene_ids:
            self.logger.info(f"Found {len(missing_gene_ids)} genes with missing names, querying MyGene.info")
            # Batch query with larger chunks for efficiency
            id_batches = {"ensemblgene": list(missing_gene_ids)}
            gene_name_mapping = self.batch_query_mygene(id_batches)
            self._update_gene_names(gene_results, gene_name_mapping)
            
            mygene_time = time.time()
            self.logger.info(f"MyGene.info queries completed in {mygene_time - overlap_time:.2f} seconds")
        
        total_time = time.time() - start_time
        self.logger.info(f"Total coordinate mapping completed in {total_time:.2f} seconds for {num_probes:,} probes")
        
        return gene_results
    
    @override
    def _is_valid_single_id(self, values: List[str]) -> bool:
        for value in values:
            matched_any = False
            for id_type, patterns in self.id_patterns.items():
                if any(pattern.match(value) for pattern in patterns):
                    matched_any = True
                    break
            if not matched_any:
                return False
        return True
    
    @override
    def extract_ids_from_value(self, values: List[str]) -> List[str]:
        stripped_vals = [x.strip() for x in values]
        if (len(values) == 0) or (len(stripped_vals) < len(values)):
            return []
        if self._is_valid_single_id(stripped_vals):
            return stripped_vals
        return []
        
    @override
    def get_mapping_value_for_probe(self, row: Dict[str, str], probe_id: str) -> Tuple[List[str] | str, List[str] | str, str]:
        ids = self.extract_ids_from_value([row[field].strip() for field in self.primary_fields if field in row])
        if len(ids) == len(self.primary_fields):
            self._update_stats(field_used=", ".join(self.primary_fields))
            return ids, self.primary_fields, "primary"
        return "", "", "none"
    
    @override
    def detect_id_type(self, values: List[str]) -> Tuple[List[str], float]:
        values_stripped = [x.strip() for x in values]
        if (len(values) == 0) or (len(values_stripped) < len(values)):
            return "unknown", 0.0
        id_types = []
        for value in values_stripped:
            for id_type, patterns in self.id_patterns.items():
                for pattern in patterns:
                    if pattern.match(value):
                        id_types.append(id_type)
        return id_types, 0.95
    
    _range_gb_cache = {}
    _spot_id_cache = {}
    @staticmethod
    def get_chr_from_range_gb_batch(range_gb: str) -> str | None:
        if range_gb in CoordinateLookupGPLMapperUtils._range_gb_cache:
            return CoordinateLookupGPLMapperUtils._range_gb_cache[range_gb]
        mito_accessions = {"012920", "001807"}  
        base = re.sub(r'^(NC|NG|NT)_', '', range_gb, flags=re.I).split('.')[0]
        result = None
        try:
            base_int = int(base)
            if 1 <= base_int <= 22:
                result = base
            elif base_int == 23:
                result = "X"
            elif base_int == 24:
                result = "Y"
            elif base in mito_accessions:
                result = "MT"
        except ValueError:
            pass
        CoordinateLookupGPLMapperUtils._range_gb_cache[range_gb] = result
        return result
    
    @staticmethod
    def normalise_chromosome(chromosome: str) -> str:
        if not chromosome:
            return chromosome
        return chromosome.replace("Chr", "").replace("chr", "")
    
    @staticmethod
    def get_lookup_values_from_spot_id(spot_id: str) -> Tuple[str | None, str | None, str | None]:
        if spot_id in CoordinateLookupGPLMapperUtils._spot_id_cache:
            return CoordinateLookupGPLMapperUtils._spot_id_cache[spot_id]
        regex = re.compile(r"^(chr[0-9XYM]+):\s*(\d+)-(\d+)$", re.I)
        match = regex.match(spot_id)
        result = (None, None, None)
        if match:
            try:
                result = (
                    CoordinateLookupGPLMapperUtils.normalise_chromosome(match.group(1)), 
                    match.group(2), 
                    match.group(3)
                )
            except IndexError:
                pass
        
        CoordinateLookupGPLMapperUtils._spot_id_cache[spot_id] = result
        return result
    
    @override
    def process_with_fallback(self, rows: List[Dict[str, str]]) -> None:
        probe_col = "ID" if "ID" in rows[0] else "SPOT_ID"
        to_be_mapped = []
        probe_to_mapping_info = {}
        for row in rows:
            probe_id = row.get(probe_col, "").strip()
            if not probe_id:
                continue
            ids, fields_used, source_type = self.get_mapping_value_for_probe(row, probe_id)
            if isinstance(ids, list) and (len(ids) != 0) and (len(ids) == len(fields_used)):
                id_types, confidence = self.detect_id_type(ids)
                fields_to_ids = [{k:v} for k, v in zip(fields_used, ids)]
                # to_be_mapped = [{probe_id:[('chr', {"col_a":1}), ('range', {"col_b":2}), ('range', {"col_c":3})]}]
                to_be_mapped.append({
                    probe_id:list(zip(id_types, fields_to_ids))
                })
                probe_to_mapping_info[probe_id] = {
                    'ids': ids,
                    'id_types': id_types,
                    'field_used': fields_used,
                    'source_type': source_type,
                    'confidence': confidence
                }
                self._update_stats(id_type=", ".join(id_types))
            else:
                probe_to_mapping_info[probe_id] = {
                    'ids': '',
                    'id_type': 'none',
                    'field_used': 'none',
                    'source_type': 'none',
                    'confidence': 0.0
                }
        self.mapping_stats["total_probes"] = len(probe_to_mapping_info)
        probe_lookups = self.map_chr_start_end_to_gene(to_be_mapped)
        gene_results = self.map_coordinates_to_genes_optimized(probe_lookups)
        mapped_count = 0
        for k, v in gene_results.items():
            if len(v) == 0:
                self.mapping[k] = ''
            else:
                self.mapping[k] = v[0]['gene_name']
                mapped_count+=1
        self.mapping_stats['mapped_probes'] = mapped_count
        self.mapping_stats['unmapped_probes'] = len(probe_to_mapping_info) - mapped_count

    #{probe_id:[('chr', {"col_a":1}), ('range', {"col_b":2}), ('range', {"col_c":3})]}
    def map_chr_start_end_to_gene(self, rows:List[Dict[str, List[Tuple[str, Dict[str, str]]]]]) -> Dict[str, str]:
        probe_lookups = {} #{probe_id:{chromosome:"17", range_start:"2345567", "range_end":"2345567"}}
        for row in rows:
            probe_id = next(iter(row))
            lookup_values = {"chromosome":None, "range_start": None, "range_end": None}
            sequence = None
            spot_id = None
            for k, v in row[probe_id]:
                try:
                    sub_key = next(iter(v)).lower()
                    value = next(iter(v.values()))
                except StopIteration:
                    continue
                if k == "spot_id":
                    spot_id = value  # Defer processing until after loop
                elif k == "chromosome" and lookup_values["chromosome"] is None:
                    lookup_values["chromosome"] = CoordinateLookupGPLMapperUtils.normalise_chromosome(value)
                elif k == "range_gb" and lookup_values["chromosome"] is None:
                    lookup_values["chromosome"] = CoordinateLookupGPLMapperUtils.get_chr_from_range_gb_batch(value)
                elif k == "sequence":
                    sequence = value
                elif sub_key in ["range_start", "start"]:
                    lookup_values["range_start"] = value
                elif sub_key in ["range_end", "stop", "range_stop", "end"]:
                    lookup_values["range_end"] = value
            if spot_id is not None:
                lookup_values["chromosome"], lookup_values["range_start"], lookup_values["range_end"] = CoordinateLookupGPLMapperUtils.get_lookup_values_from_spot_id(spot_id)
            elif lookup_values["range_end"] is None and lookup_values["range_start"] is not None and sequence is not None:
                try:
                    lookup_values["range_end"] = str(int(lookup_values["range_start"]) + len(sequence) - 1)
                except ValueError:
                    pass
            probe_lookups[probe_id] = lookup_values 
        return probe_lookups
    
    @override
    def run(self) -> Dict[str, Any]:
        if self.mapping_method != "coordinate_lookup":
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


class CoordinateLookupGPLProcessor(AccessionLookupGPLProcessor):
      
    def __init__(self, gpl_records: List[Dict[str, Any]], 
                 zarr_path: str = "gpl_mappings.zarr",
                 max_workers: int = None):
          
        super().__init__(gpl_records, zarr_path, max_workers)

    @override
    def filter_records(self) -> List[Dict[str, Any]]:
        """Filter GPL records for processing."""
        records = [
            record for record in self.gpl_records
            if record.get("mapping_method") == "coordinate_lookup"
        ]
        self.logger.info(f"Found {len(records)} suitable GPL records for processing")
        return records
    
    @override
    def process_single_gpl_enhanced(self, entry: Dict[str, Any]) -> Tuple[str, bool, Dict[str, str], Dict]:
        """Process a single GPL entry using enhanced mapper."""
        gpl_id = entry["gpl_id"]
        try:
            mapper = CoordinateLookupGPLMapperUtils(
                entry=entry,
                log_level="WARNING",
                max_workers=2,
                rate_limit_delay=0.5
            )
            results = mapper.run()
            self.logger.info("results saved in error.txt")
            return (gpl_id, results["success"], results.get("mapping", {}), 
                    results.get("stats", {"error": results.get("reason", "unknown")}))
        except Exception as e:
            self.logger.error(f"Failed to process {gpl_id}: {str(e)}")
            return gpl_id, False, {}, {"error": str(e)}
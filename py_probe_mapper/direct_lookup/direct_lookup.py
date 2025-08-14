from typing import Dict, List, Any, Tuple, override
import time
from py_probe_mapper.accession_lookup.accession_lookup import EnhancedGPLMapperV2, AccessionLookupGPLProcessor

class DirectLookupUtils(EnhancedGPLMapperV2):
    """Optimized direct gene symbol lookup - maps probe IDs directly to gene symbols."""

    def __init__(self, entry: Dict[str, Any], log_level: str = "INFO",
                 max_workers: int = 3, rate_limit_delay: float = 0.5):
        
        super().__init__(entry, log_level, max_workers, rate_limit_delay)

    def _is_valid_gene_symbol(self, value: str) -> bool:
        """Ultra-fast validation - just check if value exists and isn't empty."""
        return bool(value and str(value).strip() and str(value).strip().upper() not in {'NA', 'NULL', 'NAN', '', '---'})

    @override
    def process_with_fallback(self, rows: List[Dict[str, str]]) -> None:
        """Optimized direct mapping - single pass through data."""
        if not rows:
            return
            
        start_time = time.time()
        probe_col = "ID" if "ID" in rows[0] else "SPOT_ID"
        
        # Get the primary gene symbol column (should be single column for direct mapping)
        gene_symbol_col = self.primary_fields[0] if self.primary_fields else None
        
        if not gene_symbol_col:
            self.logger.error("No gene symbol column specified for direct mapping")
            return
            
        if gene_symbol_col not in rows[0]:
            self.logger.error(f"Gene symbol column '{gene_symbol_col}' not found in data")
            return
        
        total_probes = len(rows)
        mapped_count = 0
        empty_probes = 0
        invalid_symbols = 0
        
        self.logger.info(f"Processing {total_probes:,} probes for direct gene symbol mapping")
        
        # Single optimized pass through all rows
        for row in rows:
            probe_id = row.get(probe_col, "").strip()
            
            if not probe_id:
                empty_probes += 1
                continue
                
            # Extract gene symbol
            gene_symbol = row.get(gene_symbol_col, "")
            
            if self._is_valid_gene_symbol(gene_symbol):
                # Clean and store the gene symbol
                clean_symbol = str(gene_symbol).strip()
                self.mapping[probe_id] = clean_symbol
                mapped_count += 1
            else:
                # Store empty string for unmapped probes
                self.mapping[probe_id] = ''
                invalid_symbols += 1
        
        # Update statistics
        self.mapping_stats.update({
            "total_probes": total_probes,
            "mapped_probes": mapped_count,
            "unmapped_probes": total_probes - mapped_count,
            "empty_probe_ids": empty_probes,
            "invalid_symbols": invalid_symbols,
            "mapping_rate": (mapped_count / total_probes * 100) if total_probes > 0 else 0,
            "primary_field_used": gene_symbol_col,
            "processing_time_seconds": time.time() - start_time
        })
        
        self.logger.info(f"Direct mapping completed in {self.mapping_stats['processing_time_seconds']:.2f}s: "
                        f"{mapped_count:,}/{total_probes:,} probes mapped "
                        f"({self.mapping_stats['mapping_rate']:.1f}%)")
        
        if empty_probes > 0:
            self.logger.warning(f"Found {empty_probes} rows with empty probe IDs")
        if invalid_symbols > 0:
            self.logger.warning(f"Found {invalid_symbols} invalid/empty gene symbols")

    @override  
    def run(self) -> Dict[str, Any]:
        """Streamlined execution for direct mapping."""
        if self.mapping_method != "direct":
            self.logger.warning(f"Mapping method is '{self.mapping_method}', expected 'direct'")
            return {"success": False, "reason": "unsupported_mapping_method"}
        
        if not self.primary_fields:
            self.logger.error("No primary mapping fields provided")
            return {"success": False, "reason": "no_primary_fields"}
            
        if len(self.primary_fields) > 1:
            self.logger.warning(f"Multiple primary fields provided for direct mapping: {self.primary_fields}. Using first: {self.primary_fields[0]}")
        
        try:
            start_time = time.time()
            
            # Download and validate data
            rows = self.download_gpl_file()
            if not rows:
                return {"success": False, "reason": "no_data_downloaded"}
            
            available_columns = set(rows[0].keys())
            gene_symbol_col = self.primary_fields[0]
            
            if gene_symbol_col not in available_columns:
                self.logger.error(f"Required gene symbol column '{gene_symbol_col}' not found. Available: {list(available_columns)}")
                return {"success": False, "reason": "missing_gene_symbol_column"}
            
            # Perform direct mapping
            self.process_with_fallback(rows)
            
            total_time = time.time() - start_time
            
            # Return comprehensive results
            return {
                "success": True,
                "gpl_id": self.gpl_id,
                "mapping": dict(self.mapping),
                "stats": dict(self.mapping_stats),
                "total_processing_time": total_time,
                "available_columns": list(available_columns),
                "gene_symbol_column_used": gene_symbol_col
            }
            
        except Exception as e:
            self.logger.error(f"Direct mapping failed for {self.gpl_id}: {str(e)}")
            return {"success": False, "reason": f"processing_error: {str(e)}"}

    # Remove unused override methods - inherit defaults from parent
    # This eliminates unnecessary complexity for direct mapping


class DirectLookupGPLProcessor(AccessionLookupGPLProcessor):
    """Optimized processor for direct gene symbol lookup."""
      
    def __init__(self, gpl_records: List[Dict[str, Any]], 
                 zarr_path: str = "gpl_mappings.zarr",
                 max_workers: int = None):
        super().__init__(gpl_records, zarr_path, max_workers)

    @override
    def filter_records(self) -> List[Dict[str, Any]]:
        """Filter for direct mapping records only."""
        records = [r for r in self.gpl_records if r.get("mapping_method") == "direct"]
        self.logger.info(f"Filtered {len(records)} GPL records for direct lookup processing")
        return records
    
    @override
    def process_single_gpl_enhanced(self, entry: Dict[str, Any]) -> Tuple[str, bool, Dict[str, str], Dict]:
        """Process single GPL with optimized direct lookup."""
        gpl_id = entry["gpl_id"]
        
        try:
            start_time = time.time()
            
            mapper = DirectLookupUtils(
                entry=entry,
                log_level="WARNING",  # Reduce noise for batch processing
                max_workers=1,        # Direct mapping doesn't need threading
                rate_limit_delay=0.1  # Minimal delay for direct mapping
            )
            
            results = mapper.run()
            processing_time = time.time() - start_time
            
            if results["success"]:
                stats = results.get("stats", {})
                stats["total_processing_time"] = processing_time
                self.logger.info(f"✓ {gpl_id}: {stats.get('mapped_probes', 0):,} probes mapped in {processing_time:.2f}s")
            else:
                self.logger.warning(f"✗ {gpl_id}: Failed - {results.get('reason', 'unknown error')}")
            
            return (gpl_id, results["success"], results.get("mapping", {}), 
                   results.get("stats", {"error": results.get("reason", "unknown")}))
                    
        except Exception as e:
            self.logger.error(f"Exception processing {gpl_id}: {str(e)}")
            return gpl_id, False, {}, {"error": str(e)}
        
import requests
from requests.adapters import HTTPAdapter, Retry
import logging
from typing import Dict, List, Any, Optional, Tuple, override
from collections import Counter, defaultdict
from threading import Lock, Thread
import re
from py_probe_mapper.accession_lookup.accession_lookup import EnhancedGPLMapperV2

class CoordinateLookupGPLMapperUtils(EnhancedGPLMapperV2):

    def __init__(self, entry: Dict[str, Any], log_level: str = "INFO",
                 max_workers: int = 3, rate_limit_delay: float = 0.5):
        
        super.__init__(entry, log_level, max_workers, rate_limit_delay)

        self.id_patterns = {
            "range_gb": [
                re.compile(r"^NC_\d+\.\d+$", re.I),
                re.compile(r"^NG_\d+\.\d+$", re.I),
                re.compile(r"^NT_\d+\.\d+$", re.I),
            ],
            "sequence": [re.compile(r'^[ACGTNRY]+$', re.I)],
            "ranges":[re.compile("^\d+$")],
            "chromosome":[re.compile("^chr(\d+|[XYM])$", re.I)],
            "spot_id":[re.compile(r"^chr[0-9XYM]+:\s*\d+-\d+$", re.I)]
        }

    @override
    def _is_valid_single_id(self, values: List[str]) -> bool:
        for i in values:
            for id_type, patterns in self.id_patterns.items():
                for pattern in patterns:
                    if not pattern.match(i):
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
    
    @staticmethod
    def get_chr_from_range_gb_batch(range_gb: str) -> str | None:
        mito_accessions = {"012920", "001807"}  
        base = re.sub(r'^(NC|NG|NT)_', '', range_gb, flags=re.I).split('.')[0]
        if 1 <= int(base) <= 22:
            return base
        elif int(base) == 23:
            return "X"
        elif int(base) == 24:
            return "Y"
        elif base in mito_accessions:
            return "MT"
        else:
            return None
    
    @staticmethod
    def normalise_chromosome(chromosome: str) -> str:
        if not chromosome:
            return chromosome
        return chromosome.replace("Chr", "").replace("chr", "")
    
    @staticmethod
    def get_lookup_values_from_spot_id(spot_id: str) -> Tuple[str | None, str | None, str | None]:
        regex = re.compile(r"^(chr[0-9XYM]+):\s*(\d+)-(\d+)$", re.I)
        match = regex.match(spot_id)
        if match is None:
            return None, None, None
        try:
            return CoordinateLookupGPLMapperUtils.normalise_chromosome(match.group(1)), match.group(2), match.group(3)
        except IndexError as e:
            return None, None, None
    
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
                    'clean_id': '',
                    'id_type': 'none',
                    'field_used': 'none',
                    'source_type': 'none',
                    'confidence': 0.0
                }
        self.mapping_stats["total_probes"] = len(probe_to_mapping_info)

    #{probe_id:[('chr', {"col_a":1}), ('range', {"col_b":2}), ('range', {"col_c":3})]}
    def map_chr_start_end_to_gene(self, rows:List[Dict[str, List[Tuple[str, Dict[str, str]]]]]) -> Dict[str, str]:
        mapped_values = {}
        probe_lookups = {}
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
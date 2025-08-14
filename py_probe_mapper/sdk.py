import fsspec
import zarr
import pandas as pd
import json
from typing import Dict, List
from py_probe_mapper.metadata_builder.build_metadata import GPLDatasetBuilder
from py_probe_mapper.lookup_classifier.optimised_lookup_classifier import process_large_gpl_inference
from py_probe_mapper.accession_lookup.accession_lookup import AccessionLookupGPLProcessor 
from py_probe_mapper.coordinate_lookup.coordinate_lookup import CoordinateLookupGPLProcessor

def fetch_gene_to_probe_mappings(gpl_id: str, return_dataframe: bool = False) -> Dict[str, str]:
    try:
        mapper = fsspec.get_mapper(f"https://huggingface.co/datasets/Tinfloz/probe-gene-map/resolve/main")
        root = zarr.open_group(mapper, mode='r')
        gpl = root['mappings'][gpl_id]
        probe_ids = [str(p) for p in gpl['probe_ids'][:]]
        gene_symbols = [str(g) for g in gpl['gene_symbols'][:]]

        df = pd.DataFrame({
            'probe_id': probe_ids,
            'gene_symbol': gene_symbols
        })
        df = df[df['gene_symbol'].str.strip() != '']

        if return_dataframe:
            return df.reset_index(drop=True)
        return {row['probe_id']:row['gene_symbol'] for _, row in df.iterrows()}
    except Exception as e:
        return {}

def runner(gpl_ids: List[str], api_url: str = None, api_key: str = None) -> None:
    to_be_mapped = []
    for i in gpl_ids:
        gene_mappings = fetch_gene_to_probe_mappings(i)
        if len(gene_mappings) == 0:
            to_be_mapped.append(i)
        else:
            with open(f"{i}_mappings.json", 'w') as f:
                json.dump(gene_mappings, f, indent=4)
    if len(to_be_mapped) == 0:
        return
    metadata_builder = GPLDatasetBuilder(max_workers=2)
    res = metadata_builder.build_dataset(to_be_mapped)
    lookup_input = [x for x in res['results'].values()]

    kwargs = {}
    if api_key:
        kwargs['api_key'] = api_key
    if api_url:
        kwargs['api_url'] = api_url
    lookup_res = process_large_gpl_inference(lookup_input, **kwargs)
    mapping_input = [x for x in lookup_res['results'].values()]
    accession_mappings = []
    coordinate_mappings = []
    for i in mapping_input:
        try:
            json_i = json.loads(i)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON string in lookup_res: {i[:100]}... Error: {str(e)}")
            continue
        if json_i['mapping_method'] == "accession_lookup":
            accession_mappings.append(json_i)
        elif json_i['mapping_method'] == "coordinate_lookup":
            coordinate_mappings.append(json_i)
        else:
            continue
    if len(accession_mappings) != 0:
        accession_processor = AccessionLookupGPLProcessor(gpl_records=accession_mappings, zarr_path="gpl_mappings.zarr")
        mappings = accession_processor.process_all_enhanced()
        for i in mappings.keys():
            with open(f"{i}_mappings.json", 'w') as f:
                json.dump(mappings[i], f, indent=4)
    if len(coordinate_mappings) != 0:
        print("Using coordinate lookup")
        coordinate_processor = CoordinateLookupGPLProcessor(gpl_records=coordinate_mappings, zarr_path="gpl_mappings.zarr")
        mappings = coordinate_processor.process_all_enhanced()
        for i in mappings.keys():
            with open(f"{i}_mappings.json", 'w') as f:
                json.dump(mappings[i], f, indent=4)
    return

if __name__ == '__main__':
    runner(["GPL3558"])
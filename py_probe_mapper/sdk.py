import fsspec
import zarr
import pandas as pd
from huggingface_hub import HfApi
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from py_probe_mapper.metadata_builder.build_metadata import GPLDatasetBuilder
from py_probe_mapper.lookup_classifier.optimised_lookup_classifier import process_large_gpl_inference
from py_probe_mapper.accession_lookup.accession_lookup import AccessionLookupGPLProcessor 
from py_probe_mapper.coordinate_lookup.coordinate_lookup import CoordinateLookupGPLProcessor
from py_probe_mapper.direct_lookup.direct_lookup import DirectLookupGPLProcessor


class GPLProbeMapper:
    """GPL probe-to-gene mapping SDK entry point."""
    
    def __init__(
        self,
        output_dir: Union[str, Path] = ".",
        zarr_path: str = "gpl_mappings.zarr",
        hf_repo: str = "Tinfloz/probe-gene-map",
        max_workers: int = 2,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize the GPL Probe Mapper.
        
        Args:
            output_dir: Directory to save mapping files
            zarr_path: Path for zarr dataset storage
            hf_repo: HuggingFace repository ID
            max_workers: Number of workers for parallel processing
            api_url: Default API URL for inference service
            api_key: Default API key for inference service
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.zarr_path = zarr_path
        self.hf_repo = hf_repo
        self.max_workers = max_workers
        self.api_url = api_url
        self.api_key = api_key
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # Initialize components
        self.hf_api = HfApi()
        
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup structured logging for the SDK."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler with structured format
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler for persistent logs
        log_file = self.output_dir / "gpl_mapper.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger

    def fetch_gene_to_probe_mappings(
        self, 
        gpl_id: str, 
        return_dataframe: bool = False
    ) -> Union[Dict[str, str], pd.DataFrame]:
        """
        Fetch existing probe-to-gene mappings for a GPL ID.
        
        Args:
            gpl_id: GPL platform identifier
            return_dataframe: If True, return DataFrame; otherwise return dict
            
        Returns:
            Dictionary or DataFrame of probe-to-gene mappings
        """
        self.logger.info(f"Fetching mappings for GPL ID: {gpl_id}")
        
        try:
            mapper = fsspec.get_mapper(
                f"https://huggingface.co/datasets/{self.hf_repo}/resolve/main"
            )
            root = zarr.open_group(mapper, mode='r')
            gpl = root['mappings'][gpl_id]
            
            probe_ids = [str(p) for p in gpl['probe_ids'][:]]
            gene_symbols = [str(g) for g in gpl['gene_symbols'][:]]
            
            self.logger.debug(f"Retrieved {len(probe_ids)} probe mappings for {gpl_id}")
            
            df = pd.DataFrame({
                'probe_id': probe_ids,
                'gene_symbol': gene_symbols
            })
            
            # Filter out empty gene symbols
            df = df[df['gene_symbol'].str.strip() != '']
            self.logger.info(f"Found {len(df)} valid mappings for {gpl_id}")
            
            if return_dataframe:
                return df.reset_index(drop=True)
            
            return {row['probe_id']: row['gene_symbol'] for _, row in df.iterrows()}
            
        except KeyError:
            self.logger.warning(f"No existing mappings found for GPL ID: {gpl_id}")
            return pd.DataFrame() if return_dataframe else {}
        except Exception as e:
            self.logger.error(f"Failed to fetch mappings for {gpl_id}: {str(e)}")
            return pd.DataFrame() if return_dataframe else {}

    def _save_mappings(self, gpl_id: str, mappings: Dict[str, str]) -> Path:
        """Save mappings to JSON file."""
        output_file = self.output_dir / f"{gpl_id}_mappings.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(mappings, f, indent=4)
            
            self.logger.info(f"Saved {len(mappings)} mappings to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to save mappings for {gpl_id}: {str(e)}")
            raise

    def _push_to_huggingface(self) -> bool:
        """Push Zarr dataset to HuggingFace Hub."""
        self.logger.info(f"Pushing dataset to HuggingFace: {self.hf_repo}")
        
        try:
            self.hf_api.upload_large_folder(
                folder_path=str(self.zarr_path),
                repo_id=self.hf_repo,
                repo_type="dataset"
            )
            
            url = f"https://huggingface.co/datasets/{self.hf_repo}"
            self.logger.info(f"Successfully pushed to: {url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to push to HuggingFace: {str(e)}")
            return False

    def _process_mappings_by_method(
        self, 
        mappings: List[Dict], 
        method: str, 
        processor_class
    ) -> Dict[str, Dict[str, str]]:
        """Process mappings using the specified method and processor."""
        if not mappings:
            return {}
            
        self.logger.info(f"Processing {len(mappings)} GPL records using {method}")
        
        try:
            processor = processor_class(
                gpl_records=mappings, 
                zarr_path=self.zarr_path
            )
            results = processor.process_all_enhanced()
            
            self.logger.info(f"Successfully processed {len(results)} GPL records")
            
            # Save individual mapping files
            for gpl_id, mapping_dict in results.items():
                self._save_mappings(gpl_id, mapping_dict)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process mappings with {method}: {str(e)}")
            return {}

    def map_gpl_probes(
        self,
        gpl_ids: List[str],
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        force_rebuild: bool = False
    ) -> Dict[str, Union[Dict[str, str], str]]:
        """
        Map probe IDs to gene symbols for given GPL platform IDs.
        
        Args:
            gpl_ids: List of GPL platform identifiers
            api_url: Optional API URL for inference service
            api_key: Optional API key for inference service
            force_rebuild: If True, rebuild mappings even if they exist
            
        Returns:
            Dictionary with GPL IDs as keys and mapping results or error messages as values
        """
        self.logger.info(f"Starting probe mapping for {len(gpl_ids)} GPL IDs: {gpl_ids}")
        
        results = {}
        to_be_mapped = []
        
        # Check existing mappings
        for gpl_id in gpl_ids:
            if not force_rebuild:
                existing_mappings = self.fetch_gene_to_probe_mappings(gpl_id)
                if existing_mappings:
                    self.logger.info(f"Using existing mappings for {gpl_id}")
                    self._save_mappings(gpl_id, existing_mappings)
                    results[gpl_id] = existing_mappings
                    continue
            
            to_be_mapped.append(gpl_id)
        
        if not to_be_mapped:
            self.logger.info("All GPL IDs have existing mappings")
            return results
        
        self.logger.info(f"Need to create mappings for {len(to_be_mapped)} GPL IDs")
        
        # Build metadata
        try:
            metadata_builder = GPLDatasetBuilder(max_workers=self.max_workers)
            metadata_results = metadata_builder.build_dataset(to_be_mapped)
            
            if not metadata_results or 'results' not in metadata_results:
                self.logger.error("Failed to build metadata")
                return results
                
            lookup_input = [x for x in metadata_results['results'].values()]
            self.logger.info(f"Built metadata for {len(lookup_input)} GPL records")
            
        except Exception as e:
            self.logger.error(f"Failed to build metadata: {str(e)}")
            return results
        
        # Perform inference
        try:
            # Use method parameters if provided, otherwise fall back to instance defaults
            final_api_url = api_url or self.api_url
            final_api_key = api_key or self.api_key
            
            kwargs = {}
            if final_api_key:
                kwargs['api_key'] = final_api_key
                self.logger.debug("Using API key for inference")
            if final_api_url:
                kwargs['api_url'] = final_api_url
                self.logger.info(f"Using API URL: {final_api_url}")
                
            lookup_results = process_large_gpl_inference(lookup_input, **kwargs)
            
            if not lookup_results or 'results' not in lookup_results:
                self.logger.error("Failed to perform GPL inference")
                return results
                
            mapping_input = [x for x in lookup_results['results'].values()]
            self.logger.info(f"Completed inference for {len(mapping_input)} GPL records")
            
        except Exception as e:
            self.logger.error(f"Failed to perform inference: {str(e)}")
            return results
        
        # Parse and categorize results
        accession_mappings = []
        coordinate_mappings = []
        direct_mappings = []
        
        for item in mapping_input:
            try:
                json_item = json.loads(item)
                method = json_item.get('mapping_method')
                
                if method == "accession_lookup":
                    accession_mappings.append(json_item)
                elif method == "coordinate_lookup":
                    coordinate_mappings.append(json_item)
                elif method == "direct":
                    direct_mappings.append(json_item)
                else:
                    self.logger.warning(f"Unknown mapping method: {method}")
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON: {str(e)}")
                continue
        
        # Process mappings by method
        all_processed = {}
        
        if accession_mappings:
            processed = self._process_mappings_by_method(
                accession_mappings, 
                "accession_lookup", 
                AccessionLookupGPLProcessor
            )
            all_processed.update(processed)
        
        if coordinate_mappings:
            processed = self._process_mappings_by_method(
                coordinate_mappings, 
                "coordinate_lookup", 
                CoordinateLookupGPLProcessor
            )
            all_processed.update(processed)
        
        if direct_mappings:
            processed = self._process_mappings_by_method(
                direct_mappings, 
                "direct_lookup", 
                DirectLookupGPLProcessor
            )
            all_processed.update(processed)
        
        # Update results
        results.update(all_processed)
        
        # Push to HuggingFace if we have new mappings
        if all_processed:
            self._push_to_huggingface()
        
        self.logger.info(f"Mapping completed. Results for {len(results)} GPL IDs")
        return results


# Convenience function for simple usage
def map_probes(
    gpl_ids: List[str],
    output_dir: str = ".",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    force_rebuild: bool = False,
    log_level: str = "INFO"
) -> Dict[str, Union[Dict[str, str], str]]:
    """
    Simple function to map GPL probe IDs to gene symbols.
    
    Args:
        gpl_ids: List of GPL platform identifiers
        output_dir: Directory to save mapping files
        api_url: API URL for inference service
        api_key: API key for inference service
        force_rebuild: If True, rebuild mappings even if they exist
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Dictionary with GPL IDs as keys and mapping results as values
        
    Example:
        >>> results = map_probes(["GPL570", "GPL96"])
        >>> print(f"Found {len(results['GPL570'])} mappings for GPL570")
        
        >>> # With custom API configuration
        >>> results = map_probes(
        ...     ["GPL570"], 
        ...     api_url="https://api.example.com/v1/inference",
        ...     api_key="your-api-key"
        ... )
    """
    mapper = GPLProbeMapper(
        output_dir=output_dir, 
        api_url=api_url,
        api_key=api_key,
        log_level=log_level
    )
    return mapper.map_gpl_probes(
        gpl_ids=gpl_ids,
        force_rebuild=force_rebuild
    )


if __name__ == "__main__":
    map_probes(
        gpl_ids=["GPL13534", "GPL6102"]    
    )

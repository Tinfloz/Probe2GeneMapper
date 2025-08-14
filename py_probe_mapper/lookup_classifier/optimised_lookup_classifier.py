import os
import json
import time
import threading
import logging
from typing import Dict, Optional, List, Any, NamedTuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from jinja2 import Environment, BaseLoader, select_autoescape
from dotenv import load_dotenv
import random
import math

load_dotenv()


# Configuration Management
@dataclass
class APIConfig:
    """API configuration with validation"""
    api_url: str
    api_key: str
    model: str = "gpt-4o"
    temperature: float = 0.2
    timeout: int = 60
    max_retries: int = 3
    
    def __post_init__(self):
        if not self.api_url or not self.api_key:
            raise ValueError("API URL and key are required")
        if not self.api_url.startswith(('http://', 'https://')):
            raise ValueError("API URL must be a valid HTTP/HTTPS URL")


@dataclass 
class ProcessingConfig:
    """Processing configuration optimized for large datasets"""
    max_workers: int = 4
    batch_size: int = 100
    progress_update_interval: float = 5.0
    sample_size: int = 400  # Optimal for 50k+ row datasets
    min_sample_size: int = 100
    quality_threshold: float = 0.7  # 70% population rate
    
    def __post_init__(self):
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.sample_size < 50:
            raise ValueError("sample_size must be >= 50")


# Data Models
class ProcessingResult(NamedTuple):
    """Result of processing a single GPL"""
    gpl_id: str
    success: bool
    result: Optional[str]
    error: Optional[str]
    processing_time: float
    retry_count: int = 0


class ProcessingStats(NamedTuple):
    """Processing statistics"""
    total_requested: int
    successful: int
    failed: int
    total_retries: int
    success_rate_percent: float
    total_processing_time_seconds: float
    average_time_per_gpl_seconds: float
    gpls_per_second: float
    max_workers_used: int
    processed_at: str


# Custom Exceptions
class GPLInferenceError(Exception):
    """Base exception for GPL inference errors"""
    pass


class APIError(GPLInferenceError):
    """API-related errors"""
    pass


class ConfigurationError(GPLInferenceError):
    """Configuration-related errors"""
    pass


# Logging Configuration
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("gpl_inference")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Progress Tracking
class InferenceProgressTracker:
    """Thread-safe progress tracker with comprehensive metrics"""

    def __init__(self, total_items: int, update_interval: float = 5.0):
        self.total = total_items
        self.completed = 0
        self.failed = 0
        self.retried = 0
        self.start_time = time.time()
        self.update_interval = update_interval
        self.lock = threading.Lock()
        self.last_update = 0
        self.logger = logging.getLogger("gpl_inference.progress")

    def update(self, result: ProcessingResult):
        """Update progress with processing result"""
        with self.lock:
            if result.success:
                self.completed += 1
            else:
                self.failed += 1
            
            self.retried += result.retry_count

            # Update progress based on interval
            now = time.time()
            if now - self.last_update >= self.update_interval:
                self._print_progress(result.gpl_id)
                self.last_update = now

    def _print_progress(self, current_gpl: str = ""):
        """Print current progress"""
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

        print(f"\r🤖 [{bar}] {percentage:5.1f}% | "
              f"✅ {self.completed} | ❌ {self.failed} | 🔄 {self.retried} | "
              f"🚀 {rate:.1f}/s | ⏱️ ETA: {eta_minutes:.1f}m | 🎯 {current_gpl}",
              end="", flush=True)

    def get_final_stats(self) -> ProcessingStats:
        """Get final processing statistics"""
        elapsed = time.time() - self.start_time
        elapsed = max(elapsed, 0.001)  # Prevent division by zero
        
        return ProcessingStats(
            total_requested=self.total,
            successful=self.completed,
            failed=self.failed,
            total_retries=self.retried,
            success_rate_percent=(self.completed / max(self.completed + self.failed, 1)) * 100,
            total_processing_time_seconds=elapsed,
            average_time_per_gpl_seconds=elapsed / max(self.total, 1),
            gpls_per_second=self.total / elapsed,
            max_workers_used=0,  # Will be set by caller
            processed_at=datetime.now().isoformat()
        )

    def print_final_report(self, max_workers: int = 0):
        """Print comprehensive final report"""
        stats = self.get_final_stats()
        stats = stats._replace(max_workers_used=max_workers)
        
        print(f"\n\n🎉 INFERENCE COMPLETE!")
        print(f"{'='*60}")
        print(f"⏰ Total time: {stats.total_processing_time_seconds:.1f}s "
              f"({stats.total_processing_time_seconds/60:.1f} minutes)")
        print(f"✅ Successful: {stats.successful}")
        print(f"❌ Failed: {stats.failed}")
        print(f"🔄 Total retries: {stats.total_retries}")
        print(f"📈 Success rate: {stats.success_rate_percent:.1f}%")
        print(f"🚀 Average rate: {stats.gpls_per_second:.1f} GPL/second")
        print(f"👥 Workers used: {max_workers}")
        print(f"{'='*60}")


# Optimized Data Quality Analyzer for Large Datasets
class DataQualityAnalyzer:
    """Optimized data quality analysis for large datasets (50k+ rows)"""
    
    EMPTY_PATTERNS = {
        "", "N/A", "NA", "n/a", "na", "NULL", "null", "Null",
        "---", "--", "-", ".", "..", "...", "???",
        "unknown", "Unknown", "UNKNOWN", "not available",
        "not applicable", "none", "None", "NONE", "0",
        "blank", "Blank", "BLANK", "empty", "Empty", "EMPTY",
        "missing", "Missing", "MISSING", "no data", "NO DATA", "[]"
    }
    
    def __init__(self, target_sample_size: int = 400):
        self.target_sample_size = target_sample_size
        self.logger = logging.getLogger("gpl_inference.quality")
    
    def optimize_sample(self, all_rows: List[Dict], columns: List[str]) -> List[Dict]:
        """Create optimized sample using random sampling"""
        if len(all_rows) <= self.target_sample_size:
            return all_rows
        
        # Use random sampling for large datasets
        return random.sample(all_rows, self.target_sample_size)
    
    def calculate_population_rates(self, sample_rows: List[Dict], columns: List[str]) -> Dict[str, float]:
        """Calculate population rate for each column"""
        if not sample_rows or not columns:
            return {col: 0.0 for col in columns}
        
        population_rates = {}
        for col in columns:
            empty_count = 0
            for row in sample_rows:
                value = str(row.get(col, "")).strip()
                if self._is_empty_value(value):
                    empty_count += 1
            population_rate = ((len(sample_rows) - empty_count) / len(sample_rows) * 100) if sample_rows else 0.0
            
            self.logger.debug(
                f"Column {col}, Population Rate: {population_rate}% ({len(sample_rows) - empty_count}/{len(sample_rows)}), "
                f"Empty Count: {empty_count}"
            )
            print(
                f"DEBUG: Column {col}, Population Rate: {population_rate}% ({len(sample_rows) - empty_count}/{len(sample_rows)}), "
                f"Empty Count: {empty_count}"
            )
            
            population_rates[col] = population_rate
        
        return population_rates
    
    def _is_empty_value(self, value: str) -> bool:
        """Check if value is empty or placeholder"""
        if not value or len(value) <= 2:
            return True
        return value.strip() in self.EMPTY_PATTERNS


# Optimized Prompt Builder
class OptimizedPromptBuilder:
    """Streamlined prompt builder optimized for large dataset inference"""
    
    PROMPT_TEMPLATE = """
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
- A combination of 3 columns:
    - **Chromosome** (indicators: 'chromosome', 'chr', 'seqname' in column name, case-insensitive)
    - **Start position** (indicators: 'start', 'start_range', 'begin', 'range_start' in column name)
    - **End position** (indicators: 'end', 'stop', 'end_range', 'range_end' in column name)
- Or a single column with chr:start-end format (indicators: 'spot_id', 'spot', 'coordinate', 'position' in column name)

### Priority 4: Gene Symbols (Least Reliable)
- **Gene symbols/names** only if no higher priority options exist
- These are prone to ambiguity and version changes

## Column Evaluation Process

For each potential mapping column, evaluate:

1. **Column name and description**: What type of identifier does it claim to contain?
2. **Actual values**: Do the values match the expected pattern?
3. **Population rate** (non-empty values) using the population rates provided below:
        {% for col, rate in population_rates.items() %}
        **{{ col }}**: {{ rate }}%
        {% endfor %}
4. **Value format consistency**: Are values in expected format?

## Mapping Method Classification

- **"direct"**: Platform provides direct gene symbols AND they are the best available option
- **"accession_lookup"**: Platform provides database accessions (RefSeq, GenBank, UniProt, Entrez)
- **"coordinate_lookup"**: Platform provides genomic coordinates
- **"unknown"**: No reliable mapping method identified

### Analysis Process:
1. Identify ALL potential mapping columns by examining column names, descriptions, and sample values
2. Rank identified columns by their population rates and priority system above - lay more emphasis on the population rates
3. For coordinates:
   - Identify columns matching chromosome, start, and end indicators (case-insensitive) and check their population rates
   - Or identify a single column with chr:start-end format (e.g., 'chr1:1000-2000')
4. Select highest priority column that meets quality requirements
5. If multiple columns qualify, prefer the column(s) with the highest population rate, especially if a lower-priority column has >=90% population rate and higher-priority columns have <50%
6. If no high-priority column meets requirements, select best available lower-priority option

## CRITICAL: Population Rate Override
If genomic coordinate column(s) have ≥90% population rate and accession fields have <90% population rate, 
ALWAYS choose coordinate_lookup regardless of priority ranking.

## Platform: {{ gpl_id }}
Organism: {{ organism }}
Sample Size: {{ sample_size }} rows (from {{ total_rows }} total)

### Column Descriptions:
{% for col, desc in column_descriptions.items() %}
**{{ col }}**: {{ desc }}
{% endfor %}

### Sample Data:
{{ table_header }}
{% for row in sample_rows %}
{{ row }}
{% endfor %}

## Required Output (JSON only):
```json
{
  "gpl_id": "{{ gpl_id }}",
  "organism": "{{ organism }}",
  "mapping_method": "direct|accession_lookup|coordinate_lookup|unknown",
  "description": "Selected [field_name] for [mapping_method] based on [population_rate]% population rate and [format_consistency]",
  "fields_used_for_mapping": ["primary_field_selected"],
  "alternate_fields_for_mapping": ["other_viable_fields"],
  "quality_assessment": {
    "selected_field": "field_name",
    "population_rate": "percentage",
    "format_type": "detected_format",
    "sample_size": {{ sample_size }},
    "confidence": "high"
  }
}
```

Return only valid JSON with your analysis. Do NOT include the placeholder template, code blocks, or any non-JSON content.
""".strip()

    def __init__(self):
        self.env = Environment(loader=BaseLoader(), autoescape=select_autoescape(['html', 'xml']))
    
    def build_prompt(self, gpl_record: Dict[str, Any], quality_analysis: Dict[str, float]) -> str:
        """Build optimized prompt for large dataset analysis"""
        gpl_id = gpl_record.get("gpl_id", "")
        metadata = gpl_record.get("metadata", {})
        table = gpl_record.get("table", {})
        columns = table.get("columns", [])
        sample_rows = table.get("sample_rows", [])
        
        context = {
            "gpl_id": gpl_id,
            "organism": metadata.get("organism", ""),
            "total_rows": gpl_record.get("total_rows", "50000+"),
            "sample_size": len(sample_rows),
            "column_descriptions": metadata.get("column_descriptions", {}),
            "population_rates":quality_analysis,
            "table_header": "\t".join(columns),
            "sample_rows": ["\t".join(str(row.get(col, "")) for col in columns) for row in sample_rows[:50]]
        }
        
        template = self.env.from_string(self.PROMPT_TEMPLATE)
        return template.render(**context)


# API Client (unchanged - already optimized)
class OpenAIClient:
    """Robust OpenAI API client with retry logic"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.logger = logging.getLogger("gpl_inference.api")
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.RequestException, APIError))
    )
    def _make_request(self, prompt: str) -> str:
        """Make API request with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature
        }

        try:
            response = requests.post(
                self.config.api_url, 
                headers=headers, 
                json=data, 
                timeout=self.config.timeout
            )

            if response.status_code == 429:
                retry_after = int(response.headers.get('retry-after', 60))
                raise APIError(f"Rate limited, retry after {retry_after}s")

            response.raise_for_status()
            result = response.json()

            if 'choices' not in result or not result['choices']:
                raise APIError("Invalid API response format")

            content = result["choices"][0]["message"]["content"].strip()
            
            if content.startswith("```json"):
                content = content[7:].strip()
            if content.endswith("```"):
                content = content[:-3].strip()
                
            return content

        except requests.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise APIError(f"API request failed: {e}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse API response: {e}")
            raise APIError(f"Invalid JSON response: {e}")

    def infer_mapping_strategy(self, prompt: str) -> str:
        """Get mapping strategy inference from API"""
        try:
            return self._make_request(prompt)
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise


# Main Optimized Processing Engine
class OptimizedGPLInferenceEngine:
    """Optimized processing engine for large datasets"""
    
    def __init__(self, api_config: APIConfig, processing_config: ProcessingConfig):
        self.api_config = api_config
        self.processing_config = processing_config
        self.quality_analyzer = DataQualityAnalyzer(processing_config.sample_size)
        self.prompt_builder = OptimizedPromptBuilder()
        self.api_client = OpenAIClient(api_config)
        self.logger = logging.getLogger("gpl_inference.optimized_engine")
        
    def _optimize_gpl_sample(self, gpl_record: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize sample size for better statistical confidence"""
        table = gpl_record.get("table", {})
        all_rows = table.get("sample_rows", [])
        columns = table.get("columns", [])
        
        if len(all_rows) > self.processing_config.sample_size:
            # Create optimized sample
            optimized_sample = self.quality_analyzer.optimize_sample(all_rows, columns)
            
            # Update the record with optimized sample
            optimized_record = gpl_record.copy()
            optimized_record["table"] = table.copy()
            optimized_record["table"]["sample_rows"] = optimized_sample
            
            self.logger.debug(f"Optimized sample for {gpl_record.get('gpl_id')}: "
                             f"{len(all_rows)} -> {len(optimized_sample)} rows")
            
            return optimized_record
        
        return gpl_record
    
    def _process_single_gpl(self, gpl_record: Dict[str, Any], progress: InferenceProgressTracker) -> ProcessingResult:
        """Process a single GPL with optimized sampling"""
        gpl_id = gpl_record.get("gpl_id", "unknown")
        start_time = time.time()
        
        try:
            # Optimize sample size
            optimized_record = self._optimize_gpl_sample(gpl_record)
            # Quick quality analysis
            columns = optimized_record.get("table", {}).get("columns", [])
            sample_rows = optimized_record.get("table", {}).get("sample_rows", [])
            quality_analysis = self.quality_analyzer.calculate_population_rates(sample_rows, columns)       
            # Build prompt
            prompt = self.prompt_builder.build_prompt(optimized_record, quality_analysis)
            
            # Get inference
            result = self.api_client.infer_mapping_strategy(prompt)
            
            # Validate JSON
            try:
                parsed_result = json.loads(result)
                # Add quality metrics
                parsed_result["sample_optimization"] = {
                    "original_sample_size": len(gpl_record.get("table", {}).get("sample_rows", [])),
                    "optimized_sample_size": len(sample_rows),
                    "total_platform_rows": gpl_record.get("total_rows", "unknown")
                }
                result = json.dumps(parsed_result)
            except json.JSONDecodeError:
                raise GPLInferenceError(f"Invalid JSON response for {gpl_id}")
            
            processing_time = time.time() - start_time
            return ProcessingResult(
                gpl_id=gpl_id,
                success=True,
                result=result,
                error=None,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            self.logger.error(f"Failed to process {gpl_id}: {error_msg}")
            
            return ProcessingResult(
                gpl_id=gpl_id,
                success=False,
                result=None,
                error=error_msg,
                processing_time=processing_time
            )

    def process_gpl_records(self, gpl_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process multiple GPL records"""
        if not gpl_records:
            raise ValueError("No GPL records provided")

        self.logger.info(f"Starting optimized processing of {len(gpl_records)} GPL records")
        
        print(f"🚀 GPL Inference Pipeline for Large GPL Datasets")
        print(f"{'='*65}")
        print(f"📋 Total GPLs: {len(gpl_records)}")
        print(f"👥 Workers: {self.processing_config.max_workers}")
        print(f"🎯 Target sample size: {self.processing_config.sample_size} rows")
        print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*65}\n")

        progress = InferenceProgressTracker(len(gpl_records), self.processing_config.progress_update_interval)
        results = {}
        errors = {}

        try:
            with ThreadPoolExecutor(max_workers=self.processing_config.max_workers) as executor:
                future_to_gpl = {
                    executor.submit(self._process_single_gpl, record, progress): record.get("gpl_id", "unknown")
                    for record in gpl_records
                }

                for future in as_completed(future_to_gpl):
                    gpl_id = future_to_gpl[future]
                    try:
                        result = future.result()
                        progress.update(result)
                        
                        if result.success:
                            results[gpl_id] = result.result
                        else:
                            errors[gpl_id] = result.error
                            
                    except Exception as e:
                        error_msg = f"Unexpected error: {e}"
                        self.logger.error(f"Unexpected error for {gpl_id}: {error_msg}")
                        errors[gpl_id] = error_msg
                        
                        failed_result = ProcessingResult(
                            gpl_id=gpl_id, success=False, result=None,
                            error=error_msg, processing_time=0
                        )
                        progress.update(failed_result)

        except KeyboardInterrupt:
            self.logger.warning("Process interrupted by user")
            print(f"\n\n⚠️ Process interrupted!")

        finally:
            progress.print_final_report(self.processing_config.max_workers)
            stats = progress.get_final_stats()
            stats = stats._replace(max_workers_used=self.processing_config.max_workers)
            self.logger.info(f"Optimized processing complete. Success rate: {stats.success_rate_percent:.1f}%")

        return {
            'results': results,
            'errors': errors,
            'statistics': stats._asdict()
        }


# Configuration Factory
class ConfigurationFactory:
    """Factory for creating optimized configurations"""
    
    @staticmethod
    def create_api_config(
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o"
    ) -> APIConfig:
        """Create API configuration"""
        api_url = api_url or os.getenv("GPT_API_URL")
        api_key = api_key or os.getenv("GPT_API_KEY")
        
        if not api_url or not api_key:
            raise ConfigurationError(
                "API URL and key must be provided either as parameters or environment variables "
                "(GPT_API_URL, GPT_API_KEY)"
            )
        
        return APIConfig(api_url=api_url, api_key=api_key, model=model)
    
    @staticmethod
    def create_processing_config(
        max_workers: int = 4,
        sample_size: int = 400
    ) -> ProcessingConfig:
        """Create optimized processing configuration for large datasets"""
        return ProcessingConfig(
            max_workers=max_workers,
            sample_size=sample_size,
            min_sample_size=100,
            quality_threshold=0.7
        )


# Main Interface Function
def process_large_gpl_inference(
    gpl_records: List[Dict[str, Any]],
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    max_workers: int = 4,
    sample_size: int = 400,
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Optimized GPL inference processing for large datasets (50k+ rows)
    
    Args:
        gpl_records: List of GPL record dictionaries
        api_url: OpenAI API URL (or set GPT_API_URL env var)
        api_key: OpenAI API key (or set GPT_API_KEY env var)
        model: Model to use for inference
        max_workers: Number of parallel workers
        sample_size: Optimal sample size for analysis (200 recommended)
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Dictionary with results, errors, and statistics
    """
    
    logger = setup_logging(log_level, log_file)
    
    try:
        # Create configurations optimized for large datasets
        kwargs = {}
        if api_key:
            kwargs['api_key'] = api_key
        if api_url:
            kwargs['api_url'] = api_url
        api_config = ConfigurationFactory.create_api_config(**kwargs)
        processing_config = ConfigurationFactory.create_processing_config(max_workers, sample_size)
        
        # Create and run optimized engine
        engine = OptimizedGPLInferenceEngine(api_config, processing_config)
        return engine.process_gpl_records(gpl_records)
        
    except Exception as e:
        logger.error(f"Optimized processing failed: {e}")
        raise
import os
import json
import time
import threading
import logging
from typing import Dict, Optional, List, Any, NamedTuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from jinja2 import Environment, BaseLoader, select_autoescape
from dotenv import load_dotenv

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
    """Processing configuration"""
    max_workers: int = 4
    batch_size: int = 100
    progress_update_interval: float = 5.0
    base_delay: float = 1.0
    max_delay: float = 60.0
    
    def __post_init__(self):
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


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


# Prompt Engineering
class PromptBuilder:
    """Template-based prompt builder with proper escaping"""
    
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
  "gpl_id": "{{ gpl_id }}",
  "organism": "{{ organism }}",
  "mapping_method": "<direct|accession_lookup|coordinate_lookup|unknown>",
  "description": "<explanation of mapping strategy and field chosen>",
  "fields_used_for_mapping": ["<list of column names used>"],
  "alternate_fields_for_mapping": ["<list of other viable mapping columns if available>"],
  "available_fields": {{ available_fields | tojson }},
  "mapping_priority_rationale": "<explanation of why this field was chosen>"
}
```

**Platform Data**:
- Platform ID: {{ gpl_id }}
- Metadata:
{% for key, value in metadata.items() %}
  - {{ key }}: {{ value }}
{% endfor %}

- Column Descriptions:
{% if column_descriptions %}
{% for key, value in column_descriptions.items() %}
  **{{ key }}**: {{ value }}
{% endfor %}
{% else %}
  No column descriptions available.
{% endif %}

- Sample Table:
{{ table_header }}
{% for row in sample_rows %}
{{ row }}
{% endfor %}

**Final Instruction**: Return only valid JSON with actual data for the GPL, based on the analysis. Do NOT include the placeholder template, code blocks, or any non-JSON content.
""".strip()

    def __init__(self):
        """Initialize prompt builder with Jinja2 environment"""
        self.env = Environment(
            loader=BaseLoader(),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
    def build_prompt(self, gpl_record: Dict[str, Any]) -> str:
        """Build prompt from GPL record using template"""
        try:
            # Extract and validate data
            gpl_id = gpl_record.get("gpl_id", "")
            metadata = gpl_record.get("metadata", {})
            table = gpl_record.get("table", {})
            columns = table.get("columns", [])
            sample_rows = table.get("sample_rows", [])
            column_descriptions = metadata.get("column_descriptions", {})
            
            # Build template context
            context = {
                "gpl_id": gpl_id,
                "organism": metadata.get("organism", ""),
                "metadata": {k: v for k, v in metadata.items() if k != "column_descriptions"},
                "column_descriptions": column_descriptions,
                "available_fields": columns,
                "table_header": "\t".join(columns),
                "sample_rows": []
            }
            
            # Format sample rows (limit to 10)
            for row in sample_rows[:10]:
                row_cells = [str(row.get(col, "")) for col in columns]
                context["sample_rows"].append("\t".join(row_cells))
            
            # Render template
            template = self.env.from_string(self.PROMPT_TEMPLATE)
            return template.render(**context)
            
        except Exception as e:
            raise GPLInferenceError(f"Failed to build prompt for {gpl_id}: {e}")


# API Client
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

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('retry-after', 60))
                raise APIError(f"Rate limited, retry after {retry_after}s")

            response.raise_for_status()
            result = response.json()

            if 'choices' not in result or not result['choices']:
                raise APIError("Invalid API response format")

            content = result["choices"][0]["message"]["content"].strip()
            
            # Clean JSON response
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


# Main Processing Engine
class GPLInferenceEngine:
    """Main processing engine for GPL inference"""
    
    def __init__(self, api_config: APIConfig, processing_config: ProcessingConfig):
        self.api_config = api_config
        self.processing_config = processing_config
        self.prompt_builder = PromptBuilder()
        self.api_client = OpenAIClient(api_config)
        self.logger = logging.getLogger("gpl_inference.engine")
        
    def _process_single_gpl(self, gpl_record: Dict[str, Any], progress: InferenceProgressTracker) -> ProcessingResult:
        """Process a single GPL record"""
        gpl_id = gpl_record.get("gpl_id", "unknown")
        start_time = time.time()
        retry_count = 0

        try:
            # Build prompt
            prompt = self.prompt_builder.build_prompt(gpl_record)
            
            # Get inference
            result = self.api_client.infer_mapping_strategy(prompt)
            
            # Validate result is valid JSON
            try:
                json.loads(result)
            except json.JSONDecodeError:
                raise GPLInferenceError(f"Invalid JSON response for {gpl_id}")
            
            processing_time = time.time() - start_time
            return ProcessingResult(
                gpl_id=gpl_id,
                success=True,
                result=result,
                error=None,
                processing_time=processing_time,
                retry_count=retry_count
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
                processing_time=processing_time,
                retry_count=retry_count
            )

    def process_gpl_records(self, gpl_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process multiple GPL records with parallel execution
        
        Args:
            gpl_records: List of GPL record dictionaries
            
        Returns:
            Dictionary with results, errors, and statistics
        """
        if not gpl_records:
            raise ValueError("No GPL records provided")

        self.logger.info(f"Starting processing of {len(gpl_records)} GPL records")
        
        print(f"🚀 Starting GPL Inference Pipeline")
        print(f"{'='*60}")
        print(f"📋 Total GPLs to process: {len(gpl_records)}")
        print(f"👥 Workers: {self.processing_config.max_workers}")
        print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # Initialize progress tracker
        progress = InferenceProgressTracker(
            len(gpl_records), 
            self.processing_config.progress_update_interval
        )
        
        results = {}
        errors = {}

        try:
            with ThreadPoolExecutor(max_workers=self.processing_config.max_workers) as executor:
                # Submit all tasks
                future_to_gpl = {
                    executor.submit(self._process_single_gpl, record, progress): record.get("gpl_id", "unknown")
                    for record in gpl_records
                }

                # Process completed tasks
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
                        
                        # Create failed result for progress tracking
                        failed_result = ProcessingResult(
                            gpl_id=gpl_id,
                            success=False,
                            result=None,
                            error=error_msg,
                            processing_time=0,
                            retry_count=0
                        )
                        progress.update(failed_result)

        except KeyboardInterrupt:
            self.logger.warning("Process interrupted by user")
            print(f"\n\n⚠️ Process interrupted by user!")

        finally:
            # Print final report
            progress.print_final_report(self.processing_config.max_workers)
            
            stats = progress.get_final_stats()
            stats = stats._replace(max_workers_used=self.processing_config.max_workers)
            
            self.logger.info(f"Processing complete. Success rate: {stats.success_rate_percent:.1f}%")

        return {
            'results': results,
            'errors': errors,
            'statistics': stats._asdict()
        }


# Configuration Factory
class ConfigurationFactory:
    """Factory for creating configurations from environment variables"""
    
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
        
        return APIConfig(
            api_url=api_url,
            api_key=api_key,
            model=model
        )
    
    @staticmethod
    def create_processing_config(
        max_workers: int = 4,
        batch_size: int = 100
    ) -> ProcessingConfig:
        """Create processing configuration"""
        return ProcessingConfig(
            max_workers=max_workers,
            batch_size=batch_size
        )


# Main Interface Function
def process_gpl_inference(
    gpl_records: List[Dict[str, Any]],
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    max_workers: int = 4,
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main interface function for GPL inference processing
    
    Args:
        gpl_records: List of GPL record dictionaries
        api_url: OpenAI API URL (or set GPT_API_URL env var)
        api_key: OpenAI API key (or set GPT_API_KEY env var)
        model: Model to use for inference
        max_workers: Number of parallel workers
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Dictionary with results, errors, and statistics
    """
    
    # Setup logging
    logger = setup_logging(log_level, log_file)
    
    try:
        # Create configurations
        kwargs = {}
        if api_key:
            kwargs['api_key'] = api_key
        if api_url:
            kwargs['api_url'] = api_url
        api_config = ConfigurationFactory.create_api_config(**kwargs)
        processing_config = ConfigurationFactory.create_processing_config(max_workers)
        
        # Create and run engine
        engine = GPLInferenceEngine(api_config, processing_config)
        return engine.process_gpl_records(gpl_records)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

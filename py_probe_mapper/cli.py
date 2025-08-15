from py_probe_mapper.sdk import map_probes
from typing import Dict, List, Union, Optional
import questionary
import re

def validate_gpl_ids(text):
    """Validate GPL ID input with proper format checking and count limits."""
    # Check if input is empty
    if not text.strip():
        return "❌ Please enter at least one GPL ID."
    
    # Split and clean the input
    gpl_ids = [id.strip().upper() for id in text.split(',') if id.strip()]
    
    # Check count limit
    if len(gpl_ids) > 5:
        return "❌ Maximum of 5 GPL IDs allowed."
    
    # Validate each GPL ID format (should start with GPL followed by numbers)
    gpl_pattern = re.compile(r'^GPL\d+$')
    invalid_ids = [gpl_id for gpl_id in gpl_ids if not gpl_pattern.match(gpl_id)]
    
    if invalid_ids:
        return f"❌ Invalid GPL ID format: {', '.join(invalid_ids)}. Use format like GPL570, GPL96."
    
    return True

def run_cli():
    """
    Interactive CLI tool for mapping GPL probe IDs to gene symbols.
    """
    print("\n🌟 Welcome to the GPL Probe Mapper CLI! 🌟\n")

    # Prompt for GPL IDs
    gpl_ids_str = questionary.text(
        "🧬 Enter up to 5 GPL platform identifiers (comma-separated, e.g., GPL570,GPL96):",
        validate=validate_gpl_ids
    ).ask()
    gpl_ids = [id.strip() for id in gpl_ids_str.split(',') if id.strip()]
    print(f"✅ Selected GPL IDs: {', '.join(gpl_ids)}")

    # Prompt for output directory
    output_dir = questionary.path(
        "📂 Enter output directory (default: .):",
        only_directories=True,
        default="."
    ).ask()
    print(f"✅ Output directory set to: {output_dir}")

    api_url = questionary.text(
        "🔗 Enter OpenAI API URL (e.g. https://api.openai.com/v1) for inference service (optional if running locally with a .env, press enter to skip):"
    ).ask()
    api_url = api_url if api_url.strip() else None
    print(f"{'✅ API URL set to: ' + api_url if api_url else 'ℹ️ No API URL provided'}")

    # Prompt for API Key (optional)
    api_key = questionary.password(
        "🔑 Enter OpenAI API key for inference service (optional if running locally with a .env, press enter to skip):"
    ).ask()
    api_key = api_key if api_key.strip() else None
    print(f"{'✅ API key provided' if api_key else 'ℹ️ No API key provided'}")

    # Prompt for force rebuild
    force_rebuild = questionary.confirm(
        "🔄 Force rebuild mappings even if they exist? (default: No)",
        default=False
    ).ask()
    print(f"✅ Force rebuild: {'Yes' if force_rebuild else 'No'}")

    # Prompt for log level
    log_level = questionary.select(
        "📋 Select logging level:",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO"
    ).ask()
    print(f"✅ Logging level set to: {log_level}")

    # Call the map_probes function
    print("\n🚀 Starting probe mapping... Please wait! ⏳")
    results: Dict[str, Union[Dict[str, str], str]] = map_probes(
        gpl_ids=gpl_ids,
        output_dir=output_dir,
        api_url=api_url,
        api_key=api_key,
        force_rebuild=force_rebuild,
        log_level=log_level
    )

    # Display summary
    print("\n🎉 Mapping completed! 🎉")
    print("\n📊 Results:")
    for gpl_id, result in results.items():
        if isinstance(result, dict):
            print(f"✅ {gpl_id}: Found {len(result)} mappings 🧬")
        else:
            print(f"❌ {gpl_id}: {result}")

if __name__ == "__main__":
    run_cli()
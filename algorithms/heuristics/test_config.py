#!/usr/bin/env python3
"""Test configuration loading."""

from pathlib import Path

from config import load_config


def test_config_loading():
    """Test loading the configuration."""
    print("Testing configuration loading...")
    print(f"Current directory: {Path.cwd()}")
    print(f"Script location: {Path(__file__).parent}")

    try:
        config = load_config()
        print("✓ Configuration loaded successfully!")
        print(f"Service name: {config.service.name}")
        print(f"Service port: {config.service.port}")
        print(f"Unsafe threshold: {config.rules_engine.unsafe_threshold}")
        print(f"Number of rules: {len(config.rules_engine.rules)}")

        # Print first rule as example
        if config.rules_engine.rules:
            first_rule = config.rules_engine.rules[0]
            print("\nFirst rule:")
            print(f"  Name: {first_rule.name}")
            print(f"  Type: {first_rule.rule_type}")
            print(f"  Weight: {first_rule.weight}")
            print(f"  Patterns: {len(first_rule.patterns)} patterns")

    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config_loading()


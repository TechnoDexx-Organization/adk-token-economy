from harness.modules.retry import build_retry_config


def test_returns_only_litellm_known_keys():
    """P1-12: в kwargs не должно быть ключей, которые LiteLLM не понимает."""
    config = build_retry_config()
    
    assert set(config.keys()) == {"num_retries", "timeout"}
    assert "retry_policy" not in config
    assert "max_retries" not in config


def test_values_are_passed_through():
    config = build_retry_config(max_retries=5, timeout_seconds=30)
    
    assert config["num_retries"] == 5
    assert config["timeout"] == 30
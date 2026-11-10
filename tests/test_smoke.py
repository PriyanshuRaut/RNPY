def test_import():
    import npguard
    assert callable(npguard.memory_watcher)

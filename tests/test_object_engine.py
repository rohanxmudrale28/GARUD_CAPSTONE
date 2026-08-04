from engines.object_engine import ObjectEngine

def test_object_engine_initializes():
    engine = ObjectEngine()
    assert engine.model is not None
    assert engine.device in ("cpu", "mps", "cuda")

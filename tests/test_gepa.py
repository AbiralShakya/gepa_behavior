from gepa.gepa import GEPAOptimizer, MockLLM, Prompt


def test_gepa_optimize_mock():
    opt = GEPAOptimizer(llm=MockLLM(), population_size=4, pareto_front_k=2, mutation_rate=1.0)
    calls = {"n": 0}

    def evaluate(p: Prompt):
        calls["n"] += 1
        return {"reward": float(len(p.text))}

    opt.evaluate_fn = evaluate
    final = opt.optimize("base prompt", iterations=2)
    assert len(final) <= 2
    assert calls["n"] > 0

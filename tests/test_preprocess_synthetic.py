import preprocess


def test_generate_synthetic_data_has_required_columns():
    df = preprocess._generate_synthetic_data(num_units=3, min_cycles=60, max_cycles=80)
    expected = {"unit_id", "cycle", "temperature", "voltage", "vibration", "pressure", "fuel_flow", "rpm", "rul"}

    assert expected.issubset(set(df.columns))
    assert len(df) > 0

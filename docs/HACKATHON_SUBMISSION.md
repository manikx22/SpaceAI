# Hackathon Submission Checklist

## Included

- End-to-end data pipeline
- Anomaly detection model
- Failure prediction model
- Recommendation engine
- Interactive Streamlit dashboard
- Live tracking visualization
- Documentation and CI workflow

## Demo Steps

```bash
python -m pip install -r requirements.txt
python train_anomaly.py
python train_lstm.py
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

## Judging Notes

- Synthetic fallback ensures reproducible demo without external dataset dependency.
- Dashboard combines prediction + explainability + operational recommendation flow.

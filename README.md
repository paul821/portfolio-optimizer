# Portfolio Optimizer (Streamlit)

One-file Streamlit app for:
- One-Fund (single risky + rf)
- Global Minimum-Variance (GMV)
- Tangency / Max-Sharpe
- Efficient Frontier (unconstrained + long-only)

## Local dev

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

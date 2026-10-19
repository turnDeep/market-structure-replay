from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_entrypoint_and_switching():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=45).run()
    assert not app.exception
    assert "v0.12" in app.caption[0].value
    assert app.radio[0].value == "nq"
    assert app.query_params["market"] == "nq"
    app.radio[0].set_value("es").run()
    assert not app.exception
    assert app.radio[0].value == "es"
    assert "ES=F" in app.caption[1].value
    app.radio[0].set_value("nq").run()
    assert not app.exception
    assert "NQ=F" in app.caption[1].value


def test_market_deep_link_and_invalid_input():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=45)
    app.query_params["market"] = "es"
    app.run()
    assert not app.exception
    assert app.radio[0].value == "es"
    other = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=45)
    other.query_params["market"] = "../../untrusted.html"
    other.run()
    assert not other.exception
    assert other.radio[0].value == "nq"

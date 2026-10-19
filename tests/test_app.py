"""
Automated tests for the E-Commerce Return Prediction app.
Run from the project folder:   python tests/test_app.py
(On a server with no display use:  xvfb-run -a python tests/test_app.py)
Results are printed and saved to tests/test_results.txt
"""
import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location("app_module", os.path.join(ROOT, "app.py"))
appm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(appm)
import pandas as pd

# --- capture message boxes instead of showing them (so tests never block) ---
errors, infos = [], []
appm.messagebox.showerror = lambda title, msg, **k: errors.append((title, msg))
appm.messagebox.showinfo = lambda title, msg, **k: infos.append((title, msg))

app = appm.App()
results = []


def test(name):
    """Decorator: run the function now, record PASS / FAIL."""
    def deco(fn):
        try:
            detail = fn() or ""
            results.append((name, "PASS", detail))
        except Exception as e:                      # noqa
            results.append((name, "FAIL", repr(e)))
        return fn
    return deco


def setv(**kw):
    for k, v in kw.items():
        app.v[k].set(str(v))


def predict_with(**kw):
    app.go("predict")
    appm.App.clear(app)
    setv(**kw)
    errors.clear()
    app.predict()
    return app.last


# ---------------------------------------------------------------- data/model
@test("T01 Model + schema files load")
def _():
    assert appm.MODEL is not None and appm.SCHEMA["feature_order"]


@test("T02 feature_order matches the model's training columns")
def _():
    assert list(appm.MODEL.feature_names_in_) == appm.FEATURE_ORDER
    return f"{len(appm.FEATURE_ORDER)} columns, same order"


@test("T03 model_comparison.csv: 5 models, best = schema best_model")
def _():
    df = appm.COMPARISON
    assert len(df) == 5
    assert df.iloc[0]["Model"] == appm.BEST_MODEL_NAME
    return f"best = {df.iloc[0]['Model']} (ROC-AUC {df.iloc[0]['ROC-AUC']})"


# ---------------------------------------------------------------- GUI basics
@test("T04 Window title and not resizable")
def _():
    assert app.title() == "E-Commerce Return Prediction"
    assert app.resizable() == (False, False)


@test("T05 Default inputs come from schema medians")
def _():
    d = appm.default_inputs()
    assert d["price"] == "254" and d["qty"] == "3" and d["age"] == "44"
    assert d["month"].startswith("6 ") and d["day"].startswith("3 ") and d["hour"] == "12:00"
    return "price 254, qty 3, age 44, June, Thursday, 12:00"


@test("T06 All four pages render without errors")
def _():
    for p in ("dashboard", "predict", "compare", "history"):
        app.go(p)
        app.update()
    return "dashboard, predict, compare, history"


# ---------------------------------------------------------------- auto-calc
@test("T07 Auto-calculated Total = Price x Quantity")
def _():
    setv(price=254, qty=5)
    assert "Total: 1,270.00" in app.auto_var.get(), app.auto_var.get()
    assert "Price/Unit: 254.00" in app.auto_var.get()


@test("T08 Is_Weekend is Yes only for Saturday/Sunday (days 5, 6)")
def _():
    for i, name in enumerate(appm.DAYS):
        setv(day=name)
        expected = "Yes" if i in (5, 6) else "No"
        assert f"Weekend: {expected}" in app.auto_var.get(), (name, app.auto_var.get())


# ---------------------------------------------------------------- validation
BAD = [
    ("price", "abc", "text price"), ("price", "0", "zero price"),
    ("price", "-5", "negative price"), ("price", "", "empty price"),
    ("qty", "0", "quantity 0"), ("qty", "-1", "negative quantity"),
    ("qty", "2.5", "decimal quantity"), ("qty", "abc", "text quantity"),
    ("age", "17", "age 17 (<18)"), ("age", "101", "age 101 (>100)"),
    ("age", "abc", "text age"), ("age", "", "empty age"), ("age", "30.5", "decimal age"),
]


@test(f"T09 Validation rejects {len(BAD)} bad inputs with an error popup")
def _():
    app.go("predict")
    for key, val, label in BAD:
        appm.App.clear(app)
        setv(**{key: val})
        errors.clear()
        out = app.read_inputs()
        assert out is None and len(errors) == 1, f"not rejected: {label}"
    return ", ".join(l for _, _, l in BAD)


@test("T10 Validation accepts boundary values (age 18 & 100, qty 1, price 0.01)")
def _():
    for age in (18, 100):
        appm.App.clear(app)
        setv(age=age, qty=1, price="0.01")
        errors.clear()
        assert app.read_inputs() is not None and not errors


@test("T11 Missing dropdown selection shows an error")
def _():
    appm.App.clear(app)
    setv(cat="")
    errors.clear()
    assert app.read_inputs() is None and len(errors) == 1


# ---------------------------------------------------------------- prediction
@test("T12 Default input gives a probability between 0 and 1")
def _():
    p = predict_with()
    assert 0 <= p <= 1
    return f"{p * 100:.1f}%"


@test("T13 App probability equals a direct model.predict_proba call")
def _():
    p = predict_with(cat="Electronics", qty=5, pay="Crypto", age=25, gender="Male",
                     day="5 - Saturday", hour="23:00", month="1 - January")
    row = {"Product Category": "Electronics", "Product Price": 254.0, "Quantity": 5,
           "Total Purchase Amount": 1270.0, "Payment Method": "Crypto", "Customer Age": 25,
           "Gender": "Male", "Purchase_Month": 1, "Purchase_DayOfWeek": 5,
           "Purchase_Hour": 23, "Is_Weekend": 1, "Price_per_Unit": 254.0}
    direct = float(appm.MODEL.predict_proba(pd.DataFrame([row])[appm.FEATURE_ORDER])[0, 1])
    assert abs(p - direct) < 1e-12
    return f"app {p:.6f} == model {direct:.6f}"


@test("T14 Probability >= 50% -> 'Likely to be RETURNED' in red")
def _():
    p = predict_with(cat="Electronics", qty=5)
    assert p >= 0.5
    assert app.result_lbl.cget("text") == "Likely to be RETURNED"
    assert app.result_lbl.cget("fg") == appm.T["red"]
    return f"{p * 100:.1f}% -> red"


@test("T15 Probability < 50% -> 'Unlikely to be returned' in green")
def _():
    p = predict_with(cat="Books", qty=1, age=60)
    assert p < 0.5
    assert app.result_lbl.cget("text") == "Unlikely to be returned"
    assert app.result_lbl.cget("fg") == appm.T["green"]
    return f"{p * 100:.1f}% -> green"


@test("T16 Progress bar shows the probability")
def _():
    p = predict_with(cat="Electronics", qty=5)
    assert abs(float(app.bar.cget("value")) - p * 100) < 1e-6


@test("T17 Sweep every category x payment x gender x weekday x quantity (128 runs)")
def _():
    probs = []
    app.history.clear()
    for cat in appm.CATEGORICAL["Product Category"]:
        for pay in appm.CATEGORICAL["Payment Method"]:
            for g in appm.CATEGORICAL["Gender"]:
                for day in (appm.DAYS[0], appm.DAYS[5]):
                    for qty in (1, 5):
                        p = predict_with(cat=cat, pay=pay, gender=g, day=day, qty=qty)
                        assert not errors and 0 <= p <= 1
                        probs.append(p)
    app.history.clear()
    return f"{len(probs)} runs, no errors, range {min(probs) * 100:.1f}%-{max(probs) * 100:.1f}%"


# ---------------------------------------------------------------- buttons/state
@test("T18 Clear button restores default values and resets the result")
def _():
    predict_with(cat="Electronics", qty=5)
    app.clear()
    assert {k: v.get() for k, v in app.v.items()} == appm.default_inputs()
    assert app.last is None and float(app.bar.cget("value")) == 0


@test("T19 Dark/Light toggle switches palette and keeps data")
def _():
    app.history.clear()
    predict_with(price=123, qty=2)
    n = len(app.history)
    dark_bg = appm.T["bg"]
    app.toggle_theme()
    assert appm.T["bg"] == appm.LIGHT["bg"] != dark_bg
    assert app.v["price"].get() == "123" and len(app.history) == n
    app.toggle_theme()
    assert appm.T["bg"] == dark_bg


@test("T20 History records predictions; Export CSV writes a valid file")
def _():
    app.history.clear()
    predict_with(cat="Books")
    predict_with(cat="Home")
    assert len(app.history) == 2
    out = os.path.join(ROOT, "prediction_history.csv")
    app.export_history()
    df = pd.read_csv(out)
    os.remove(out)
    assert len(df) == 2 and "Return Probability" in df.columns
    app.history.clear()


app.destroy()

# ---------------------------------------------------------------- report
width = max(len(n) for n, _, _ in results)
lines = [f"{'PASS' if s == 'PASS' else 'FAIL'}  {n:<{width}}  {d}" for n, s, d in results]
passed = sum(1 for _, s, _ in results if s == "PASS")
lines.append("-" * 60)
lines.append(f"{passed}/{len(results)} tests passed")
text = "\n".join(lines)
print(text)
with open(os.path.join(HERE, "test_results.txt"), "w", encoding="utf-8") as f:
    f.write(text + "\n")
sys.exit(0 if passed == len(results) else 1)

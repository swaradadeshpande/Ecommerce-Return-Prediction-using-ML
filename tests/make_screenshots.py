"""Regenerates the README screenshots into ./screenshots (needs Pillow: pip install pillow)."""
import importlib.util, os, time
from PIL import ImageGrab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "screenshots")
os.makedirs(OUT, exist_ok=True)
spec = importlib.util.spec_from_file_location("app_module", os.path.join(ROOT, "app.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a = m.App()


def settle(sec=0.9):
    end = time.time() + sec
    while time.time() < end:
        a.update(); time.sleep(0.02)


def snap(name):
    settle(0.3)
    x, y, w, h = a.winfo_rootx(), a.winfo_rooty(), a.winfo_width(), a.winfo_height()
    ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(os.path.join(OUT, name + ".png"))


def run(**kw):
    a.go("predict")
    for k, v in kw.items():
        a.v[k].set(v)
    a.predict(); settle()


# two predictions so the dashboard / history have content
run(cat="Electronics", qty="5", pay="Crypto", age="25", day="5 - Saturday")
run(cat="Books", qty="1", age="60")
snap("02_predict_dark")
a.go("dashboard"); snap("01_dashboard_dark")
a.go("compare"); snap("05_compare_dark")
a.go("history"); snap("07_history_dark")
a.toggle_theme()
a.go("dashboard"); snap("04_dashboard_light")
a.go("predict"); a.show_result(a.last, animate=False); snap("03_predict_light")
a.go("compare"); snap("06_compare_light")
a.go("history"); snap("08_history_light")
a.destroy()
print("screenshots saved to", OUT)

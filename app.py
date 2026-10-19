"""
E-Commerce Return Prediction  -  Dashboard App (Tkinter)
=========================================================
Files needed in the SAME folder as this script:
    return_model.joblib    -> trained scikit-learn Pipeline (preprocessing + model)
    gui_schema.json        -> dropdown options, numeric stats and feature_order
    model_comparison.csv   -> final model comparison table from the notebook

Run:   python app.py
Needs: pandas, scikit-learn, joblib   (tkinter ships with Python)

APP STRUCTURE (like a small website):
    Sidebar  -> navigation + dark/light switch + Exit
    Pages    -> Dashboard | Predict | Model Comparison | History
"""

import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import pandas as pd

# ----------------------------------------------------------------------------
# 1. LOAD MODEL, SCHEMA AND COMPARISON TABLE
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # folder that holds app.py


def path(name):
    """Full path of a file that sits next to app.py."""
    return os.path.join(BASE_DIR, name)


MODEL = joblib.load(path("return_model.joblib"))          # sklearn Pipeline
with open(path("gui_schema.json"), encoding="utf-8") as f:
    SCHEMA = json.load(f)

FEATURE_ORDER = SCHEMA["feature_order"]     # exact column order the model expects
CATEGORICAL = SCHEMA["categorical"]         # dropdown options for text columns
NUMERIC = SCHEMA["numeric"]                 # min / max / median of numeric columns
BEST_MODEL_NAME = SCHEMA.get("best_model", "ML Pipeline")

try:
    COMPARISON = pd.read_csv(path("model_comparison.csv"))
    COMPARISON = COMPARISON.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
except FileNotFoundError:
    COMPARISON = None

# ----------------------------------------------------------------------------
# 2. THEMES - one palette per mode. T always points at the ACTIVE palette,
#    so switching theme = copy another palette into T and redraw the UI.
# ----------------------------------------------------------------------------
DARK = dict(bg="#0B1220", side="#0F1830", card="#131C2F", card2="#1A2540",
            border="#26345A", text="#E8EEFF", muted="#8B98BA", accent="#7C83FF",
            accent_hover="#6A71F5", red="#FF5C7A", green="#2DDC9A",
            best_bg="#12281F")
LIGHT = dict(bg="#F1F4FB", side="#FFFFFF", card="#FFFFFF", card2="#EEF1FB",
             border="#D9DFF2", text="#16203A", muted="#68758F", accent="#5B63F0",
             accent_hover="#4A52DD", red="#E5395B", green="#10A06D",
             best_bg="#E3F7EE")
T = dict(DARK)

FONT = "Segoe UI"     # Tk falls back to a similar font on Mac / Linux

MONTHS = ["1 - January", "2 - February", "3 - March", "4 - April", "5 - May",
          "6 - June", "7 - July", "8 - August", "9 - September",
          "10 - October", "11 - November", "12 - December"]
DAYS = ["0 - Monday", "1 - Tuesday", "2 - Wednesday", "3 - Thursday",
        "4 - Friday", "5 - Saturday", "6 - Sunday"]
HOURS = [f"{h:02d}:00" for h in range(24)]


def default_inputs():
    """Starting values for the form: medians from gui_schema.json (a typical order)."""
    return {
        "cat": CATEGORICAL["Product Category"][0],
        "pay": CATEGORICAL["Payment Method"][0],
        "gender": CATEGORICAL["Gender"][0],
        "price": f"{NUMERIC['Product Price']['median']:g}",
        "qty": f"{NUMERIC['Quantity']['median']:g}",
        "age": f"{NUMERIC['Customer Age']['median']:g}",
        "month": MONTHS[int(NUMERIC["Purchase_Month"]["median"]) - 1],
        "day": DAYS[int(NUMERIC["Purchase_DayOfWeek"]["median"])],
        "hour": HOURS[int(NUMERIC["Purchase_Hour"]["median"])],
    }


def get_factors(top=8):
    """Top model factors = largest logistic-regression coefficients.
    Numeric columns are standardised, so coefficients are comparable.
    Returns [(label, coef)] or None if the model has no coef_."""
    try:
        names = MODEL.named_steps["prep"].get_feature_names_out()
        coefs = MODEL.named_steps["model"].coef_[0]
    except Exception:
        return None
    rows = []
    for n, c in zip(names, coefs):
        if n.startswith("cat__"):
            col, val = n[5:].rsplit("_", 1)
            label = f"{col.replace('Product ', '').replace(' Method', '')}: {val}"
        else:
            label = n[5:].replace("_", " ")
        rows.append((label, float(c)))
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    return rows[:top]


# ----------------------------------------------------------------------------
# 3. CUSTOM WIDGETS (drawn on Canvas)
# ----------------------------------------------------------------------------
class Gauge(tk.Canvas):
    """Half-circle probability gauge. show() grows the arc smoothly."""

    def __init__(self, master, bg, size=300):
        super().__init__(master, width=size, height=int(size * 0.66), bg=bg,
                         highlightthickness=0)
        self.size, self.color = size, T["muted"]

    def draw(self, p, big, small):
        self.delete("all")
        s, pad = self.size, 24
        box = (pad, pad, s - pad, s - pad)
        self.create_arc(*box, start=180, extent=-180, style="arc", width=22,
                        outline=T["card2"])
        if p > 0:
            self.create_arc(*box, start=180, extent=-180 * p, style="arc",
                            width=22, outline=self.color)
        cx, cy, r = s / 2, s / 2, (s - 2 * pad) / 2
        self.create_line(cx, cy - r - 16, cx, cy - r + 16, fill=T["muted"], width=2)
        self.create_text(cx, 8, text="50% threshold", fill=T["muted"], font=(FONT, 8))
        self.create_text(cx, cy - 18, text=big, fill=T["text"], font=(FONT, 34, "bold"))
        self.create_text(cx, cy + 20, text=small, fill=T["muted"], font=(FONT, 10))
        self.create_text(pad + 2, cy + 36, text="0%", fill=T["muted"], font=(FONT, 8))
        self.create_text(s - pad - 6, cy + 36, text="100%", fill=T["muted"], font=(FONT, 8))

    def idle(self):
        self.color = T["muted"]
        self.draw(0, "--", "waiting for input")

    def show(self, p, color, animate=True, steps=24):
        self.color = color

        def step(i=0):
            if not self.winfo_exists():          # page may have been rebuilt
                return
            q = p * (i / steps)
            self.draw(q, f"{q * 100:.1f}%", "return probability")
            if i < steps:
                self.after(14, step, i + 1)
        step(0 if animate else steps)


class Switch(tk.Canvas):
    """Pill-shaped on/off switch used for the dark / light toggle."""

    def __init__(self, master, on, command, bg):
        super().__init__(master, width=48, height=26, bg=bg, highlightthickness=0,
                         cursor="hand2")
        col = T["accent"] if on else T["border"]
        self.create_oval(0, 0, 26, 26, fill=col, outline=col)
        self.create_oval(22, 0, 48, 26, fill=col, outline=col)
        self.create_rectangle(13, 0, 35, 26, fill=col, outline=col)
        x = 25 if on else 3
        self.create_oval(x, 3, x + 20, 23, fill="#FFFFFF", outline="")
        self.bind("<Button-1>", lambda e: command())


def hbars(cv, items, label_w=150, fmt="{:.4f}", row_h=40):
    """Horizontal bar chart on a canvas. items = [(label, value)], best = first."""
    cv.delete("all")
    w = int(cv["width"])
    bar_w = w - label_w - 70
    vmax = max(v for _, v in items) * 1.12
    for i, (label, v) in enumerate(items):
        y = 4 + i * row_h
        col = T["green"] if i == 0 else T["accent"]
        cv.create_text(label_w - 12, y + 15, text=label, anchor="e", fill=T["text"],
                       font=(FONT, 10))
        cv.create_rectangle(label_w, y + 3, label_w + bar_w, y + 27, fill=T["card2"], outline="")
        cv.create_rectangle(label_w, y + 3, label_w + bar_w * v / vmax, y + 27, fill=col, outline="")
        cv.create_text(label_w + bar_w * v / vmax + 8, y + 15, text=fmt.format(v),
                       anchor="w", fill=col, font=(FONT, 10, "bold"))


def diverging(cv, items, label_w=150, row_h=26):
    """Bars that go right (raises return risk) or left (lowers it) from a centre line."""
    cv.delete("all")
    w = int(cv["width"])
    x0, x1 = label_w + 10, w - 46
    cx = (x0 + x1) / 2
    m = max(abs(v) for _, v in items)
    cv.create_line(cx, 0, cx, len(items) * row_h + 6, fill=T["border"])
    for i, (label, v) in enumerate(items):
        y = 4 + i * row_h
        col = T["red"] if v > 0 else T["green"]
        end = cx + (x1 - cx) * v / m
        cv.create_text(label_w, y + 11, text=label, anchor="e", fill=T["text"], font=(FONT, 9))
        cv.create_rectangle(min(cx, end), y + 3, max(cx, end), y + 19, fill=col, outline="")
        cv.create_text(end + (6 if v > 0 else -6), y + 11, text=f"{v:+.2f}",
                       anchor="w" if v > 0 else "e", fill=col, font=(FONT, 8, "bold"))


# ----------------------------------------------------------------------------
# 4. MAIN APPLICATION
# ----------------------------------------------------------------------------
class App(tk.Tk):
    PAGES = [("dashboard", "🏠", "Dashboard"), ("predict", "🔮", "Predict"),
             ("compare", "📊", "Model Comparison"), ("history", "🕑", "History")]
    SUBTITLES = {
        "dashboard": "Overview of the model and your recent predictions",
        "predict": "Enter the order details to estimate the chance of a return",
        "compare": "How the candidate models performed on the held-out test set",
        "history": "Predictions made in this session",
    }

    def __init__(self):
        super().__init__()
        self.title("E-Commerce Return Prediction")
        self.geometry("1200x780")
        self.resizable(False, False)                 # not resizable (requirement)

        self.dark = True
        self.page = "dashboard"
        self.history = []                            # list of dicts, newest last
        self.last = None                             # last predicted probability

        # Tk variables hold the form values; they survive page + theme changes
        self.v = {k: tk.StringVar(value=val) for k, val in default_inputs().items()}
        self.auto_var = tk.StringVar()
        for k in ("price", "qty", "day"):            # live "auto-calculated" preview
            self.v[k].trace_add("write", lambda *a: self.update_auto())
        self.update_auto()

        self.bind("<Return>", lambda e: self.predict() if self.page == "predict" else None)
        self.render()
        self.center()

    # ---------------- theme + navigation ----------------
    def toggle_theme(self):
        self.dark = not self.dark
        T.clear()
        T.update(DARK if self.dark else LIGHT)
        self.render()

    def go(self, page):
        self.page = page
        self.render()

    def center(self):
        self.update_idletasks()
        x = (self.winfo_screenwidth() - self.winfo_width()) // 2
        y = max(0, (self.winfo_screenheight() - self.winfo_height()) // 2 - 30)
        self.geometry(f"+{x}+{y}")

    def render(self):
        """(Re)draw the whole window: styles -> sidebar -> current page."""
        for w in self.winfo_children():
            w.destroy()
        self.configure(bg=T["bg"])
        self.setup_styles()
        self.build_sidebar()

        main = tk.Frame(self, bg=T["bg"])
        main.pack(side="left", fill="both", expand=True, padx=28, pady=(22, 20))

        head = tk.Frame(main, bg=T["bg"])
        head.pack(fill="x", pady=(0, 16))
        names = {k: t for k, _, t in self.PAGES}
        tk.Label(head, text=names[self.page], bg=T["bg"], fg=T["text"],
                 font=(FONT, 22, "bold")).pack(anchor="w")
        tk.Label(head, text=self.SUBTITLES[self.page], bg=T["bg"], fg=T["muted"],
                 font=(FONT, 10)).pack(anchor="w")

        body = tk.Frame(main, bg=T["bg"])
        body.pack(fill="both", expand=True)
        {"dashboard": self.page_dashboard, "predict": self.page_predict,
         "compare": self.page_compare, "history": self.page_history}[self.page](body)

    # ---------------- ttk styling (re-applied on theme change) ----------------
    def setup_styles(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure(".", background=T["bg"], foreground=T["text"], font=(FONT, 10))
        st.configure("TLabel", background=T["card"], foreground=T["text"])
        st.configure("Field.TLabel", background=T["card"], foreground=T["muted"],
                     font=(FONT, 8, "bold"))

        st.configure("TEntry", fieldbackground=T["card2"], foreground=T["text"],
                     insertcolor=T["text"], bordercolor=T["border"],
                     lightcolor=T["border"], darkcolor=T["border"], padding=7)
        st.map("TEntry", bordercolor=[("focus", T["accent"])],
               lightcolor=[("focus", T["accent"])], darkcolor=[("focus", T["accent"])])
        st.configure("TCombobox", fieldbackground=T["card2"], background=T["card2"],
                     foreground=T["text"], arrowcolor=T["accent"], bordercolor=T["border"],
                     lightcolor=T["border"], darkcolor=T["border"], padding=7)
        st.map("TCombobox", fieldbackground=[("readonly", T["card2"])],
               foreground=[("readonly", T["text"])], bordercolor=[("focus", T["accent"])])
        self.option_add("*TCombobox*Listbox.background", T["card2"])
        self.option_add("*TCombobox*Listbox.foreground", T["text"])
        self.option_add("*TCombobox*Listbox.selectBackground", T["accent"])
        self.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")
        self.option_add("*TCombobox*Listbox.font", (FONT, 10))

        st.configure("Primary.TButton", background=T["accent"], foreground="#FFFFFF",
                     font=(FONT, 11, "bold"), padding=(18, 11), borderwidth=0)
        st.map("Primary.TButton", background=[("active", T["accent_hover"])])
        st.configure("Ghost.TButton", background=T["card2"], foreground=T["text"],
                     font=(FONT, 10, "bold"), padding=(16, 11), borderwidth=0)
        st.map("Ghost.TButton", background=[("active", T["border"])])
        st.configure("Danger.TButton", background=T["card2"], foreground=T["red"],
                     font=(FONT, 10, "bold"), padding=(16, 11), borderwidth=0)
        st.map("Danger.TButton", background=[("active", T["border"])])

        for name, col in (("Red", T["red"]), ("Green", T["green"])):
            st.configure(f"{name}.Horizontal.TProgressbar", troughcolor=T["card2"],
                         background=col, bordercolor=T["card"], lightcolor=col,
                         darkcolor=col, thickness=10)

        st.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])   # no white border
        st.configure("Treeview", background=T["card"], fieldbackground=T["card"],
                     foreground=T["text"], rowheight=32, borderwidth=0, font=(FONT, 10))
        st.configure("Treeview.Heading", background=T["card2"], foreground=T["accent"],
                     font=(FONT, 9, "bold"), padding=(8, 9), borderwidth=0)
        st.map("Treeview", background=[("selected", T["border"])],
               foreground=[("selected", T["text"])])
        st.map("Treeview.Heading", background=[("active", T["card2"])])

    # ---------------- small layout helpers ----------------
    def card(self, parent):
        """A panel: frame with a thin border."""
        return tk.Frame(parent, bg=T["card"], highlightbackground=T["border"],
                        highlightthickness=1)

    def card_title(self, parent, text, sub=None):
        tk.Label(parent, text=text, bg=T["card"], fg=T["text"],
                 font=(FONT, 12, "bold")).pack(anchor="w", padx=18, pady=(14, 0))
        if sub:
            tk.Label(parent, text=sub, bg=T["card"], fg=T["muted"],
                     font=(FONT, 9)).pack(anchor="w", padx=18)

    # ---------------- sidebar ----------------
    def build_sidebar(self):
        sb = tk.Frame(self, bg=T["side"], width=230, highlightbackground=T["border"],
                      highlightthickness=1)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        logo = tk.Frame(sb, bg=T["side"])
        logo.pack(fill="x", padx=20, pady=(26, 22))
        badge = tk.Canvas(logo, width=42, height=42, bg=T["side"], highlightthickness=0)
        badge.create_oval(1, 1, 41, 41, fill=T["accent"], outline="")
        badge.create_text(21, 21, text="↺", fill="#FFFFFF", font=(FONT, 20, "bold"))
        badge.pack(side="left")
        t = tk.Frame(logo, bg=T["side"])
        t.pack(side="left", padx=10)
        tk.Label(t, text="ReturnIQ", bg=T["side"], fg=T["text"],
                 font=(FONT, 15, "bold")).pack(anchor="w")
        tk.Label(t, text="Return prediction", bg=T["side"], fg=T["muted"],
                 font=(FONT, 8)).pack(anchor="w")

        tk.Label(sb, text="MENU", bg=T["side"], fg=T["muted"], font=(FONT, 8, "bold")
                 ).pack(anchor="w", padx=22, pady=(0, 6))
        for key, icon, text in self.PAGES:
            self.nav_item(sb, key, icon, text)

        # bottom area: theme switch + exit
        bottom = tk.Frame(sb, bg=T["side"])
        bottom.pack(side="bottom", fill="x", pady=(0, 20))
        ttk.Button(bottom, text="Exit", style="Danger.TButton",
                   command=self.destroy).pack(side="bottom", fill="x", padx=20, pady=(14, 0))
        row = tk.Frame(bottom, bg=T["side"])
        row.pack(side="bottom", fill="x", padx=22)
        tk.Label(row, text="🌙  Dark mode" if self.dark else "☀  Light mode",
                 bg=T["side"], fg=T["text"], font=(FONT, 10)).pack(side="left")
        Switch(row, self.dark, self.toggle_theme, T["side"]).pack(side="right")

    def nav_item(self, parent, key, icon, text):
        active = key == self.page
        bg = T["card2"] if active else T["side"]
        f = tk.Frame(parent, bg=bg, cursor="hand2")
        f.pack(fill="x", padx=12, pady=2)
        tk.Frame(f, bg=T["accent"] if active else bg, width=4).pack(side="left", fill="y")
        lbl = tk.Label(f, text=f"{icon}   {text}", bg=bg, anchor="w", padx=14, pady=11,
                       fg=T["accent"] if active else T["text"],
                       font=(FONT, 10, "bold" if active else "normal"))
        lbl.pack(side="left", fill="x", expand=True)
        for w in (f, lbl):
            w.bind("<Button-1>", lambda e, k=key: self.go(k))
            if not active:                           # hover effect
                w.bind("<Enter>", lambda e, a=f, b=lbl: (a.config(bg=T["card2"]), b.config(bg=T["card2"])))
                w.bind("<Leave>", lambda e, a=f, b=lbl: (a.config(bg=T["side"]), b.config(bg=T["side"])))

    # ========================================================================
    # PAGE 1 : DASHBOARD
    # ========================================================================
    def page_dashboard(self, body):
        # ---- KPI cards ----
        best = COMPARISON.iloc[0] if COMPARISON is not None else None
        kpis = [
            ("BEST MODEL", BEST_MODEL_NAME, "chosen by ROC-AUC", T["accent"]),
            ("ROC-AUC", f"{best['ROC-AUC']:.4f}" if best is not None else "-", "test set", T["green"]),
            ("F1-SCORE", f"{best['F1-Score']:.4f}" if best is not None else "-", "test set", T["green"]),
            ("PREDICTIONS", str(len(self.history)), "made in this session", T["accent"]),
        ]
        row = tk.Frame(body, bg=T["bg"])
        row.pack(fill="x")
        for i, (lab, val, sub, col) in enumerate(kpis):
            row.columnconfigure(i, weight=1, uniform="k")
            c = self.card(row)
            c.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 7, 0 if i == 3 else 7))
            tk.Frame(c, bg=col, height=3).pack(fill="x")
            tk.Label(c, text=lab, bg=T["card"], fg=T["muted"], font=(FONT, 8, "bold")
                     ).pack(anchor="w", padx=16, pady=(12, 0))
            tk.Label(c, text=val, bg=T["card"], fg=T["text"],
                     font=(FONT, 14 if len(val) > 10 else 17, "bold")
                     ).pack(anchor="w", padx=16)
            tk.Label(c, text=sub, bg=T["card"], fg=T["muted"], font=(FONT, 9)
                     ).pack(anchor="w", padx=16, pady=(0, 12))

        # ---- charts row ----
        mid = tk.Frame(body, bg=T["bg"])
        mid.pack(fill="x", pady=14)
        mid.columnconfigure((0, 1), weight=1, uniform="m")

        c1 = self.card(mid)
        c1.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        self.card_title(c1, "Model leaderboard", "ROC-AUC on the test set")
        cv = tk.Canvas(c1, width=400, height=5 * 34 + 8, bg=T["card"], highlightthickness=0)
        cv.pack(padx=18, pady=(8, 14))
        if COMPARISON is not None:
            hbars(cv, list(zip(COMPARISON["Model"], COMPARISON["ROC-AUC"])), row_h=34)

        c2 = self.card(mid)
        c2.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        self.card_title(c2, "What drives returns?", "Model coefficients: red raises risk, green lowers it")
        cv2 = tk.Canvas(c2, width=400, height=8 * 22 + 8, bg=T["card"], highlightthickness=0)
        cv2.pack(padx=18, pady=(8, 14))
        factors = get_factors()
        if factors:
            diverging(cv2, factors, row_h=22)
        else:
            cv2.create_text(200, 70, text="Not available for this model type", fill=T["muted"])

        # ---- recent predictions ----
        c3 = self.card(body)
        c3.pack(fill="both", expand=True)
        self.card_title(c3, "Recent predictions")
        ttk.Button(c3, text="New prediction  →", style="Primary.TButton",
                   command=lambda: self.go("predict")).place(relx=1, x=-18, y=8, anchor="ne")
        if self.history:
            self.history_table(c3, self.history[-3:][::-1], height=3, compact=True)
        else:
            tk.Label(c3, text="No predictions yet - open the Predict page to run your first one.",
                     bg=T["card"], fg=T["muted"], font=(FONT, 10)).pack(pady=22)

    # ========================================================================
    # PAGE 2 : PREDICT
    # ========================================================================
    def page_predict(self, body):
        left = self.card(body)
        left.pack(side="left", fill="y")
        right = self.card(body)
        right.pack(side="left", fill="both", expand=True, padx=(14, 0))

        form = tk.Frame(left, bg=T["card"])
        form.pack(padx=22, pady=16)

        self.section(form, "🛒  Order details", 0)
        self.combo(form, "Product Category", "cat", CATEGORICAL["Product Category"], 1, 0)
        self.combo(form, "Payment Method", "pay", CATEGORICAL["Payment Method"], 1, 1)
        self.entry(form, "Product Price ($)", "price", 2, 0)
        self.entry(form, "Quantity", "qty", 2, 1)

        self.section(form, "👤  Customer", 3)
        self.entry(form, "Customer Age (18-100)", "age", 4, 0)
        self.combo(form, "Gender", "gender", CATEGORICAL["Gender"], 4, 1)

        self.section(form, "🕒  Purchase time", 5)
        tf = tk.Frame(form, bg=T["card"])
        tf.grid(row=6, column=0, columnspan=2, sticky="w")
        for i, (lab, key, vals, w) in enumerate((("Month", "month", MONTHS, 13),
                                                  ("Day of Week", "day", DAYS, 13),
                                                  ("Hour", "hour", HOURS, 7))):
            self.combo(tf, lab, key, vals, 0, i, w=w, pad=(0, 12 if i < 2 else 0))

        auto = tk.Frame(form, bg=T["card2"])
        auto.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        tk.Label(auto, text="AUTO-CALCULATED", bg=T["card2"], fg=T["accent"],
                 font=(FONT, 8, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
        tk.Label(auto, textvariable=self.auto_var, bg=T["card2"], fg=T["text"],
                 font=(FONT, 10)).pack(anchor="w", padx=12, pady=(2, 9))

        btns = tk.Frame(form, bg=T["card"])
        btns.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        ttk.Button(btns, text="Predict", style="Primary.TButton",
                   command=self.predict).pack(side="left", fill="x", expand=True)
        ttk.Button(btns, text="Clear", style="Ghost.TButton",
                   command=self.clear).pack(side="left", padx=(8, 0))

        # ---- result panel ----
        tk.Label(right, text="PREDICTION RESULT", bg=T["card"], fg=T["muted"],
                 font=(FONT, 9, "bold")).pack(pady=(22, 4))
        self.gauge = Gauge(right, T["card"], size=300)
        self.gauge.pack(pady=(4, 0), padx=20)
        self.result_lbl = tk.Label(right, bg=T["card"], font=(FONT, 15, "bold"), wraplength=300)
        self.result_lbl.pack(pady=(6, 4))
        self.detail_lbl = tk.Label(right, bg=T["card"], fg=T["muted"], font=(FONT, 10),
                                   wraplength=300, justify="center")
        self.detail_lbl.pack()
        self.bar = ttk.Progressbar(right, length=290, maximum=100, value=0,
                                   style="Green.Horizontal.TProgressbar")
        self.bar.pack(pady=(18, 0))
        tk.Label(right, text="Return probability", bg=T["card"], fg=T["muted"],
                 font=(FONT, 8)).pack(pady=(4, 0))

        if self.last is None:
            self.gauge.idle()
            self.result_lbl.config(text="Ready - press Predict", fg=T["muted"])
        else:                                        # restore after theme / page change
            self.show_result(self.last, animate=False)

    def section(self, parent, text, r):
        tk.Label(parent, text=text, bg=T["card"], fg=T["text"], font=(FONT, 12, "bold")
                 ).grid(row=r, column=0, columnspan=2, sticky="w", pady=(0 if r == 0 else 12, 6))

    def _field(self, parent, label, r, c, pad=(0, 14)):
        f = tk.Frame(parent, bg=T["card"])
        f.grid(row=r, column=c, padx=pad, pady=(0, 6), sticky="w")
        ttk.Label(f, text=label.upper(), style="Field.TLabel").pack(anchor="w", pady=(0, 3))
        return f

    def combo(self, parent, label, key, values, r, c, w=21, pad=(0, 14)):
        ttk.Combobox(self._field(parent, label, r, c, pad), textvariable=self.v[key],
                     values=values, state="readonly", width=w).pack()

    def entry(self, parent, label, key, r, c):
        ttk.Entry(self._field(parent, label, r, c), textvariable=self.v[key], width=23).pack()

    def update_auto(self):
        """Live preview of the three auto-calculated features."""
        try:
            price, qty = float(self.v["price"].get()), int(self.v["qty"].get())
            total, ppu = f"{price * qty:,.2f}", f"{price:,.2f}"
        except ValueError:
            total = ppu = "-"
        day = self.v["day"].get()
        wk = "-" if not day else ("Yes" if int(day.split(" - ")[0]) >= 5 else "No")
        self.auto_var.set(f"Total: {total}      Price/Unit: {ppu}      Weekend: {wk}")

    # ---- validation ----
    def read_inputs(self):
        """Validate all fields. Returns clean values, or None after an error popup."""
        v = {k: x.get().strip() for k, x in self.v.items()}
        for key, name in (("cat", "Product Category"), ("pay", "Payment Method"),
                          ("gender", "Gender"), ("month", "Month"),
                          ("day", "Day of Week"), ("hour", "Hour")):
            if not v[key]:
                messagebox.showerror("Missing input", f"Please select: {name}")
                return None
        try:
            price = float(v["price"])
            if price <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid price", "Product Price must be a number greater than 0.")
            return None
        try:
            qty = int(v["qty"])
            if qty < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid quantity", "Quantity must be a whole number, 1 or more.")
            return None
        try:
            age = int(v["age"])
            if not 18 <= age <= 100:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid age", "Customer Age must be a whole number between 18 and 100.")
            return None
        return dict(cat=v["cat"], pay=v["pay"], gender=v["gender"], price=price, qty=qty,
                    age=age, month=int(v["month"].split(" - ")[0]),
                    day=int(v["day"].split(" - ")[0]), hour=int(v["hour"].split(":")[0]))

    # ---- predict ----
    def predict(self):
        d = self.read_inputs()
        if d is None:
            return

        # Auto-calculated features (the user is never asked for these)
        total = d["price"] * d["qty"]
        is_weekend = 1 if d["day"] in (5, 6) else 0
        price_per_unit = d["price"]

        # ONE row with the exact column names, then force the training order
        row = {"Product Category": d["cat"], "Product Price": d["price"],
               "Quantity": d["qty"], "Total Purchase Amount": total,
               "Payment Method": d["pay"], "Customer Age": d["age"],
               "Gender": d["gender"], "Purchase_Month": d["month"],
               "Purchase_DayOfWeek": d["day"], "Purchase_Hour": d["hour"],
               "Is_Weekend": is_weekend, "Price_per_Unit": price_per_unit}
        df = pd.DataFrame([row])[FEATURE_ORDER]

        try:
            prob = float(MODEL.predict_proba(df)[0, 1])   # P(RETURNED)
        except Exception as e:
            messagebox.showerror("Prediction failed", str(e))
            return

        self.last = prob
        self.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"), "Category": d["cat"],
            "Price": d["price"], "Qty": d["qty"], "Payment": d["pay"], "Age": d["age"],
            "Gender": d["gender"], "Return Probability": round(prob * 100, 1),
            "Verdict": "Likely returned" if prob >= 0.5 else "Unlikely returned"})
        self.show_result(prob)

    def show_result(self, prob, animate=True):
        if prob >= 0.5:
            color, msg, style = T["red"], "Likely to be RETURNED", "Red"
        else:
            color, msg, style = T["green"], "Unlikely to be returned", "Green"
        note = "  This is a borderline case." if 0.45 <= prob < 0.55 else ""
        self.result_lbl.config(text=msg, fg=color)
        self.detail_lbl.config(text=f"The model estimates a {prob * 100:.1f}% chance of return "
                                    f"({(1 - prob) * 100:.1f}% chance it is kept).{note}")
        self.bar.configure(style=f"{style}.Horizontal.TProgressbar", value=prob * 100)
        self.gauge.show(prob, color, animate)

    def clear(self):
        """Reset the form to its default values and clear the result."""
        for k, val in default_inputs().items():
            self.v[k].set(val)
        self.last = None
        self.gauge.idle()
        self.result_lbl.config(text="Ready - press Predict", fg=T["muted"])
        self.detail_lbl.config(text="")
        self.bar.configure(value=0)

    # ========================================================================
    # PAGE 3 : MODEL COMPARISON  (reads model_comparison.csv)
    # ========================================================================
    def page_compare(self, body):
        if COMPARISON is None:
            tk.Label(body, text="model_comparison.csv not found next to app.py",
                     bg=T["bg"], fg=T["red"], font=(FONT, 13, "bold")).pack(pady=60)
            return
        df = COMPARISON
        c = self.card(body)
        c.pack(fill="x")
        self.card_title(c, "Final comparison table",
                        f"Ranked by ROC-AUC  •  winner: {df.iloc[0]['Model']}")
        shown = [x for x in df.columns if x != "CV ROC-AUC (std)"]
        cols = ["Rank"] + shown
        tree = ttk.Treeview(c, columns=cols, show="headings", height=5)
        for col in cols:
            tree.heading(col, text=col.replace("CV ROC-AUC (mean)", "CV AUC").replace("Train Time (s)", "Time (s)"))
            tree.column(col, width=50 if col == "Rank" else 170 if col == "Model" else 88,
                        anchor="w" if col == "Model" else "center")
        tree.tag_configure("best", foreground=T["green"], background=T["best_bg"])
        tree.tag_configure("odd", background=T["card2"])
        for i, r in df.iterrows():
            vals = [f"#{i + 1}", ("🏆 " if i == 0 else "") + r["Model"]] + [f"{r[x]:.4f}" for x in shown[1:]]
            tree.insert("", "end", values=vals, tags=("best" if i == 0 else "odd" if i % 2 else "",))
        tree.pack(fill="x", padx=18, pady=(10, 16))

        c2 = self.card(body)
        c2.pack(fill="both", expand=True, pady=(14, 0))
        self.card_title(c2, "Compare by metric")
        self.metric = tk.StringVar(value="ROC-AUC")
        cb = ttk.Combobox(c2, textvariable=self.metric, state="readonly", width=14,
                          values=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
        cb.place(relx=1, x=-18, y=12, anchor="ne")
        chart = tk.Canvas(c2, width=860, height=5 * 40 + 8, bg=T["card"], highlightthickness=0)
        chart.pack(padx=18, pady=(8, 14))

        def draw(_=None):
            m = self.metric.get()
            d = df.sort_values(m, ascending=False)
            hbars(chart, list(zip(d["Model"], d[m])), label_w=170)
        cb.bind("<<ComboboxSelected>>", draw)
        draw()

    # ========================================================================
    # PAGE 4 : HISTORY
    # ========================================================================
    def history_table(self, parent, rows, height=10, compact=False):
        cols = ["Time", "Category", "Price", "Qty", "Payment", "Age", "Gender",
                "Return Probability", "Verdict"]
        if compact:
            cols = ["Time", "Category", "Price", "Qty", "Payment", "Return Probability", "Verdict"]
        tree = ttk.Treeview(parent, columns=cols, show="headings", height=height)
        for col in cols:
            tree.heading(col, text=col.replace("Return Probability", "Probability"))
            tree.column(col, width=130 if col in ("Verdict", "Return Probability") else 90,
                        anchor="center")
        tree.tag_configure("red", foreground=T["red"])
        tree.tag_configure("green", foreground=T["green"])
        for r in rows:
            vals = [f"{r[c]}%" if c == "Return Probability" else r[c] for c in cols]
            tree.insert("", "end", values=vals,
                        tags=("red" if r["Return Probability"] >= 50 else "green",))
        tree.pack(fill="x", padx=18, pady=(10, 16))
        return tree

    def page_history(self, body):
        c = self.card(body)
        c.pack(fill="both", expand=True)
        self.card_title(c, "Prediction history", f"{len(self.history)} prediction(s) this session")
        bar = tk.Frame(c, bg=T["card"])
        bar.place(relx=1, x=-18, y=10, anchor="ne")
        ttk.Button(bar, text="Export CSV", style="Primary.TButton",
                   command=self.export_history).pack(side="left")
        ttk.Button(bar, text="Clear", style="Ghost.TButton",
                   command=self.clear_history).pack(side="left", padx=(8, 0))
        if self.history:
            self.history_table(c, self.history[::-1], height=12)
        else:
            tk.Label(c, text="Nothing here yet.\nRun a prediction and it will appear in this list.",
                     bg=T["card"], fg=T["muted"], font=(FONT, 11)).pack(pady=80)

    def export_history(self):
        if not self.history:
            messagebox.showinfo("Export", "There is nothing to export yet.")
            return
        out = path("prediction_history.csv")
        pd.DataFrame(self.history).to_csv(out, index=False)
        messagebox.showinfo("Exported", f"Saved to:\n{out}")

    def clear_history(self):
        self.history.clear()
        self.render()


if __name__ == "__main__":
    App().mainloop()

"""
E-Commerce Return Prediction  -  Tkinter GUI
=============================================
Files needed in the SAME folder as this script:
    return_model.joblib    -> trained scikit-learn Pipeline (preprocessing + model)
    gui_schema.json        -> dropdown options + feature_order
    model_comparison.csv   -> final model comparison table from the notebook

Run:  python app.py
Needs: pandas, scikit-learn, joblib   (tkinter ships with Python)
"""

import json
import os
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import pandas as pd

# ----------------------------------------------------------------------------
# 1. LOAD MODEL, SCHEMA AND COMPARISON TABLE
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # folder of app.py


def path(name):
    """Build a full path to a file that sits next to app.py."""
    return os.path.join(BASE_DIR, name)


MODEL = joblib.load(path("return_model.joblib"))          # sklearn Pipeline
with open(path("gui_schema.json"), encoding="utf-8") as f:
    SCHEMA = json.load(f)

FEATURE_ORDER = SCHEMA["feature_order"]     # exact column order the model expects
CATEGORICAL = SCHEMA["categorical"]         # dropdown options for text columns
BEST_MODEL_NAME = SCHEMA.get("best_model", "")

try:
    COMPARISON = pd.read_csv(path("model_comparison.csv"))
except FileNotFoundError:
    COMPARISON = None

# ----------------------------------------------------------------------------
# 2. THEME (colours + fonts) - one place so the whole UI stays consistent
# ----------------------------------------------------------------------------
BG = "#0B1220"        # window background
CARD = "#131C2F"      # card background
CARD_2 = "#1A2540"    # inputs / hover
BORDER = "#26345A"
TEXT = "#E8EEFF"
MUTED = "#8B98BA"
ACCENT = "#7C83FF"    # indigo
ACCENT_2 = "#5B63F0"
RED = "#FF5C7A"
GREEN = "#2DDC9A"
AMBER = "#FFC857"

FONT = "Segoe UI"     # falls back automatically on Mac/Linux
F_TITLE = (FONT, 22, "bold")
F_SUB = (FONT, 10)
F_H = (FONT, 12, "bold")
F_LABEL = (FONT, 9, "bold")
F_BODY = (FONT, 10)

MONTHS = ["1 - January", "2 - February", "3 - March", "4 - April", "5 - May",
          "6 - June", "7 - July", "8 - August", "9 - September",
          "10 - October", "11 - November", "12 - December"]
DAYS = ["0 - Monday", "1 - Tuesday", "2 - Wednesday", "3 - Thursday",
        "4 - Friday", "5 - Saturday", "6 - Sunday"]
HOURS = [f"{h:02d}:00" for h in range(24)]


# ----------------------------------------------------------------------------
# 3. SMALL CUSTOM WIDGET: semicircle probability gauge (drawn on a Canvas)
# ----------------------------------------------------------------------------
class Gauge(tk.Canvas):
    """Half-circle gauge. set(p) animates the arc to p (0..1)."""

    def __init__(self, master, size=300, **kw):
        super().__init__(master, width=size, height=int(size * 0.66),
                         bg=CARD, highlightthickness=0, **kw)
        self.size = size
        self.value = 0.0
        self.color = MUTED
        self.draw(0.0, "--", "")

    def draw(self, p, big_text, small_text):
        self.delete("all")
        s = self.size
        pad = 24
        box = (pad, pad, s - pad, s - pad)
        # background track
        self.create_arc(*box, start=180, extent=-180, style="arc",
                        width=22, outline=CARD_2)
        # coloured value arc
        if p > 0:
            self.create_arc(*box, start=180, extent=-180 * p, style="arc",
                            width=22, outline=self.color)
        # 50% threshold tick
        cx, cy, r = s / 2, s / 2, (s - 2 * pad) / 2
        self.create_line(cx, cy - r - 16, cx, cy - r + 16, fill=MUTED, width=2)
        self.create_text(cx, 8, text="50% threshold", fill=MUTED, font=(FONT, 8))
        # texts
        self.create_text(cx, cy - 18, text=big_text, fill=TEXT,
                         font=(FONT, 34, "bold"))
        self.create_text(cx, cy + 20, text=small_text, fill=MUTED, font=F_SUB)
        self.create_text(pad + 2, cy + 36, text="0%", fill=MUTED, font=(FONT, 8))
        self.create_text(s - pad - 6, cy + 36, text="100%", fill=MUTED, font=(FONT, 8))

    def animate(self, target, color, small_text, steps=24):
        """Smoothly grow the arc from 0 to target (nice visual effect)."""
        self.color = color

        def step(i=0):
            p = target * (i / steps)
            self.draw(p, f"{p * 100:.1f}%", small_text)
            if i < steps:
                self.after(14, step, i + 1)
        step()

    def reset(self):
        self.color = MUTED
        self.draw(0.0, "--", "waiting for input")


# ----------------------------------------------------------------------------
# 4. MAIN APPLICATION
# ----------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("E-Commerce Return Prediction")
        self.geometry("1060x780")
        self.resizable(False, False)          # window not resizable (requirement)
        self.configure(bg=BG)

        self.setup_styles()
        self.build_header()
        self.build_tabs()
        self.center()

    # ---------------- styling ----------------
    def setup_styles(self):
        st = ttk.Style(self)
        st.theme_use("clam")                  # 'clam' lets us recolour everything
        st.configure(".", background=BG, foreground=TEXT, font=F_BODY)

        st.configure("TFrame", background=BG)
        st.configure("Card.TFrame", background=CARD)
        st.configure("TLabel", background=CARD, foreground=TEXT)
        st.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=F_SUB)
        st.configure("Field.TLabel", background=CARD, foreground=MUTED, font=F_LABEL)
        st.configure("H.TLabel", background=CARD, foreground=TEXT, font=F_H)

        # Notebook (tabs)
        st.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
        st.configure("TNotebook.Tab", background=BG, foreground=MUTED,
                     padding=(22, 10), font=(FONT, 10, "bold"), borderwidth=0)
        st.map("TNotebook.Tab",
               background=[("selected", CARD)],
               foreground=[("selected", ACCENT)])

        # Inputs
        st.configure("TEntry", fieldbackground=CARD_2, foreground=TEXT,
                     insertcolor=TEXT, bordercolor=BORDER, lightcolor=BORDER,
                     darkcolor=BORDER, padding=7)
        st.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)],
               darkcolor=[("focus", ACCENT)])
        st.configure("TCombobox", fieldbackground=CARD_2, background=CARD_2,
                     foreground=TEXT, arrowcolor=ACCENT, bordercolor=BORDER,
                     lightcolor=BORDER, darkcolor=BORDER, padding=7)
        st.map("TCombobox", fieldbackground=[("readonly", CARD_2)],
               foreground=[("readonly", TEXT)], bordercolor=[("focus", ACCENT)])
        self.option_add("*TCombobox*Listbox.background", CARD_2)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.font", F_BODY)

        # Buttons
        st.configure("Primary.TButton", background=ACCENT, foreground="white",
                     font=(FONT, 11, "bold"), padding=(18, 11), borderwidth=0)
        st.map("Primary.TButton", background=[("active", ACCENT_2), ("pressed", ACCENT_2)])
        st.configure("Ghost.TButton", background=CARD_2, foreground=TEXT,
                     font=(FONT, 10, "bold"), padding=(18, 11), borderwidth=0)
        st.map("Ghost.TButton", background=[("active", BORDER)])
        st.configure("Danger.TButton", background=CARD_2, foreground=RED,
                     font=(FONT, 10, "bold"), padding=(18, 11), borderwidth=0)
        st.map("Danger.TButton", background=[("active", BORDER)])

        # Progress bars
        for name, col in (("Red", RED), ("Green", GREEN)):
            st.configure(f"{name}.Horizontal.TProgressbar", troughcolor=CARD_2,
                         background=col, bordercolor=CARD, lightcolor=col,
                         darkcolor=col, thickness=10)

        # Table (Treeview)
        st.configure("Treeview", background=CARD, fieldbackground=CARD,
                     foreground=TEXT, rowheight=34, borderwidth=0, font=F_BODY)
        st.configure("Treeview.Heading", background=CARD_2, foreground=ACCENT,
                     font=(FONT, 9, "bold"), padding=(8, 9), borderwidth=0)
        st.map("Treeview", background=[("selected", BORDER)])
        st.map("Treeview.Heading", background=[("active", CARD_2)])

    def center(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth() - w) // 2
        y = max(10, (self.winfo_screenheight() - h) // 2 - 20)
        self.geometry(f"+{x}+{y}")

    # ---------------- header ----------------
    def build_header(self):
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=28, pady=(20, 6))
        # little accent badge
        badge = tk.Canvas(head, width=46, height=46, bg=BG, highlightthickness=0)
        badge.create_oval(2, 2, 44, 44, fill=ACCENT, outline="")
        badge.create_text(23, 23, text="↺", fill="white", font=(FONT, 22, "bold"))
        badge.pack(side="left", padx=(0, 14))
        txt = tk.Frame(head, bg=BG)
        txt.pack(side="left")
        tk.Label(txt, text="E-Commerce Return Prediction", bg=BG, fg=TEXT,
                 font=F_TITLE).pack(anchor="w")
        tk.Label(txt, text="Estimate the chance that an order will be sent back  •  "
                           f"Model: {BEST_MODEL_NAME or 'ML Pipeline'}",
                 bg=BG, fg=MUTED, font=F_SUB).pack(anchor="w")

    # ---------------- tabs ----------------
    def build_tabs(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=28, pady=(8, 22))
        self.tab_predict = tk.Frame(nb, bg=CARD)
        self.tab_compare = tk.Frame(nb, bg=CARD)
        nb.add(self.tab_predict, text="  🔮  Predict  ")
        nb.add(self.tab_compare, text="  📊  Model Comparison  ")
        self.build_predict_tab()
        self.build_compare_tab()

    # ========================================================================
    # TAB 1 : PREDICT
    # ========================================================================
    def build_predict_tab(self):
        root = self.tab_predict
        left = tk.Frame(root, bg=CARD)
        left.pack(side="left", fill="y", padx=(26, 12), pady=18)
        right = tk.Frame(root, bg=CARD)
        right.pack(side="left", fill="both", expand=True, padx=(12, 26), pady=18)

        # Tk variables hold the current value of each widget
        self.v = {
            "cat": tk.StringVar(), "price": tk.StringVar(), "qty": tk.StringVar(),
            "pay": tk.StringVar(), "age": tk.StringVar(), "gender": tk.StringVar(),
            "month": tk.StringVar(), "day": tk.StringVar(), "hour": tk.StringVar(),
        }

        # --- section: order ---
        self.section(left, "🛒  Order details")
        row = tk.Frame(left, bg=CARD); row.pack(fill="x")
        self.combo(row, "Product Category", self.v["cat"], CATEGORICAL["Product Category"], 0, 0)
        self.combo(row, "Payment Method", self.v["pay"], CATEGORICAL["Payment Method"], 0, 1)
        self.entry(row, "Product Price ($)", self.v["price"], 1, 0)
        self.entry(row, "Quantity", self.v["qty"], 1, 1)

        # --- section: customer ---
        self.section(left, "👤  Customer", top=14)
        row = tk.Frame(left, bg=CARD); row.pack(fill="x")
        self.entry(row, "Customer Age (18-100)", self.v["age"], 0, 0)
        self.combo(row, "Gender", self.v["gender"], CATEGORICAL["Gender"], 0, 1)

        # --- section: time ---
        self.section(left, "🕒  Purchase time", top=14)
        row = tk.Frame(left, bg=CARD); row.pack(fill="x")
        self.combo(row, "Month", self.v["month"], MONTHS, 0, 0, w=13)
        self.combo(row, "Day of Week", self.v["day"], DAYS, 0, 1, w=13)
        self.combo(row, "Hour", self.v["hour"], HOURS, 0, 2, w=8)

        # live preview of the auto-calculated fields
        self.auto_var = tk.StringVar()
        auto = tk.Frame(left, bg=CARD_2)
        auto.pack(fill="x", pady=(10, 0))
        tk.Label(auto, text="AUTO-CALCULATED", bg=CARD_2, fg=ACCENT,
                 font=(FONT, 8, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
        tk.Label(auto, textvariable=self.auto_var, bg=CARD_2, fg=TEXT,
                 font=F_BODY, justify="left").pack(anchor="w", padx=12, pady=(2, 9))
        for k in ("price", "qty", "day"):
            self.v[k].trace_add("write", lambda *a: self.update_auto())
        self.update_auto()

        # --- buttons ---
        btns = tk.Frame(left, bg=CARD); btns.pack(fill="x", pady=(16, 0))
        ttk.Button(btns, text="Predict", style="Primary.TButton",
                   command=self.predict).pack(side="left", fill="x", expand=True)
        ttk.Button(btns, text="Clear", style="Ghost.TButton",
                   command=self.clear).pack(side="left", padx=8)
        ttk.Button(btns, text="Exit", style="Danger.TButton",
                   command=self.destroy).pack(side="left")
        self.bind("<Return>", lambda e: self.predict())   # Enter key = Predict

        # --- right: result card ---
        res = tk.Frame(right, bg=CARD_2, highlightbackground=BORDER, highlightthickness=1)
        res.pack(fill="both", expand=True)
        tk.Label(res, text="PREDICTION RESULT", bg=CARD_2, fg=MUTED,
                 font=(FONT, 9, "bold")).pack(pady=(20, 4))
        self.gauge = Gauge(res, size=300)
        self.gauge.configure(bg=CARD_2)
        self.gauge.reset()
        self.gauge.pack(pady=(4, 0))

        self.result_lbl = tk.Label(res, text="Fill in the form and press Predict",
                                   bg=CARD_2, fg=MUTED, font=(FONT, 15, "bold"),
                                   wraplength=300)
        self.result_lbl.pack(pady=(6, 4))
        self.detail_lbl = tk.Label(res, text="", bg=CARD_2, fg=MUTED, font=F_SUB,
                                   wraplength=300, justify="center")
        self.detail_lbl.pack()

        # requirement: progress bar showing the probability
        self.bar = ttk.Progressbar(res, length=290, maximum=100, value=0,
                                   style="Green.Horizontal.TProgressbar")
        self.bar.pack(pady=(18, 0))
        tk.Label(res, text="Return probability", bg=CARD_2, fg=MUTED,
                 font=(FONT, 8)).pack(pady=(4, 20))

    # ---- little layout helpers (keep the code short & readable) ----
    def section(self, parent, text, top=0):
        tk.Label(parent, text=text, bg=CARD, fg=TEXT, font=F_H).pack(
            anchor="w", pady=(top, 8))

    def _field(self, parent, label, r, c):
        f = tk.Frame(parent, bg=CARD)
        f.grid(row=r, column=c, padx=(0, 14), pady=(0, 6), sticky="w")
        ttk.Label(f, text=label.upper(), style="Field.TLabel").pack(anchor="w", pady=(0, 3))
        return f

    def combo(self, parent, label, var, values, r, c, w=21):
        f = self._field(parent, label, r, c)
        ttk.Combobox(f, textvariable=var, values=values, state="readonly",
                     width=w).pack()

    def entry(self, parent, label, var, r, c):
        f = self._field(parent, label, r, c)
        ttk.Entry(f, textvariable=var, width=23).pack()

    # ---- live "auto-calculated" text ----
    def update_auto(self):
        try:
            price = float(self.v["price"].get())
            qty = int(self.v["qty"].get())
            total = f"{price * qty:,.2f}"
            ppu = f"{price:,.2f}"
        except ValueError:
            total = ppu = "-"
        day = self.v["day"].get()
        wk = "-" if not day else ("Yes (1)" if int(day.split(" - ")[0]) >= 5 else "No (0)")
        self.auto_var.set(f"Total Purchase Amount: {total}     Price per Unit: {ppu}"
                          f"     Is Weekend: {wk}")

    # ---- validation ----
    def read_inputs(self):
        """Validate every field. Returns a dict of clean values or None
        (after showing a messagebox error)."""
        v = {k: x.get().strip() for k, x in self.v.items()}

        # all dropdowns must be chosen
        for key, name in (("cat", "Product Category"), ("pay", "Payment Method"),
                          ("gender", "Gender"), ("month", "Month"),
                          ("day", "Day of Week"), ("hour", "Hour")):
            if not v[key]:
                messagebox.showerror("Missing input", f"Please select: {name}")
                return None

        # price: must be a positive number
        try:
            price = float(v["price"])
            if price <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid price", "Product Price must be a number greater than 0.")
            return None

        # quantity: whole number >= 1
        try:
            qty = int(v["qty"])
            if qty < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid quantity", "Quantity must be a whole number, 1 or more.")
            return None

        # age: whole number 18..100
        try:
            age = int(v["age"])
            if not 18 <= age <= 100:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid age", "Customer Age must be a whole number between 18 and 100.")
            return None

        return dict(cat=v["cat"], pay=v["pay"], gender=v["gender"], price=price,
                    qty=qty, age=age,
                    month=int(v["month"].split(" - ")[0]),
                    day=int(v["day"].split(" - ")[0]),
                    hour=int(v["hour"].split(":")[0]))

    # ---- predict ----
    def predict(self):
        d = self.read_inputs()
        if d is None:
            return

        # Auto-calculated features (not asked from the user)
        total = d["price"] * d["qty"]
        is_weekend = 1 if d["day"] in (5, 6) else 0
        price_per_unit = d["price"]

        # Build ONE row with EXACT column names, then force the order
        row = {
            "Product Category": d["cat"],
            "Product Price": d["price"],
            "Quantity": d["qty"],
            "Total Purchase Amount": total,
            "Payment Method": d["pay"],
            "Customer Age": d["age"],
            "Gender": d["gender"],
            "Purchase_Month": d["month"],
            "Purchase_DayOfWeek": d["day"],
            "Purchase_Hour": d["hour"],
            "Is_Weekend": is_weekend,
            "Price_per_Unit": price_per_unit,
        }
        df = pd.DataFrame([row])[FEATURE_ORDER]

        try:
            # predict_proba -> [[P(kept), P(returned)]] ; index [0,1] = P(returned)
            prob = float(MODEL.predict_proba(df)[0, 1])
        except Exception as e:                      # show any model error nicely
            messagebox.showerror("Prediction failed", str(e))
            return

        self.show_result(prob)

    def show_result(self, prob):
        if prob >= 0.5:
            color, msg, style = RED, "Likely to be RETURNED", "Red"
        else:
            color, msg, style = GREEN, "Unlikely to be returned", "Green"
        self.result_lbl.config(text=msg, fg=color)
        self.detail_lbl.config(
            text=f"The model estimates a {prob * 100:.1f}% chance of return "
                 f"({(1 - prob) * 100:.1f}% chance it is kept).")
        self.bar.configure(style=f"{style}.Horizontal.TProgressbar", value=prob * 100)
        self.gauge.animate(prob, color, "return probability")

    def clear(self):
        for var in self.v.values():
            var.set("")
        self.gauge.reset()
        self.result_lbl.config(text="Fill in the form and press Predict", fg=MUTED)
        self.detail_lbl.config(text="")
        self.bar.configure(value=0)

    # ========================================================================
    # TAB 2 : MODEL COMPARISON  (table + bar chart read from model_comparison.csv)
    # ========================================================================
    def build_compare_tab(self):
        root = self.tab_compare
        if COMPARISON is None:
            tk.Label(root, text="model_comparison.csv not found next to app.py",
                     bg=CARD, fg=RED, font=F_H).pack(pady=60)
            return

        df = COMPARISON.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
        self.cmp_df = df
        best = df.iloc[0]

        tk.Label(root, text="Final Model Comparison", bg=CARD, fg=TEXT,
                 font=F_H).pack(anchor="w", padx=26, pady=(18, 0))
        tk.Label(root, text=f"Evaluated on the held-out test set  •  ranked by ROC-AUC  •  "
                            f"winner: {best['Model']} ({best['ROC-AUC']:.4f})",
                 bg=CARD, fg=MUTED, font=F_SUB).pack(anchor="w", padx=26, pady=(0, 10))

        # ---- table ----
        shown = [c for c in df.columns if c != "CV ROC-AUC (std)"]   # std stays in the CSV
        cols = ["Rank"] + shown
        tree = ttk.Treeview(root, columns=cols, show="headings", height=5)
        widths = {"Rank": 50, "Model": 170}
        for c in cols:
            tree.heading(c, text=c.replace("CV ROC-AUC (mean)", "CV AUC"))
            tree.column(c, width=widths.get(c, 96), anchor="w" if c == "Model" else "center")
        tree.tag_configure("best", foreground=GREEN, background="#12281F")
        tree.tag_configure("odd", background=CARD_2)
        for i, r in df.iterrows():
            vals = [f"#{i + 1}"] + [r[c] if c == "Model" else f"{r[c]:.4f}" for c in shown]
            tag = "best" if i == 0 else ("odd" if i % 2 else "")
            name = f"🏆 {r['Model']}" if i == 0 else r["Model"]
            vals[1] = name
            tree.insert("", "end", values=vals, tags=(tag,))
        tree.pack(fill="x", padx=26)

        # ---- metric selector + chart ----
        bar = tk.Frame(root, bg=CARD); bar.pack(fill="x", padx=26, pady=(14, 0))
        tk.Label(bar, text="Compare by:", bg=CARD, fg=MUTED, font=F_SUB).pack(side="left")
        self.metric = tk.StringVar(value="ROC-AUC")
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        cb = ttk.Combobox(bar, textvariable=self.metric, values=metrics,
                          state="readonly", width=14)
        cb.pack(side="left", padx=10)
        cb.bind("<<ComboboxSelected>>", lambda e: self.draw_chart())

        self.chart = tk.Canvas(root, width=900, height=225, bg=CARD, highlightthickness=0)
        self.chart.pack(padx=26, pady=(6, 10))
        self.draw_chart()

    def draw_chart(self):
        """Horizontal bar chart of the chosen metric, drawn with Canvas."""
        c = self.chart
        c.delete("all")
        m = self.metric.get()
        df = self.cmp_df.sort_values(m, ascending=False).reset_index(drop=True)
        left, right, top, rh = 190, 80, 6, 43
        w = int(c["width"]) - left - right
        top_val = max(df[m].max() * 1.12, 0.01)
        for i, r in df.iterrows():
            y = top + i * rh
            c.create_text(left - 14, y + 15, text=r["Model"], anchor="e",
                          fill=TEXT, font=F_BODY)
            c.create_rectangle(left, y + 3, left + w, y + 27, fill=CARD_2, outline="")
            col = GREEN if i == 0 else ACCENT
            c.create_rectangle(left, y + 3, left + w * r[m] / top_val, y + 27,
                               fill=col, outline="")
            c.create_text(left + w * r[m] / top_val + 10, y + 15, anchor="w",
                          text=f"{r[m]:.4f}", fill=col, font=(FONT, 10, "bold"))


if __name__ == "__main__":
    App().mainloop()

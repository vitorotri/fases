
import os
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OCTAVE_LABELS = ["64", "125", "250", "500", "1k", "2k", "4k", "8k"]


def clean_group_name(raw: str) -> str:
    parts = raw.strip().split()
    if not parts:
        return "Unnamed"
    if parts[0] == "g":
        parts = parts[1:]
    filtered = [p for p in parts if p.lower() not in ("model", "mesh", "default")]
    filtered = [p for p in filtered if not p.lower().startswith("mesh")]
    if not filtered:
        filtered = parts
    return "_".join(filtered) if filtered else "Unnamed"


class SolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("P.H.A.S.E - Parallel Hybrid Acoustic Solver for Environments")
        self.root.geometry("1300x820")
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.bg = "#15141a"        # main background
        self.panel = "#1b1a22"     # panels
        self.entry_bg = "#22212b"  # entry boxes
        self.fg = "#e6e4f2"        # text
        self.accent = "#a66cff"    # violet accent
        self.accent_hover = "#c08cff"
        self.border = "#2a2835"

        
        self.configure_theme()

        self.solver_var = tk.StringVar(value="3D")
        self.scene_path = tk.StringVar()

        self.config_vars = {
            "h [m]": tk.DoubleVar(value=0.025),
            "t_max [s]": tk.DoubleVar(value=0.09),
            "f_cutoff [Hz]": tk.DoubleVar(value=1100.0),
            "Temp [°C]": tk.DoubleVar(value=22.0),
            "CFL": tk.DoubleVar(value=0.5),
            "poles": tk.IntVar(value=12),
        }

        self.pml_vars = {
            "U (z+)": tk.IntVar(value=0),
            "D (z-)": tk.IntVar(value=0),
            "R (x+)": tk.IntVar(value=0),
            "L (x-)": tk.IntVar(value=0),
            "F (y+)": tk.IntVar(value=0),
            "B (y-)": tk.IntVar(value=1),
        }

        self.src_vars = [
            tk.DoubleVar(value=1.625),
            tk.DoubleVar(value=1.00),
            tk.DoubleVar(value=1.47),
            tk.DoubleVar(value=0.01),
        ]
        self.rec_vars = [
            tk.DoubleVar(value=2.00),
            tk.DoubleVar(value=2.00),
            tk.DoubleVar(value=2.00),
        ]

        self.material_rows = {}
        self.status_var = tk.StringVar(value="Ready.")

        self.build_ui()
    
    def configure_theme(self):
        self.root.configure(bg=self.bg)
    
        self.style.configure(
            ".",
            background=self.bg,
            foreground=self.fg,
            fieldbackground=self.entry_bg,
            bordercolor=self.border,
            lightcolor=self.border,
            darkcolor=self.border
        )
    
        self.style.configure(
            "TFrame",
            background=self.bg
        )
    
        self.style.configure(
            "TLabel",
            background=self.bg,
            foreground=self.fg
        )
    
        self.style.configure(
            "TLabelframe",
            background=self.bg,
            bordercolor=self.border,
            relief="solid"
        )
    
        self.style.configure(
            "TLabelframe.Label",
            background=self.bg,
            foreground=self.accent
        )
    
        self.style.configure(
            "TButton",
            background=self.panel,
            foreground=self.fg,
            borderwidth=1,
            padding=6
        )
    
        self.style.map(
            "TButton",
            background=[("active", self.accent)],
            foreground=[("active", "#ffffff")]
        )
    
        self.style.configure(
            "TEntry",
            fieldbackground=self.entry_bg,
            foreground=self.fg,
            insertcolor=self.fg
        )
    
        self.style.configure(
            "TRadiobutton",
            background=self.bg,
            foreground=self.fg
        )
    
        self.style.configure(
            "TCheckbutton",
            background=self.bg,
            foreground=self.fg
        )

    def solver_dir(self):
        return "3D-solver" if self.solver_var.get() == "3D" else "2D-solver"

    def sim_option(self):
        return "1" if self.solver_var.get() == "3D" else "0"

    def solver_root(self):
        return os.path.join(ROOT, self.solver_dir())

    def build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x")

        solver_frame = ttk.LabelFrame(top, text="Solver")
        solver_frame.pack(fill="x", pady=4)
        ttk.Radiobutton(
            solver_frame, text="2D", variable=self.solver_var, value="2D"
        ).pack(side="left", padx=5, pady=4)
        ttk.Radiobutton(
            solver_frame, text="3D", variable=self.solver_var, value="3D"
        ).pack(side="left", padx=5, pady=4)

        scene_frame = ttk.LabelFrame(outer, text="Scene")
        scene_frame.pack(fill="x", pady=4)

        ttk.Entry(scene_frame, textvariable=self.scene_path, width=85).grid(
            row=0, column=0, padx=5, pady=5, sticky="ew"
        )
        ttk.Button(scene_frame, text="Browse", command=self.browse_scene).grid(
            row=0, column=1, padx=5, pady=5
        )
        ttk.Button(scene_frame, text="Load scene data", command=self.load_materials).grid(
            row=0, column=2, padx=5, pady=5
        )
        ttk.Button(scene_frame, text="Preview", command=self.preview_scene).grid(
            row=0, column=3, padx=5, pady=5
        )
        scene_frame.columnconfigure(0, weight=1)

        middle = ttk.Frame(outer)
        middle.pack(fill="x", pady=4)

        config_frame = ttk.LabelFrame(middle, text="Config")
        config_frame.pack(side="left", fill="both", expand=False, padx=(0, 6))

        for i, (k, v) in enumerate(self.config_vars.items()):
            ttk.Label(config_frame, text=k).grid(row=i, column=0, sticky="w", padx=4, pady=3)
            ttk.Entry(config_frame, textvariable=v, width=12).grid(
                row=i, column=1, sticky="w", padx=4, pady=3
            )

        pml_frame = ttk.LabelFrame(middle, text="PML flags")
        pml_frame.pack(side="left", fill="both", expand=False, padx=6)

        for i, (k, v) in enumerate(self.pml_vars.items()):
            ttk.Checkbutton(pml_frame, text=k, variable=v).grid(
                row=i // 2, column=i % 2, sticky="w", padx=8, pady=4
            )

        pos_frame = ttk.Frame(middle)
        pos_frame.pack(side="left", fill="both", expand=True, padx=(6, 0))

        src_frame = ttk.LabelFrame(pos_frame, text="Source")
        src_frame.pack(fill="x", pady=(0, 6))
        for c, lab in enumerate(["x", "y", "z", "value"]):
            ttk.Label(src_frame, text=lab).grid(row=0, column=c, padx=6, pady=2)
        for c, v in enumerate(self.src_vars):
            ttk.Entry(src_frame, textvariable=v, width=10).grid(row=1, column=c, padx=6, pady=4)

        rec_frame = ttk.LabelFrame(pos_frame, text="Receiver")
        rec_frame.pack(fill="x")
        for c, lab in enumerate(["x", "y", "z"]):
            ttk.Label(rec_frame, text=lab).grid(row=0, column=c, padx=6, pady=2)
        for c, v in enumerate(self.rec_vars):
            ttk.Entry(rec_frame, textvariable=v, width=10).grid(row=1, column=c, padx=6, pady=4)

        materials_outer = ttk.LabelFrame(outer, text="Scene objects / materials")
        materials_outer.pack(fill="both", expand=True, pady=6)

        canvas = tk.Canvas(
            materials_outer,
            height=320,
            bg=self.panel,
            highlightthickness=1,
            highlightbackground=self.border
        )
        scrollbar = ttk.Scrollbar(materials_outer, orient="vertical", command=canvas.yview)
        self.material_frame = ttk.Frame(canvas)

        self.material_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.material_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        bottom = ttk.Frame(outer)
        bottom.pack(fill="x", pady=(8, 0))

        ttk.Button(bottom, text="Reset fittings", command=self.reset_fittings).pack(
            side="left", padx=5
        )
        ttk.Button(bottom, text="Run selected fittings", command=self.run_selected_fittings).pack(
            side="left", padx=5
        )
        ttk.Button(bottom, text="Run simulation", command=self.run_solver).pack(
            side="left", padx=5
        )
        ttk.Button(bottom, text="Plot IR", command=self.plot_ir).pack(side="left", padx=5)
        ttk.Button(bottom, text="Compute params", command=self.compute_acoustic_params).pack(
            side="left", padx=5
        )
        ttk.Button(bottom, text="Plot slices", command=self.plot_slices).pack(side="left", padx=5)

        ttk.Label(bottom, textvariable=self.status_var).pack(side="right", padx=5)

        self.render_material_headers()

    def selected_material_names(self):
        selected = []
        for name, row in self.material_rows.items():
            if bool(row["fit_now"].get()):
                selected.append(name)
        return selected

    def reset_fittings(self):
        try:
            poles_path = os.path.join(self.solver_root(), "config", "poles.txt")
            with open(poles_path, "w", encoding="utf-8") as f:
                f.write("0 0\n")

            self.status_var.set("Reset poles.txt to 0 0")
            messagebox.showinfo("Reset fittings", f"Saved '0 0' to:\n{poles_path}")

        except Exception as e:
            self.status_var.set("Error.")
            messagebox.showerror("Error", str(e))

    def run_selected_fittings(self):
        scene = self.scene_path.get().strip()
        if not scene:
            messagebox.showerror("Error", "Choose an OBJ scene first.")
            return
        if not self.material_rows:
            messagebox.showerror("Error", "Load scene data first.")
            return

        selected = self.selected_material_names()
        if not selected:
            messagebox.showerror("Error", "No material selected in the 'fit now' column.")
            return

        try:
            self.status_var.set("Writing .mat files...")
            self.root.update_idletasks()

            self.write_materials()

            self.status_var.set("Running selected fittings...")
            self.root.update_idletasks()

            fitted = self.run_vectfit(selected_only=True)

            self.load_materials()

            if fitted:
                self.status_var.set("Fitted: " + ", ".join(fitted))
                messagebox.showinfo(
                    "Fitting done",
                    "Fitted materials:\n" + "\n".join(fitted)
                )
            else:
                self.status_var.set("No fitting was run.")

        except Exception as e:
            self.status_var.set("Error.")
            messagebox.showerror("Error", str(e))

    def render_material_headers(self):
        for w in self.material_frame.winfo_children():
            w.destroy()

        headers = ["Object/material", "fit file", "fit now"] + OCTAVE_LABELS
        for c, h in enumerate(headers):
            ttk.Label(
                self.material_frame, text=h, font=("", 10, "bold")
            ).grid(row=0, column=c, padx=4, pady=4, sticky="w")

        self.material_rows = {}

    def browse_scene(self):
        initial_dir = os.path.join(self.solver_root(), "input", "scene")
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("OBJ files", "*.obj")],
        )
        if path:
            self.scene_path.set(path)

    def parse_scene_entities(self, obj_path):
        groups = []
        usemtls = []

        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("g "):
                    name = clean_group_name(line)
                    if name not in groups:
                        groups.append(name)
                elif line.startswith("usemtl "):
                    mat = line.split(maxsplit=1)[1].strip()
                    if mat not in usemtls:
                        usemtls.append(mat)

        meaningful_groups = [g for g in groups if g.lower() not in ("frontcolor", "default")]
        if meaningful_groups:
            return meaningful_groups
        return usemtls

    def read_existing_mat_values(self, name):
        mat_path = os.path.join(self.solver_root(), "input", "mat", f"{name}.mat")
        if not os.path.exists(mat_path):
            return [0.30] * 8

        try:
            with open(mat_path, "r", encoding="utf-8", errors="ignore") as f:
                tokens = []
                for line in f:
                    line = line.split("#", 1)[0].strip()
                    if line:
                        tokens.extend(line.split())
            vals = [float(x) for x in tokens[:8]]
            if len(vals) < 8:
                vals += [0.30] * (8 - len(vals))
            return vals[:8]
        except Exception:
            return [0.30] * 8

    def fit_exists(self, name):
        fit_path = os.path.join(self.solver_root(), "input", "fit", f"{name}.fit")
        return os.path.exists(fit_path)

    def load_materials(self):
        scene = self.scene_path.get().strip()
        if not scene:
            messagebox.showerror("Error", "Choose an OBJ scene first.")
            return
        if not os.path.exists(scene):
            messagebox.showerror("Error", "Scene file not found.")
            return

        names = self.parse_scene_entities(scene)
        self.render_material_headers()

        if not names:
            messagebox.showwarning(
                "Warning",
                "No groups/material names found in the OBJ. I expected names like floor, ceiling, walls.",
            )
            return

        for r, name in enumerate(names, start=1):
            vals = self.read_existing_mat_values(name)
            fit_exists = self.fit_exists(name)

            ttk.Label(self.material_frame, text=name).grid(
                row=r, column=0, padx=4, pady=3, sticky="w"
            )
            ttk.Label(
                self.material_frame,
                text="yes" if fit_exists else "no",
            ).grid(row=r, column=1, padx=4, pady=3, sticky="w")

            fit_now_var = tk.IntVar(value=0 if fit_exists else 1)
            ttk.Checkbutton(self.material_frame, variable=fit_now_var).grid(
                row=r, column=2, padx=4, pady=3
            )

            entry_vars = []
            for c in range(8):
                var = tk.DoubleVar(value=vals[c])
                ttk.Entry(self.material_frame, textvariable=var, width=7).grid(
                    row=r, column=c + 3, padx=3, pady=3
                )
                entry_vars.append(var)

            self.material_rows[name] = {
                "values": entry_vars,
                "fit_now": fit_now_var,
            }

        self.status_var.set(f"Loaded {len(names)} scene object/material rows.")

    def write_config(self):
        path = os.path.join(self.solver_root(), "config", "config.txt")
        lines = [
            self.config_vars["h [m]"].get(),
            self.config_vars["t_max [s]"].get(),
            self.config_vars["f_cutoff [Hz]"].get(),
            self.config_vars["Temp [°C]"].get(),
            self.config_vars["CFL"].get(),
            int(self.config_vars["poles"].get()),
            self.pml_vars["U (z+)"].get(),
            self.pml_vars["D (z-)"].get(),
            self.pml_vars["R (x+)"].get(),
            self.pml_vars["L (x-)"].get(),
            self.pml_vars["F (y+)"].get(),
            self.pml_vars["B (y-)"].get(),
        ]
        with open(path, "w", encoding="utf-8") as f:
            for x in lines:
                f.write(f"{x}\n")

    def write_src_rec(self):
        src_path = os.path.join(self.solver_root(), "input", "src", "src.txt")
        rec_path = os.path.join(self.solver_root(), "input", "rec", "rec.txt")

        with open(src_path, "w", encoding="utf-8") as f:
            f.write(
                f"{self.src_vars[0].get()} {self.src_vars[1].get()} {self.src_vars[2].get()} {self.src_vars[3].get()}\n"
            )

        with open(rec_path, "w", encoding="utf-8") as f:
            f.write(
                f"{self.rec_vars[0].get()} {self.rec_vars[1].get()} {self.rec_vars[2].get()}\n"
            )

    def write_materials(self):
        mat_dir = os.path.join(self.solver_root(), "input", "mat")
        os.makedirs(mat_dir, exist_ok=True)

        for name, row in self.material_rows.items():
            mat_path = os.path.join(mat_dir, f"{name}.mat")
            values = [var.get() for var in row["values"]]
            with open(mat_path, "w", encoding="utf-8") as f:
                f.write("# Frequencies in octave bands, from 64 Hz to 8 kHz\n")
                f.write(" ".join(str(v) for v in values) + "\n")

    def run_vectfit(self, selected_only=False):
        common_dir = os.path.join(ROOT, "common")
        script_path = os.path.join(common_dir, "fminconFIT.py")

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Could not find {script_path}")

        ran = []

        if selected_only:
            names_to_fit = self.selected_material_names()
        else:
            names_to_fit = []
            for name, row in self.material_rows.items():
                need_fit = bool(row["fit_now"].get()) or not self.fit_exists(name)
                if need_fit:
                    names_to_fit.append(name)

        for name in names_to_fit:
            mat_path = os.path.join(self.solver_root(), "input", "mat", f"{name}.mat")

            cmd = ["python3", script_path, mat_path, self.sim_option()]
            proc = subprocess.run(
                cmd,
                cwd=common_dir,
                capture_output=True,
                text=True,
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"fminconFIT failed for {name}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
                )

            if not self.fit_exists(name):
                raise RuntimeError(
                    f"fminconFIT ended for {name}, but {name}.fit was not created in:\n"
                    f"{os.path.join(self.solver_root(), 'input', 'fit')}"
                )

            ran.append(name)

        return ran

    def preview_scene(self):
        scene = self.scene_path.get().strip()
        if not scene:
            messagebox.showerror("Error", "Choose an OBJ scene first.")
            return

        script = os.path.join(ROOT, "common", "preview_scene.py")
        args = [
            "python3",
            script,
            scene,
            str(self.src_vars[0].get()),
            str(self.src_vars[1].get()),
            str(self.src_vars[2].get()),
            str(self.rec_vars[0].get()),
            str(self.rec_vars[1].get()),
            str(self.rec_vars[2].get()),
        ]
        subprocess.run(args)

    def run_solver(self):
        scene = self.scene_path.get().strip()
        if not scene:
            messagebox.showerror("Error", "Choose an OBJ scene first.")
            return
        if not self.material_rows:
            messagebox.showerror(
                "Error",
                "Load scene data first so the GUI can detect floor / ceiling / walls and write the .mat files.",
            )
            return

        missing_fit = [
            name for name, row in self.material_rows.items()
            if bool(row["fit_now"].get()) and not self.fit_exists(name)
        ]

        if missing_fit:
            messagebox.showerror(
                "Error",
                "These selected materials do not have .fit files yet:\n"
                + "\n".join(missing_fit)
                + "\n\nRun selected fittings first."
            )
            return

        try:
            self.status_var.set("Writing input files...")
            self.root.update_idletasks()

            self.write_config()
            self.write_src_rec()
            self.write_materials()

            solver = self.solver_dir()
            exe = "phases3D" if solver == "3D-solver" else "phases2D"
            exe_path = os.path.join(ROOT, solver, "build", exe)

            args = [
                exe_path,
                scene,
                os.path.join(ROOT, solver, "config", "config.txt"),
                os.path.join(ROOT, solver, "input", "src", "src.txt"),
                os.path.join(ROOT, solver, "input", "rec", "rec.txt"),
            ]

            self.status_var.set("Running solver...")
            self.root.update_idletasks()
            #proc = subprocess.run(args, capture_output=True, text=True)
            proc = subprocess.run(args, cwd=self.solver_root())

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Solver failed.\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
                )

            msg = "Simulation finished."
            self.status_var.set(msg)
            messagebox.showinfo("Done", msg)

            self.load_materials()

        except Exception as e:
            self.status_var.set("Error.")
            messagebox.showerror("Error", str(e))

    def plot_ir(self):
        ir_path = os.path.join(self.solver_root(), "output", "IR.bin")
        if not os.path.exists(ir_path):
            messagebox.showerror("Error", "IR.bin not found.")
            return

        import matplotlib.pyplot as plt

        data = np.fromfile(ir_path, dtype=np.float32)
        if data.size == 0:
            messagebox.showerror("Error", "IR.bin is empty.")
            return

        plt.figure()
        plt.plot(data, 'b')
        plt.xlabel("Sample [n]")
        plt.ylabel("Sound Pressure [Pa]")
        plt.title("Impulse Response")
        plt.grid(True, alpha=0.5)
        plt.show()

    def _fit_decay_rt60(self, t, edc_db, db_lo, db_hi):
        """
        Fit a straight line to the Schroeder decay curve between db_hi and db_lo [dB]
        and extrapolate to RT60.
        Example:
            EDT -> db_hi = 0,   db_lo = -10
            T20 -> db_hi = -5,  db_lo = -25
            T30 -> db_hi = -5,  db_lo = -35
        """
        mask = (edc_db <= db_hi) & (edc_db >= db_lo)

        if np.count_nonzero(mask) < 2:
            return np.nan

        tt = t[mask]
        yy = edc_db[mask]

        # Linear fit: yy = a * t + b
        a, b = np.polyfit(tt, yy, 1)

        # Physical decay must have negative slope
        if a >= 0:
            return np.nan

        return -60.0 / a

    def _compute_ir_parameters(self, ir, dt):
        """
        Compute broadband room-acoustic-style parameters from a single IR.
        Uses energy E = h^2 and Schroeder backward integration.
        """
        if ir.size < 4:
            raise ValueError("IR.bin is too short.")

        t = np.arange(ir.size, dtype=np.float64) * dt
        energy = ir.astype(np.float64) ** 2

        total_energy = np.sum(energy)
        if total_energy <= 0.0:
            raise ValueError("IR energy is zero.")

        # Schroeder curve
        edc = np.cumsum(energy[::-1])[::-1]

        if edc[0] <= 0.0:
            raise ValueError("Could not build valid Schroeder decay.")

        edc_db = 10.0 * np.log10(np.maximum(edc / edc[0], 1e-30))

        # Reverberation-related metrics
        edt = self._fit_decay_rt60(t, edc_db, db_lo=-10.0, db_hi=0.0)
        t20 = self._fit_decay_rt60(t, edc_db, db_lo=-25.0, db_hi=-5.0)
        t30 = self._fit_decay_rt60(t, edc_db, db_lo=-35.0, db_hi=-5.0)

        # Clarity / definition / center time
        n50 = int(round(0.050 / dt))
        n80 = int(round(0.080 / dt))

        n50 = max(0, min(n50, ir.size))
        n80 = max(0, min(n80, ir.size))

        early50 = np.sum(energy[:n50])
        late50  = np.sum(energy[n50:])
        early80 = np.sum(energy[:n80])
        late80  = np.sum(energy[n80:])

        eps = 1e-30
        c50 = 10.0 * np.log10((early50 + eps) / (late50 + eps))
        c80 = 10.0 * np.log10((early80 + eps) / (late80 + eps))
        d50 = 100.0 * early50 / (total_energy + eps)
        ts  = np.sum(t * energy) / (total_energy + eps)

        return {
            "N samples": ir.size,
            "dt [s]": dt,
            "EDT [s]": edt,
            "T20 [s]": t20,
            "T30 [s]": t30,
            "C50 [dB]": c50,
            "C80 [dB]": c80,
            "D50 [%]": d50,
            "Ts [s]": ts,
        }

    def compute_acoustic_params(self):
        ir_path = os.path.join(self.solver_root(), "output", "IR.bin")
        if not os.path.exists(ir_path):
            messagebox.showerror("Error", "IR.bin not found.")
            return

        try:
            ir = np.fromfile(ir_path, dtype=np.float32)

            if ir.size == 0:
                messagebox.showerror("Error", "IR.bin is empty.")
                return

            t_max = float(self.config_vars["t_max [s]"].get())
            if ir.size < 2 or t_max <= 0.0:
                messagebox.showerror("Error", "Invalid t_max or IR too short.")
                return

            # MVP choice: infer dt from GUI t_max and number of samples
            dt = t_max / float(ir.size - 1)

            params = self._compute_ir_parameters(ir, dt)

            def fmt(x, nd=4):
                if isinstance(x, (int, np.integer)):
                    return str(x)
                if isinstance(x, float) or isinstance(x, np.floating):
                    if np.isnan(x) or np.isinf(x):
                        return "not available"
                    return f"{x:.{nd}f}"
                return str(x)

            msg = (
                "Broadband parameters from output/IR.bin\n"
                "(single source-receiver IR, local values)\n\n"
                f"N samples: {fmt(params['N samples'])}\n"
                f"dt [s]:    {fmt(params['dt [s]'], 6)}\n\n"
                f"EDT [s]:   {fmt(params['EDT [s]'])}\n"
                f"T20 [s]:   {fmt(params['T20 [s]'])}\n"
                f"T30 [s]:   {fmt(params['T30 [s]'])}\n\n"
                f"C50 [dB]:  {fmt(params['C50 [dB]'])}\n"
                f"C80 [dB]:  {fmt(params['C80 [dB]'])}\n"
                f"D50 [%]:   {fmt(params['D50 [%]'])}\n"
                f"Ts [s]:    {fmt(params['Ts [s]'])}"
            )

            self.status_var.set("Acoustic parameters computed.")
            win = tk.Toplevel(self.root)
            win.title("Acoustic parameters")
            win.geometry("420x320")
            
            txt = tk.Text(win, wrap="word", height=20, width=50)
            txt.pack(fill="both", expand=True, padx=10, pady=10)
            txt.insert("1.0", msg)
            txt.config(state="disabled")

        except Exception as e:
            self.status_var.set("Error.")
            messagebox.showerror("Error", str(e))

    def plot_slices(self):
        solver = self.solver_dir()
        script = "plot3D.py" if solver == "3D-solver" else "plot.py"
        script_path = os.path.join(self.solver_root(), "scripts", script)

        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"{script} not found.")
            return

        anim_name = "anim3d_slices.mp4" if solver == "3D-solver" else "anim.mp4"
        anim_path = os.path.join(self.solver_root(), "output", anim_name)

        try:
            self.status_var.set("Generating animation...")
            self.root.update_idletasks()

            proc = subprocess.run(
                ["python3", script_path],
                cwd=self.solver_root(),
                capture_output=True,
                text=True,
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Failed to run {script}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
                )

            if os.path.exists(anim_path):
                self.status_var.set("Animation finished.")
                messagebox.showinfo("Done", f"Animation finished:\n{anim_path}")
            else:
                self.status_var.set("Plot script finished.")
                messagebox.showwarning(
                    "Done",
                    f"{script} finished, but the animation file was not found:\n{anim_path}"
                )

        except Exception as e:
            self.status_var.set("Error.")
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = SolverGUI(root)
    root.mainloop()

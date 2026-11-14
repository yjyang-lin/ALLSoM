import tkinter as tk
from tkinter import messagebox
import tkinter.font as tkfont
from io import BytesIO
import math

from PIL import Image, ImageTk
from rdkit import Chem
from rdkit.Chem import rdCoordGen
from rdkit.Chem.Draw import rdMolDraw2D


class RoundedButton(tk.Canvas):
    def __init__(self, parent, command, fill, outline, foreground, font, radius=10, padx=14, pady=7):
        self.command = command
        self.fill = fill
        self.outline = outline
        self.foreground = foreground
        self.font_spec = font
        self.radius = radius
        self.padx = padx
        self.pady = pady
        self.text = ""
        self.font = tkfont.Font(font=font)

        super().__init__(
            parent, bg=parent.cget("bg"), bd=0, highlightthickness=0,
            cursor="hand2", takefocus=1
        )

        self.bind("<Button-1>", self._invoke)
        self.bind("<Return>", self._invoke)
        self.bind("<space>", self._invoke)

    def _rounded_rectangle(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2, x2 - radius, y2,
            x1 + radius, y2, x1, y2, x1, y2 - radius,
            x1, y1 + radius, x1, y1
        ]
        return self.create_polygon(points, smooth=True, splinesteps=24, **kwargs)

    def _draw(self):
        width = self.font.measure(self.text) + self.padx * 2
        height = self.font.metrics("linespace") + self.pady * 2

        self.configure(width=width, height=height)
        self.delete("all")
        self._rounded_rectangle(
            1, 1, width - 1, height - 1, min(self.radius, height // 2),
            fill=self.fill, outline=self.outline, width=1
        )
        self.create_text(width / 2, height / 2, text=self.text, fill=self.foreground, font=self.font_spec)

    def _invoke(self, event=None):
        if self.command:
            self.command()

    def configure(self, cnf=None, **kwargs):
        if cnf:
            kwargs.update(cnf)
        if "text" in kwargs:
            self.text = kwargs.pop("text")
            self._draw()
        if kwargs:
            return super().configure(**kwargs)

    config = configure


class ClickableMolEditor:
    TEXT = {
        "zh": {
            "window_title": "RDKit 原子映射编辑器",
            "brand": "RDKit Atom Mapper",
            "clear": "清空",
            "copy": "复制 SMILES",
            "show_stereo": "显示 R/S",
            "hide_stereo": "隐藏 R/S",
            "language": "EN",
            "current_smiles": "当前 SMILES",
            "next_smiles": "下一个 SMILES",
            "load": "加载",
            "help": "操作说明",
            "help_title": "操作说明",
            "help_text": "点击分子中的原子可切换 :1 标记。\n清空：移除全部 :1 标记。\n复制 SMILES：复制当前结果。\n显示 R/S：显示或隐藏手性中心构型。\n输入下一个 SMILES 后点击加载，或按 Enter。",
            "close": "关闭",
            "ready": "点击原子切换 :1 标记",
            "enter_smiles": "请输入下一个分子的 SMILES",
            "parse_failed": "SMILES 解析失败，请检查输入",
            "loaded": "已加载分子",
            "copied": "SMILES 已复制到剪贴板",
            "stereo_on": "已显示 R/S 构型",
            "stereo_off": "已隐藏 R/S 构型",
            "missed": "未点中原子，请靠近原子中心",
            "marked": "已标记 atom {atom}",
            "unmarked": "已取消 atom {atom} 的标记",
            "cleared": "已清空全部标记",
            "error_title": "错误",
            "error_message": "SMILES 解析失败：\n{smiles}",
        },
        "en": {
            "window_title": "RDKit Atom Map Editor",
            "brand": "RDKit Atom Mapper",
            "clear": "Clear",
            "copy": "Copy SMILES",
            "show_stereo": "Show R/S",
            "hide_stereo": "Hide R/S",
            "language": "中文",
            "current_smiles": "Current SMILES",
            "next_smiles": "Next SMILES",
            "load": "Load",
            "help": "Help",
            "help_title": "How to use",
            "help_text": "Click an atom to toggle the :1 map.\nClear: remove all :1 marks.\nCopy SMILES: copy the current result.\nShow R/S: show or hide stereocenter configurations.\nEnter the next SMILES and click Load, or press Enter.",
            "close": "Close",
            "ready": "Click an atom to toggle the :1 map",
            "enter_smiles": "Enter the next molecule SMILES",
            "parse_failed": "SMILES parsing failed. Check the input",
            "loaded": "Molecule loaded",
            "copied": "SMILES copied to clipboard",
            "stereo_on": "R/S labels shown",
            "stereo_off": "R/S labels hidden",
            "missed": "No atom selected. Click closer to an atom center",
            "marked": "Marked atom {atom}",
            "unmarked": "Removed mark from atom {atom}",
            "cleared": "All marks cleared",
            "error_title": "Error",
            "error_message": "Failed to parse SMILES:\n{smiles}",
        },
    }

    def __init__(self, smiles, width=900, height=700, click_radius=22):
        self.original_smiles = smiles
        self.width = width
        self.height = height
        self.click_radius = click_radius

        self.mol = None
        self.atom_draw_coords = {}
        self.selected_atoms = set()
        self.show_stereo = False
        self.tk_img = None
        self.language = "en"
        self.status_key = "ready"
        self.status_args = {}
        self.resize_job = None

        self.bg = "#F1FAF3"
        self.surface = "#FFFFFF"
        self.text = "#17211A"
        self.muted = "#66736A"
        self.border = "#CFE4D4"
        self.accent = "#2F7D4A"
        self.accent_hover = "#276B3F"
        self.soft = "#EAF6ED"
        self.selection = "#F4A261"

        self.root = tk.Tk()
        self.root.geometry(f"{width + 100}x{height + 170}")
        self.root.minsize(760, 620)
        self.root.configure(bg=self.bg)

        self._build_ui()
        self._apply_texts()
        self.set_molecule(smiles)
        self.root.after(100, self.next_smiles_entry.focus_set)

    def _font(self, size=16, weight="normal"):
        return ("TkDefaultFont", size, weight)

    def _make_entry(self, parent, textvariable):
        return tk.Entry(
            parent, textvariable=textvariable, bg=self.surface, fg=self.text, insertbackground=self.text,
            font=self._font(), relief="flat", bd=0, highlightthickness=1,
            highlightbackground=self.border, highlightcolor=self.accent
        )

    def _make_action_button(self, parent, command):
        return RoundedButton(
            parent, command=command, fill=self.soft, outline=self.border,
            foreground=self.text, font=self._font(), radius=10, padx=14, pady=7
        )

    def _make_language_button(self, parent, command):
        return RoundedButton(
            parent, command=command, fill=self.surface, outline=self.border,
            foreground=self.text, font=self._font(), radius=10, padx=14, pady=7
        )

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = tk.Frame(self.root, bg=self.bg)
        main.grid(row=0, column=0, sticky="nsew", padx=24, pady=(18, 16))
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)

        top = tk.Frame(main, bg=self.bg)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        top.columnconfigure(0, weight=1)

        self.brand_label = tk.Label(top, bg=self.bg, fg=self.text, font=self._font(20, "bold"), anchor="w")
        self.brand_label.grid(row=0, column=0, sticky="w")

        self.language_button = self._make_language_button(top, self.toggle_language)
        self.language_button.grid(row=0, column=1, sticky="e")

        form = tk.Frame(main, bg=self.surface, highlightbackground=self.border, highlightthickness=1)
        form.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        form.columnconfigure(0, weight=1)

        form_inner = tk.Frame(form, bg=self.surface)
        form_inner.grid(row=0, column=0, sticky="ew", padx=16, pady=14)
        form_inner.columnconfigure(5, weight=1)

        self.current_label = tk.Label(form_inner, bg=self.surface, fg=self.text, font=self._font(weight="bold"), anchor="w")
        self.current_label.grid(row=0, column=0, sticky="w", padx=(0, 18), pady=(0, 8))

        self.clear_button = self._make_action_button(form_inner, self.clear_marks)
        self.clear_button.grid(row=0, column=1, sticky="w", padx=(0, 8), pady=(0, 8))

        self.copy_button = self._make_action_button(form_inner, self.copy_smiles)
        self.copy_button.grid(row=0, column=2, sticky="w", padx=(0, 8), pady=(0, 8))

        self.stereo_button = self._make_action_button(form_inner, self.toggle_stereo_labels)
        self.stereo_button.grid(row=0, column=3, sticky="w", padx=(0, 8), pady=(0, 8))

        self.help_button = self._make_action_button(form_inner, self.show_help)
        self.help_button.grid(row=0, column=4, sticky="w", pady=(0, 8))

        self.smiles_var = tk.StringVar()
        self.smiles_entry = self._make_entry(form_inner, self.smiles_var)
        self.smiles_entry.grid(row=1, column=0, columnspan=6, sticky="ew", ipady=7, pady=(0, 16))

        self.next_label = tk.Label(form_inner, bg=self.surface, fg=self.text, font=self._font(weight="bold"), anchor="w")
        self.next_label.grid(row=2, column=0, sticky="w", padx=(0, 18), pady=(0, 8))

        self.load_button = self._make_action_button(form_inner, self.load_next_smiles)
        self.load_button.grid(row=2, column=1, sticky="w", pady=(0, 8))

        self.next_smiles_var = tk.StringVar()
        self.next_smiles_entry = self._make_entry(form_inner, self.next_smiles_var)
        self.next_smiles_entry.grid(row=3, column=0, columnspan=6, sticky="ew", ipady=7)
        self.next_smiles_entry.bind("<Return>", lambda event: self.load_next_smiles())

        canvas_card = tk.Frame(main, bg=self.surface, highlightbackground=self.border, highlightthickness=1)
        canvas_card.grid(row=2, column=0, sticky="nsew")
        canvas_card.columnconfigure(0, weight=1)
        canvas_card.rowconfigure(0, weight=1)

        canvas_container = tk.Frame(canvas_card, bg=self.surface)
        canvas_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        canvas_container.columnconfigure(0, weight=1)
        canvas_container.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_container, bg=self.surface, highlightthickness=0, bd=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self.on_click)

        status = tk.Frame(main, bg=self.bg)
        status.grid(row=3, column=0, sticky="ew", pady=(9, 0))
        status.columnconfigure(1, weight=1)

        marker = tk.Frame(status, bg=self.accent, width=3, height=18)
        marker.grid(row=0, column=0, sticky="w", padx=(1, 8))
        marker.grid_propagate(False)

        self.tip_var = tk.StringVar()
        self.status_label = tk.Label(status, textvariable=self.tip_var, bg=self.bg, fg=self.muted,
                                     font=self._font(), anchor="w")
        self.status_label.grid(row=0, column=1, sticky="ew")

    def _t(self, key, **kwargs):
        return self.TEXT[self.language][key].format(**kwargs)

    def _set_status(self, key, **kwargs):
        self.status_key = key
        self.status_args = kwargs
        self.tip_var.set(self._t(key, **kwargs))

    def _apply_texts(self):
        self.root.title(self._t("window_title"))
        self.brand_label.configure(text=self._t("brand"))
        self.clear_button.configure(text=self._t("clear"))
        self.copy_button.configure(text=self._t("copy"))
        self.stereo_button.configure(text=self._t("hide_stereo") if self.show_stereo else self._t("show_stereo"))
        self.language_button.configure(text=self._t("language"))
        self.current_label.configure(text=self._t("current_smiles"))
        self.next_label.configure(text=self._t("next_smiles"))
        self.load_button.configure(text=self._t("load"))
        self.help_button.configure(text=self._t("help"))
        self.tip_var.set(self._t(self.status_key, **self.status_args))

    def toggle_language(self):
        self.language = "en" if self.language == "zh" else "zh"
        self._apply_texts()

    def show_help(self):
        window = tk.Toplevel(self.root)
        window.title(self._t("help_title"))
        window.configure(bg=self.bg)
        window.transient(self.root)
        window.resizable(False, False)

        card = tk.Frame(window, bg=self.surface, highlightbackground=self.border, highlightthickness=1)
        card.pack(fill="both", expand=True, padx=18, pady=18)

        tk.Label(card, text=self._t("help_title"), bg=self.surface, fg=self.text,
                 font=self._font(20, "bold"), anchor="w").pack(fill="x", padx=18, pady=(16, 10))

        tk.Label(card, text=self._t("help_text"), bg=self.surface, fg=self.text, font=self._font(),
                 justify="left", anchor="w", wraplength=560).pack(fill="x", padx=18, pady=(0, 16))

        close_button = self._make_action_button(card, window.destroy)
        close_button.configure(text=self._t("close"))
        close_button.pack(anchor="e", padx=18, pady=(0, 16))

        window.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - window.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - window.winfo_height()) // 2
        window.geometry(f"+{x}+{y}")

    def _build_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"SMILES parsing failed: {smiles}")
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        rdCoordGen.AddCoords(mol)
        return mol

    def set_molecule(self, smiles):
        mol = self._build_mol(smiles)
        self.original_smiles = smiles
        self.mol = mol
        self.atom_draw_coords = {}
        self.selected_atoms = set()
        self.redraw()
        self._set_status("ready")

    def load_next_smiles(self):
        new_smiles = self.next_smiles_var.get().strip()
        if not new_smiles:
            self._set_status("enter_smiles")
            self.next_smiles_entry.focus_set()
            return

        try:
            self.set_molecule(new_smiles)
        except Exception:
            messagebox.showerror(self._t("error_title"), self._t("error_message", smiles=new_smiles))
            self._set_status("parse_failed")
            self.next_smiles_entry.focus_set()
            self.next_smiles_entry.selection_range(0, tk.END)
            return

        self.next_smiles_var.set("")
        self._set_status("loaded")
        self.next_smiles_entry.focus_set()

    def get_output_smiles(self):
        try:
            return Chem.MolToSmiles(self.mol, canonical=False, kekuleSmiles=True, isomericSmiles=True)
        except Exception:
            return Chem.MolToSmiles(self.mol, canonical=False)

    def clear_marks(self):
        for atom in self.mol.GetAtoms():
            atom.SetAtomMapNum(0)
        self.selected_atoms.clear()
        self.redraw()
        self._set_status("cleared")

    def copy_smiles(self):
        s = self.get_output_smiles()
        self.root.clipboard_clear()
        self.root.clipboard_append(s)
        self._set_status("copied")

    def toggle_stereo_labels(self):
        self.show_stereo = not self.show_stereo
        self.stereo_button.configure(text=self._t("hide_stereo") if self.show_stereo else self._t("show_stereo"))
        self._set_status("stereo_on" if self.show_stereo else "stereo_off")
        self.redraw()

    def redraw(self):
        # Use RDKit for atom indices and stereochemical notes so annotations participate in layout.
        self.root.update_idletasks()
        w = max(self.canvas.winfo_width(), 300)
        h = max(self.canvas.winfo_height(), 300)

        Chem.AssignStereochemistry(self.mol, force=True, cleanIt=True)
        for atom in self.mol.GetAtoms():
            if atom.HasProp("atomNote"):
                atom.ClearProp("atomNote")
            if self.show_stereo and atom.HasProp("_CIPCode"):
                atom.SetProp("atomNote", atom.GetProp("_CIPCode"))

        drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
        opts = drawer.drawOptions()
        opts.addAtomIndices = True
        opts.centreMoleculesBeforeDrawing = True
        opts.padding = 0.10
        opts.annotationFontScale = 0.80
        opts.additionalAtomLabelPadding = 0.12
        opts.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
        opts.setAnnotationColour((0.184, 0.490, 0.290, 1.0))

        for atom in self.mol.GetAtoms():
            idx = atom.GetIdx()
            if atom.GetAtomMapNum() == 1:
                opts.atomLabels[idx] = f"{atom.GetSymbol()}(1)"

        highlight_atoms = sorted(self.selected_atoms)
        highlight_atom_colors = {idx: (0.957, 0.635, 0.380) for idx in highlight_atoms}
        drawer.DrawMolecule(self.mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors)

        self.atom_draw_coords = {}
        for i in range(self.mol.GetNumAtoms()):
            p = drawer.GetDrawCoords(i)
            self.atom_draw_coords[i] = (p.x, p.y)

        drawer.FinishDrawing()
        image = Image.open(BytesIO(drawer.GetDrawingText()))
        self.tk_img = ImageTk.PhotoImage(image)

        self.canvas.delete("all")
        self.canvas.create_image(w / 2, h / 2, anchor="center", image=self.tk_img)
        self.smiles_var.set(self.get_output_smiles())

    def find_nearest_atom(self, x, y):
        best_idx = None
        best_dist = 1e9

        for idx, (ax, ay) in self.atom_draw_coords.items():
            d = math.hypot(x - ax, y - ay)
            if d < best_dist:
                best_dist = d
                best_idx = idx

        if best_dist <= self.click_radius:
            return best_idx
        return None

    def on_click(self, event):
        atom_idx = self.find_nearest_atom(event.x, event.y)
        if atom_idx is None:
            self._set_status("missed")
            return

        atom = self.mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomMapNum() == 1:
            atom.SetAtomMapNum(0)
            self.selected_atoms.discard(atom_idx)
            self._set_status("unmarked", atom=atom_idx)
        else:
            atom.SetAtomMapNum(1)
            self.selected_atoms.add(atom_idx)
            self._set_status("marked", atom=atom_idx)

        self.redraw()

    def run(self):
        self.root.bind("<Configure>", self._on_resize)
        self.root.mainloop()

    def _on_resize(self, event):
        if event.widget == self.root and self.mol is not None:
            if self.resize_job is not None:
                self.root.after_cancel(self.resize_job)
            self.resize_job = self.root.after(80, self.redraw)


if __name__ == "__main__":
    smiles = "C[C@H]1[C@H]([C@H](C[C@@H](O1)O[C@H]2C[C@@](CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O"
    app = ClickableMolEditor(smiles)
    app.run()

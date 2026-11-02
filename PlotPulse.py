#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot acceleration spectra exported by Brüel & Kjær PULSE.

These files already contain frequency-domain data: this script does NOT apply
another FFT. It reads the header and indexed numeric rows, converts acceleration
to cm/s², marks each spectrum's global peak, and exports a PNG and clean CSVs.

Run from the repository root:
    python PlotPulse.py examples/pulse_baseline.txt examples/pulse_with_tmd.txt

With no arguments, the script uses ``rutas_fijas`` in main(), if configured,
or searches the CURRENT WORKING DIRECTORY for *.txt files. Explicit command-line
paths take priority. Edit CFG below to change plotting and export settings.

Comments and documentation are in English; plot labels remain in Spanish to
match the thesis. See README.md and CHANGELOG.md for scientific assumptions and
the changes made to the supplied research script.
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


# USER SETTINGS: edit VALUES here; keep the existing dictionary keys.
CFG = {
    # None builds a title from the first file's signal/date metadata.
    "titulo": "Comparativa de amortiguadores de masa sintonizada",
    # Values are ALWAYS converted to cm/s², including when this label is None.
    "etiq_y": "Aceleración [cm/s²]",
    "escala_y_log": False,
    # Display limits only: peak detection still uses the ENTIRE input spectrum.
    "rango_x": None,              # Example: [0, 25], in Hz.
    "rango_y": [0, 40],           # In cm/s²; None lets Matplotlib choose.
    "guardar_png": True,
    "guardar_csv": True,
    "dpi": 150,
    # USER PATH: None writes beside each input (PNG beside the first input).
    # Example: "outputs". Missing output directories are created automatically.
    "carpeta_salida": None,
    "nombre_png": "comparison.png",
    # USER REFERENCE: for a thesis-style comparison, set the exact filename of
    # the undamped measurement, e.g. "pulse_baseline.txt", and include that file
    # in the inputs. A unique full input path is also accepted.
    # None preserves the original behavior: use the largest loaded peak.
    "archivo_referencia": None,
}


def _float_p(s: str) -> float:
    """Parse a PULSE number with either a comma or a point as decimal mark."""
    return float(s.strip().replace(",", "."))


def _es_fila_de_datos(partes: list) -> bool:
    """Accept exactly three fields: a positive integer index, Hz, and value."""
    if len(partes) != 3:
        return False
    try:
        if int(partes[0]) < 1:
            return False
        _float_p(partes[1])
        _float_p(partes[2])
        return True
    except ValueError:
        return False


def _limpiar_unidad(s: str) -> str:
    """Normalize supported acceleration units without losing their SI prefix.

    Latin-1 PULSE exports may contain a superscript 2; a UTF-8 superscript read
    as Latin-1 may additionally contain 'Â'. Unknown units remain unchanged so
    the conversion function can reject them instead of inventing a scale.
    """
    key = s.strip().lower().replace("Â", "").replace("â", "")
    key = key.replace("²", "2").replace("**", "").replace("^", "")
    key = "".join(key.split())
    aliases = {
        "m/s2": "m/s²",
        "cm/s2": "cm/s²",
        "mm/s2": "mm/s²",
        "g": "g", "g's": "g", "gs": "g",
    }
    return aliases.get(key, s.strip())


def convertir_a_cm_s2(valor, unidad: str):
    """Convert a scalar or NumPy array to cm/s² using a recognized unit.

    The original approximate gravity factor, 1 g ≈ 981 cm/s², is retained.
    A velocity, PSD, missing unit, or other unrecognized quantity is rejected.
    """
    factors = {"cm/s²": 1.0, "mm/s²": 0.1, "m/s²": 100.0, "g": 981.0}
    unit = _limpiar_unidad(unidad)
    if unit not in factors:
        raise ValueError(
            f"Unidad de aceleración no reconocida: {unidad!r}. "
            "Se admiten m/s², cm/s², mm/s² y g; revisa la exportación de PULSE."
        )
    return valor * factors[unit]


def leer_pulse(path: Path) -> dict:
    """Read one supported PULSE text export and return arrays plus metadata.

    ``freq`` is in Hz and ``vals`` retains the original acceleration unit.
    Conversion happens separately in graficar(), never during parsing.
    """
    lineas = path.read_text(encoding="latin-1").splitlines()
    if not lineas:
        raise ValueError(f"El archivo '{path.name}' está vacío.")

    match = re.match(r"Header Size:\s+(\d+)", lineas[0])
    hs = int(match.group(1)) if match else 0
    if hs > len(lineas):
        raise ValueError("Header Size supera el número de líneas del archivo.")

    # Header Size counts the header lines, INCLUDING the Header Size line.
    meta = {}
    for linea in lineas[:hs]:
        match = re.match(r"^([^:]+):\s+(.+)", linea)
        if match:
            meta[match.group(1).strip()] = match.group(2).strip()

    date_str = time_str = avg = None
    for linea in lineas[hs:]:
        if not date_str:
            match = re.match(r"Date:\s+([\d/]+)", linea)
            if match:
                date_str = match.group(1).strip()
        if not time_str:
            match = re.match(r"Time:\s+([\d:]+)", linea)
            if match:
                time_str = match.group(1).strip()
        if not avg:
            match = re.match(r"AverageNumber.*?([\d,\.]+)\s*$", linea)
            if match:
                avg = match.group(1).strip().replace(",", ".")

    freqs, vals = [], []
    for linea in lineas[hs:]:
        partes = linea.split()  # Tabs and runs of spaces are both accepted.
        if _es_fila_de_datos(partes):
            freqs.append(_float_p(partes[1]))
            vals.append(_float_p(partes[2]))
    if not freqs:
        raise ValueError(f"No se encontraron datos numéricos en '{path.name}'.")

    freq = np.array(freqs)
    values = np.array(vals)
    if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(values)):
        raise ValueError("El espectro contiene NaN o valores infinitos.")
    if np.any(freq < 0) or np.any(np.diff(freq) <= 0):
        raise ValueError("Las frecuencias deben ser no negativas y crecientes.")
    if np.any(values < 0):
        raise ValueError("Se requieren magnitudes de aceleración no negativas.")

    unit_raw = meta.get("SignalUnit", meta.get("PowerUnit", ""))
    return {
        "freq": freq,
        "vals": values,
        "signal": meta.get("Signal", path.stem),
        "unit": _limpiar_unidad(unit_raw),
        "date": date_str or "–",
        "time": time_str or "–",
        "avg": avg or "–",
    }


def guardar_csv(datos: dict, carpeta: Path) -> Path:
    """Save frequency and converted acceleration, without the PULSE header."""
    out = carpeta / (datos["path"].stem + ".clean.csv")
    np.savetxt(
        out,
        np.column_stack([datos["freq"], datos["vals_cm"]]),
        delimiter=",",
        header="Frecuencia_Hz,Aceleracion_cm_s2",
        comments="",
        fmt="%.6e",
    )
    return out


def calcular_reducciones(picos, referencia=None):
    """Return peak reductions (%) against a chosen index or the largest peak.

    A zero reference makes a relative reduction undefined: return NaNs so the
    table can display N/D. Negative percentages mean the peak exceeds a chosen
    reference. This compares GLOBAL peaks, which may occur at different Hz.
    """
    picos = np.asarray(picos, dtype=float)
    ref = np.max(picos) if referencia is None else picos[referencia]
    if ref == 0:
        return np.full(picos.shape, np.nan)
    return (ref - picos) / ref * 100.0


COLORES = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
           "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def graficar(rutas: list, cfg: dict):
    """Load spectra, plot their peaks and summary table, and export results.

    Individual invalid inputs are reported and skipped. An explicitly selected
    reference must be loaded successfully; it is never silently substituted.
    """
    todos = []
    for i, ruta in enumerate(rutas):
        try:
            d = leer_pulse(ruta)
            d["vals_cm"] = convertir_a_cm_s2(d["vals"], d["unit"])
        except (OSError, ValueError) as exc:
            print(f"Aviso: {ruta.name}: {exc}", file=sys.stderr)
            continue
        d["path"] = ruta
        d["color"] = COLORES[i % len(COLORES)]
        todos.append(d)
    if not todos:
        raise ValueError("No se pudo cargar ningún espectro de aceleración válido.")

    # An explicit reference is selected by exact basename or resolved full path.
    ref_name = cfg.get("archivo_referencia")
    ref_index = None
    if ref_name:
        matches = [
            i for i, d in enumerate(todos)
            if d["path"].name == str(ref_name)
            or d["path"].resolve() == Path(ref_name).resolve()
        ]
        if len(matches) != 1:
            raise ValueError(
                "La referencia debe coincidir con un único archivo cargado: "
                f"{ref_name!r}."
            )
        ref_index = matches[0]

    n_curvas = len(todos)
    fig = plt.figure(figsize=(14, 7 + 0.45 * n_curvas))
    gs = GridSpec(2, 1, figure=fig,
                  height_ratios=[6, 0.55 + 0.45 * n_curvas], hspace=0.55)
    ax = fig.add_subplot(gs[0])
    ax_tab = fig.add_subplot(gs[1])
    ax_tab.axis("off")

    for d in todos:
        f, v = d["freq"], d["vals_cm"]
        idx_max = np.argmax(v)
        ax.plot(f, v, color=d["color"], linewidth=1.7,
                label=d["path"].stem, zorder=3)
        x_off = (f.max() - f.min()) * 0.025
        ax.annotate(
            f"  {f[idx_max]:.3g} Hz\n  {v[idx_max]:.2f} cm/s²",
            xy=(f[idx_max], v[idx_max]),
            xytext=(f[idx_max] + x_off, v[idx_max]),
            fontsize=8.5, color=d["color"],
            arrowprops=dict(arrowstyle="->", color=d["color"], lw=1.1),
        )

        carpeta = Path(cfg["carpeta_salida"]) if cfg["carpeta_salida"] else d["path"].parent
        if cfg["guardar_csv"]:
            carpeta.mkdir(parents=True, exist_ok=True)
            print(f"CSV: {guardar_csv(d, carpeta)}")
        print(f"\nArchivo: {d['path'].name}")
        print(f"Señal: {d['signal']} | Fecha/hora: {d['date']} {d['time']}")
        print(f"Promedios: {d['avg']} | Unidad original: {d['unit']}")
        print(f"Puntos: {len(f)} | Frecuencias: {f.min():.3g}–{f.max():.3g} Hz")
        print(f"Pico: {v[idx_max]:.2f} cm/s² @ {f[idx_max]:.4g} Hz")

    title = cfg["titulo"] or (
        f"{todos[0]['signal']} – {todos[0]['date']}"
        if n_curvas == 1 else "Comparativa PULSE – TMD"
    )
    if cfg["escala_y_log"]:
        ax.set_yscale("log")
    if cfg["rango_x"]:
        ax.set_xlim(cfg["rango_x"])
    if cfg["rango_y"]:
        ax.set_ylim(cfg["rango_y"])
    ax.set_xlabel("Frecuencia [Hz]", fontsize=13)
    ax.set_ylabel(cfg["etiq_y"] or "Aceleración [cm/s²]", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.55)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.3, alpha=0.3)
    ax.minorticks_on()
    if n_curvas > 1:
        ax.legend(fontsize=10, framealpha=0.9)

    picos = [d["vals_cm"].max() for d in todos]
    reductions = calcular_reducciones(picos, ref_index)
    col_labels = ["Archivo", "Frec. pico [Hz]", "Aceleración máx. [cm/s²]"]
    if n_curvas > 1:
        col_labels.append("Reducción vs. referencia [%]" if ref_name
                          else "Reducción resp. al máx. [%]")
        ref_desc = todos[ref_index]["path"].name if ref_name else "mayor pico cargado"
        print(f"Referencia de reducción: {ref_desc}")
    table_data = []
    for d, pico, reduccion in zip(todos, picos, reductions):
        idx_max = np.argmax(d["vals_cm"])
        fila = [d["path"].stem, f"{d['freq'][idx_max]:.3f}", f"{pico:.2f}"]
        if n_curvas > 1:
            fila.append(f"{reduccion:.1f} %" if np.isfinite(reduccion) else "N/D")
        table_data.append(fila)

    tabla = ax_tab.table(cellText=table_data, colLabels=col_labels,
                         loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.auto_set_column_width(col=list(range(len(col_labels))))
    tabla.scale(1, 1.6)
    for (row, col), cell in tabla.get_celld().items():
        cell.set_edgecolor("#b0b8c1")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#1a4f7a")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#eef3f9" if row % 2 == 0 else "#ffffff")
    for i, d in enumerate(todos):
        cell = tabla[(i + 1, 0)]
        cell.set_facecolor(d["color"])
        cell.set_text_props(color="white", fontweight="bold")

    if cfg["guardar_png"]:
        carpeta = Path(cfg["carpeta_salida"]) if cfg["carpeta_salida"] else rutas[0].parent
        carpeta.mkdir(parents=True, exist_ok=True)
        png_path = carpeta / cfg["nombre_png"]
        fig.savefig(png_path, dpi=cfg["dpi"], bbox_inches="tight")
        print(f"PNG: {png_path}")
    # For unattended runs, set MPLBACKEND=Agg before launching this script.
    plt.show()
    plt.close(fig)


def main():
    # USER PATHS (OPTIONAL): for an IDE's Run button, add Path(...) entries.
    # Leave this list empty when supplying files on the command line.
    # Windows example: Path(r"C:\measurements\sin_tmd.txt").
    rutas_fijas = []

    if len(sys.argv) > 1:
        rutas = [Path(a) for a in sys.argv[1:]]
    elif rutas_fijas:
        rutas = rutas_fijas
    else:
        rutas = sorted(Path(".").glob("*.txt"))
        if not rutas:
            print("No hay archivos .txt en el directorio actual.\n"
                  "Uso: python PlotPulse.py archivo1.txt [archivo2.txt ...]",
                  file=sys.stderr)
            return 1

    rutas_ok = [r for r in rutas if r.is_file()]
    for ruta in (r for r in rutas if not r.is_file()):
        print(f"Aviso: archivo no encontrado: {ruta}", file=sys.stderr)
    if not rutas_ok:
        return 1
    try:
        graficar(rutas_ok, cfg=CFG)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

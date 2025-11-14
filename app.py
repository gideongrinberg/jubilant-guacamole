import numpy as np
import polars as pl
from bokeh.io import curdoc
from bokeh.events import Tap
from bokeh.plotting import figure
from bokeh.palettes import Viridis256
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, Div

df = pl.read_parquet("./data/results.parquet")
freq = np.load("./data/frequency.npy")
lightcurves = pl.scan_parquet("./data/lightcurves.parquet")
periodograms = pl.scan_parquet("./data/periodograms.parquet")

def get_lightcurve(tic, sector):
    lc = (
        lightcurves
        .filter((pl.col("tic") == pl.lit(int(tic))) & (pl.col("sector") == pl.lit(int(sector))))
        .select(["time", "flux"])
        .collect()
    )
    return lc.get_column("time").to_numpy(), lc.get_column("flux").to_numpy()

def get_periodogram(tic, sector):
    pg = (
        periodograms
        .filter((pl.col("tic") == pl.lit(tic)) & (pl.col("sector") == pl.lit(sector)))
        .select(["power"])
        .collect()
    )
    return freq, pg.get_column("power").to_numpy()

tic_arr = df["tic_id"].to_numpy()
sec_arr = df["sector"].to_numpy()

source_pca = ColumnDataSource({
    "tic": tic_arr,
    "sector": sec_arr,
    "p_sub": df["p_subalfvenic"].to_numpy(),
    "x": df["PC1"].to_numpy(),
    "y": df["PC2"].to_numpy()
})

source_lc = ColumnDataSource(data=dict(time=[], flux=[]))
source_flc = ColumnDataSource(data=dict(phase=[], flux=[]))
source_pg = ColumnDataSource(data=dict(freq=[], power=[]))

info = Div()

cmap = LinearColorMapper(
    palette=Viridis256,
    low=min(source_pca.data["p_sub"]),
    high=max(source_pca.data["p_sub"]),
)

pca_dim = 600
fig_pca = figure(width=pca_dim, height=pca_dim)
fig_lc = figure(width=pca_dim, height=pca_dim//3)
fig_flc = figure(width=pca_dim, height=pca_dim//3)
fig_pg = figure(width=pca_dim, height=pca_dim//3)

x, y, c = df["PC1"], df["PC2"], df["p_subalfvenic"]
fig_pca.scatter(source=source_pca, x="x", y="y", fill_color={"field": "p_sub", "transform": cmap}, line_color=None, size=3)
cbar = ColorBar(color_mapper=cmap, label_standoff=8, location=(0, 0))
fig_pca.add_layout(cbar, "right")

# Circle for clicks
sel_source = ColumnDataSource(dict(x=[], y=[]))
fig_pca.circle("x", "y", source=sel_source, size=10, fill_color=None, line_color="black", line_width=2)

# Highlight same tic
source_highlight = ColumnDataSource(dict(x=[], y=[]))
fig_pca.circle("x", "y", source=source_highlight, size=6, fill_color="red", line_color="black", line_width=1)

fig_lc.scatter(source=source_lc, x="x", y="y", size=3)
fig_flc.scatter(source=source_flc, x="x", y="y", size=3)
fig_pg.line(source=source_pg, x="x", y="y", width=1)

def tap_callback(event):
    x_click, y_click = event.x, event.y
    if not (fig_pca.x_range.start <= event.x <= fig_pca.x_range.end and fig_pca.y_range.start <= event.y <= fig_pca.y_range.end):
        source_pg.data = dict(x=[], y=[])
        source_lc.data = dict(x=[], y=[])
        source_flc.data = dict(x=[], y=[])
        info.text = ""
        source_highlight.data = dict(x=[], y=[])
        source_pca.selected.indices = []
        sel_source.data = dict(x=[], y=[])
        return

    xs = np.array(source_pca.data["x"])
    ys = np.array(source_pca.data["y"])
    tics = np.array(source_pca.data["tic"])
    d2 = (xs - x_click)**2 + (ys - y_click)**2
    i = int(np.argmin(d2))
    source_pca.selected.indices = [i]
    sel_source.data = dict(x=[source_pca.data["x"][i]], y=[source_pca.data["y"][i]])

    tic = source_pca.data["tic"][i]
    sector = source_pca.data["sector"][i]
    info.text = f"<strong>Currently selected star:</strong> TIC {tic}, sector {sector}"
    info.text += f"<br>View on: <a href=\"https://simbad.cfa.harvard.edu/simbad/sim-basic?Ident=TIC+{tic}&submit=SIMBAD+search\">Simbad</a> &bullet; <a href=\"https://exofop.ipac.caltech.edu/tess/target.php?id={tic}\">ExoFOP</a>"
    mask = (tics == tic)
    source_highlight.data = dict(x=xs[mask], y=ys[mask])

    freq, power = get_periodogram(tic, sector)
    source_pg.data = dict(x=freq, y=power)
    time, flux = get_lightcurve(tic, sector)
    source_lc.data = dict(x=time, y=flux)
    P = 1/freq[np.argmax(power)]
    phase = ((time - time[0]) / np.max(P)) % 1.0
    order = np.argsort(phase)
    source_flc.data = dict(x=phase[order], y=flux[order])



fig_pca.on_event(Tap, tap_callback)
title = Div(text="<h2>PCA Lightcurve Explorer</h2>")
blurb = Div(text="""
<p>Click in the PCA plot to load the corresponding periodogram and lightcurve. Click on the color bar or axes to unselect the point. Red points indicate other observations of the selected star.</p>
""")
curdoc().add_root(column(
    title,
    row(fig_pca, column(fig_pg, fig_lc, fig_flc)),
    blurb,
    info,
    sizing_mode="fixed",
    align="center"
))
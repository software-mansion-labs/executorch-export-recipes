import json
import pandas as pd
import plotly.offline as plo
import plotly.graph_objects as go

from typing import Any
from pathlib import Path
from string import Template
from itertools import permutations

_HTML_TEMPLATE = Template(
    """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>$title</title>
  <script type="text/javascript">$plotlyjs</script>
  <style>
    body { font-family: sans-serif; padding: 16px; }
    .controls { display: flex; gap: 24px; align-items: center; margin-bottom: 12px; }
    label { font-size: 14px; }
    select { padding: 4px 8px; font-size: 14px; margin-left: 6px; }
  </style>
</head>
<body>
  <h2>$title</h2>
  <div class="controls">
    <label>X-axis: <select id="x-sel" onchange="update()">$x_options</select></label>
    <label>Y-axis: <select id="y-sel" onchange="update()">$y_options</select></label>
    <span id="error" style="color: red; font-size: 14px;"></span>
  </div>
  <div id="plot" style="width:100%;height:600px;"></div>
  <script>
    const figures = $figures;
    function update() {
      const x = document.getElementById('x-sel').value;
      const y = document.getElementById('y-sel').value;
      if (x === y) { document.getElementById('error').textContent = 'X and Y axes must be different metrics.'; return; }
      document.getElementById('error').textContent = '';
      Plotly.react('plot', figures[x][y].data, figures[x][y].layout);
    }
    update();
  </script>
</body>
</html>"""
)


def _pareto_front(xs: list, ys: list, max_x: bool, max_y: bool) -> list[list]:
    pts = sorted(zip(xs, ys), reverse=max_x)
    front = [list(pts[0])]
    for x, y in pts[1:]:
        if (max_y and y >= front[-1][1]) or (not max_y and y <= front[-1][1]):
            front.append([x, y])
    return front


def create_plot(df: pd.DataFrame, metric_registry: dict[str, bool], save_path: Path, title: str = "") -> None:
    metrics = list(metric_registry.keys())
    marker_sizes = (10 + df["size_mb"] / df["size_mb"].max() * 40).tolist()
    hover = [
        "<b>{}</b><br>{}".format(r["name"], "<br>".join(f"{c}: {r[c]:.4g}" for c in metrics if pd.notna(r[c])))
        for _, r in df.iterrows()
    ]
    opts = lambda ms, d: "".join(f'<option value="{m}"{"selected" if m==d else ""}>{m}</option>' for m in ms)

    fig_data: dict[str, dict[str, Any]] = {m: {} for m in metrics}
    for x, y in permutations(metrics, 2):
        pareto = _pareto_front(df[x].tolist(), df[y].tolist(), metric_registry[x], metric_registry[y])
        fig = go.Figure(
            [
                go.Scatter(
                    x=df[x],
                    y=df[y],
                    mode="markers+text",
                    text=df["name"],
                    textposition="top center",
                    marker={"size": marker_sizes},
                    hovertext=hover,
                    hoverinfo="text",
                ),
                go.Scatter(
                    x=[p[0] for p in pareto],
                    y=[p[1] for p in pareto],
                    mode="lines",
                    hoverinfo="skip",
                    line={"dash": "dot", "color": "red"},
                ),
            ]
        )
        fig.update_layout(
            title=title,
            showlegend=False,
            xaxis_title=x + (" ↑" if metric_registry[x] else " ↓"),
            yaxis_title=y + (" ↑" if metric_registry[y] else " ↓"),
        )
        fig_data[x][y] = json.loads(fig.to_json())  # type: ignore[arg-type]

    save_path.write_text(
        _HTML_TEMPLATE.substitute(
            title=title,
            plotlyjs=plo.get_plotlyjs(),
            x_options=opts(metrics, metrics[0]),
            y_options=opts(metrics, metrics[1]),
            figures=json.dumps(fig_data),
        )
    )

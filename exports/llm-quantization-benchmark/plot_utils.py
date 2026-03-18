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
      if (x === y) {
        document.getElementById('error').textContent = 'X and Y axes must be different metrics.';
        return;
      }
      document.getElementById('error').textContent = '';
      Plotly.react('plot', figures[x][y].data, figures[x][y].layout);
    }

    update();
  </script>
</body>
</html>"""
)


def _pareto_front(xs: list, ys: list, max_x: bool = True, max_y: bool = True) -> list[list]:
    pts = sorted(zip(xs, ys), reverse=max_x)
    front = [list(pts[0])]
    for x, y in pts[1:]:
        if (max_y and y >= front[-1][1]) or (not max_y and y <= front[-1][1]):
            front.append([x, y])
    return front


def _axis_label(metric: str, higher_is_better: bool) -> str:
    return metric + (" ↑" if higher_is_better else " ↓")


def _options_html(metrics: list[str], default: str) -> str:
    return "".join(f'<option value="{m}"{"selected" if m == default else ""}>{m}</option>' for m in metrics)


def _hover_text(df: pd.DataFrame) -> list[str]:
    metric_cols = [c for c in df.columns if c != "name"]

    def _fmt_row(row: pd.Series) -> str:
        metrics = "<br>".join(f"{c}: {row[c]:.4g}" for c in metric_cols if pd.notna(row[c]))
        return f"<b>{row['name']}</b><br>{metrics}"

    return [_fmt_row(row) for _, row in df.iterrows()]


def create_plot(
    df: pd.DataFrame,
    metric_registry: dict[str, bool],
    save_path: Path,
    title: str = "",
) -> None:
    metrics = list(metric_registry.keys())
    marker_sizes = (10 + df["size_mb"] / df["size_mb"].max() * 40).tolist()

    fig_data: dict[str, dict[str, Any]] = {m: {} for m in metrics}
    for x, y in permutations(metrics, 2):
        pareto = _pareto_front(
            df[x].tolist(),
            df[y].tolist(),
            max_x=metric_registry[x],
            max_y=metric_registry[y],
        )
        fig = go.Figure(
            [
                go.Scatter(
                    x=df[x],
                    y=df[y],
                    mode="markers+text",
                    text=df["name"],
                    textposition="top center",
                    marker={"size": marker_sizes},
                    hovertext=_hover_text(df),
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
            xaxis_title=_axis_label(x, metric_registry[x]),
            yaxis_title=_axis_label(y, metric_registry[y]),
            showlegend=False,
        )
        fig_data[x][y] = json.loads(fig.to_json())  # type: ignore[arg-type]

    save_path.write_text(
        _HTML_TEMPLATE.substitute(
            title=title,
            plotlyjs=plo.get_plotlyjs(),
            x_options=_options_html(metrics, metrics[0]),
            y_options=_options_html(metrics, metrics[1]),
            figures=json.dumps(fig_data),
        )
    )


__all__ = ["create_plot"]

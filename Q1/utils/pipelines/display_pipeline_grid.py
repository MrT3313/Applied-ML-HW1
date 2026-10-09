from html import escape
from IPython.display import HTML, display
from sklearn.utils import estimator_html_repr

def display_pipeline_grid(pipelines: dict, columns: int = 3) -> None:
    cells = "".join(
        f"""
        <div style="border:1px solid #ccc; border-radius:6px; padding:8px; overflow:auto;">
          <div style="font-weight:600; margin-bottom:6px;">{escape(f"{name} | {preprocessing}")}</div>
          {estimator_html_repr(pipeline)}
        </div>
        """
        for (name, preprocessing), pipeline in pipelines.items()
    )
    display(HTML(
        f'<div style="display:grid; grid-template-columns:repeat({columns}, minmax(0, 1fr)); gap:12px;">'
        f"{cells}</div>"
    ))
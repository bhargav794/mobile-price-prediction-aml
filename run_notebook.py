"""Execute from a clean kernel, export offline HTML, and audit notebook outputs."""
import json
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
os.environ.setdefault('PYTHONHASHSEED', '42')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('IPYTHONDIR', str(ROOT / '.ipython'))
os.environ.setdefault('JUPYTER_RUNTIME_DIR', str(ROOT / '.jupyter/runtime'))
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(parents=True, exist_ok=True)


def execute_export(path):
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=900, resources={'metadata': {'path': str(ROOT)}},
                            allow_errors=False, kernel_name='python3')
    # Explicit interpreter avoids accidentally using the system Jupyter Python.
    from jupyter_client.kernelspec import KernelSpecManager
    kernel_root = ROOT / '.jupyter/kernels/aml'
    kernel_root.mkdir(parents=True, exist_ok=True)
    (kernel_root / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'AML assignment', 'language': 'python'}))
    client.kernel_name = 'aml'
    client.kernel_manager_class = __import__('jupyter_client').AsyncKernelManager
    client.create_kernel_manager()
    client.km.kernel_spec_manager = KernelSpecManager(kernel_dirs=[str(kernel_root.parent)])
    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            print(f'Executing cell {cell_index + 1}/{len(nb.cells)}', flush=True)
    client.on_cell_start = progress
    client.execute()
    nbformat.write(nb, path)
    exporter = HTMLExporter(template_name='lab')
    exporter.embed_images = True
    body, _ = exporter.from_notebook_node(nb)
    # nbconvert embeds plots/CSS; remove external MathJax/CDN script dependencies.
    import re
    body = re.sub(r'<script\b[^>]*\bsrc=["\']https?://[^>]*>\s*</script>', '', body, flags=re.I)
    html = path.with_suffix('.html')
    html.write_text(body, encoding='utf-8')
    codes = [c for c in nb.cells if c.cell_type == 'code']
    errors = [o for c in codes for o in c.get('outputs', []) if o.output_type == 'error']
    assert not errors and all(c.execution_count is not None for c in codes)
    images = sum('image/png' in o.get('data', {}) for c in codes for o in c.get('outputs', []))
    tables = sum('text/html' in o.get('data', {}) for c in codes for o in c.get('outputs', []))
    assert images >= 8 and tables >= 15
    audit = {'executed_code_cells': len(codes), 'error_outputs': len(errors),
             'embedded_figures': images, 'html_outputs': tables, 'html_bytes': html.stat().st_size,
             'html_image_data_uris': body.count('data:image/png;base64'),
             'notebook': path.name, 'html': html.name, 'python': sys.version.split()[0]}
    (ROOT / 'reports/execution_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', nargs='?', default='submission/GROUP_NO.ipynb')
    args = parser.parse_args()
    path = Path(args.notebook)
    execute_export(path.resolve() if path.is_absolute() else ROOT / path)

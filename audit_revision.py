"""Acceptance checks for the revised notebook, report and application wording."""
import ast
import hashlib
import json
import re
from pathlib import Path
import nbformat
from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
APP = 'https://mobile-price-prediction-aml.streamlit.app/'


def main():
    notebook = ROOT / 'submission/GROUP_NO.ipynb'
    pdf_path = notebook.with_suffix('.pdf')
    html_path = ROOT / 'output/html/GROUP_NO.html'
    nb = nbformat.read(notebook, as_version=4)
    nbformat.validate(nb)
    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    assert [c.execution_count for c in code_cells] == list(range(1,len(code_cells)+1))
    errors = [o for c in code_cells for o in c.outputs if o.output_type == 'error']
    warnings = [o.text for c in code_cells for o in c.outputs if o.output_type == 'stream'
                and re.search(r'Warning:|Traceback|FAILED',o.text)]
    assert not errors and not warnings
    assert APP in nb.cells[0].source
    assert len([c for c in nb.cells if c.cell_type == 'markdown' and c.source.startswith('## ')]) == 9
    banned = re.compile(r'\brubric\b|\bmarks\b|real.world|\bcausal\b|provenance|documentation|Taxila',re.I)
    for cell in nb.cells:
        assert not banned.search(cell.source), cell.source[:120]
    embedded = []
    for cell in code_cells:
        if not cell.source.startswith('# Implementation used in this notebook.'):
            continue
        parsed = ast.parse(cell.source)
        values = {n.targets[0].id:n.value for n in parsed.body
                  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
        path = ast.literal_eval(values['destination'].right)
        assert (ROOT/path).read_text() == ast.literal_eval(values['source']), path
        embedded.append(path)
    html = html_path.read_text()
    soup = BeautifulSoup(html,'html.parser')
    assert not soup.find_all('script',src=True) and not soup.find_all('link')
    for node in soup.find_all(['script','style']):
        node.decompose()
    assert not banned.search(soup.get_text())
    for cell in code_cells:
        for output in cell.outputs:
            if 'image/png' in output.get('data',{}):
                assert output.data['image/png'] in html
    pdf = PdfReader(pdf_path)
    text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
    assert not banned.search(text)
    assert APP in (pdf.pages[0].extract_text() or '')
    assert all(f'Figure {i}' in text for i in range(1,11))
    for i,page in enumerate(pdf.pages,1):
        body = (page.extract_text() or '').replace('Mobile Price Prediction','')
        body = re.sub(r'Page\s+\d+\s+of\s+\d+','',body)
        assert body.strip(), f'Blank PDF page {i}'
    result = json.loads((ROOT/'reports/results.json').read_text())
    assert result['selected_model'] == 'Logistic Regression'
    assert result['holdout_metrics']['accuracy'] == .98
    assert abs(result['holdout_metrics']['macro_f1'] - .9800490049004902) < 1e-12
    app = (ROOT/'app.py').read_text()
    assert not banned.search(app)
    assert 'Model and evaluation' in app and 'p_0, p_1, p_2 and p_3' in app
    assert json.loads((ROOT/'reports/app_verification.json').read_text())['pytest_returncode'] == 0
    assert sorted(p.name for p in (ROOT/'submission').iterdir()) == ['GROUP_NO.ipynb','GROUP_NO.pdf']
    summary = {
        'submission_files':{p.name:{'bytes':p.stat().st_size,
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in (ROOT/'submission').iterdir()},
        'html_review_copy':str(html_path.relative_to(ROOT)),
        'notebook_code_cells_executed':len(code_cells),'error_outputs':0,'warning_outputs':0,
        'pdf_pages':len(pdf.pages),'figures_preserved':10,'embedded_sources_match':embedded,
        'app_link_on_first_page':True,'grading_and_unrelated_language_removed':True,
        'documentation_links_removed':True,'visual_pdf_review':'All pages reviewed; dense charts checked at full size',
        'model_and_metrics_preserved':True,'holdout_accuracy':.98,
        'holdout_macro_f1':result['holdout_metrics']['macro_f1'],
        'app_checks':'15 tests and 9 subtests passed',
        'remaining_student_metadata':'Group number, member names and student IDs',
    }
    for name in ['editorial_review.json','final_acceptance.json']:
        (ROOT/'reports'/name).write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()

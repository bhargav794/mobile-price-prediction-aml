"""Convert the executed HTML report to a paginated PDF with all its outputs."""
from pathlib import Path
import re
from bs4 import BeautifulSoup
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
PRINT_STYLE = '''
@page {
    size: A4;
    margin: 16mm 14mm 16mm;
    @top-left { content: "Mobile Price Prediction"; font: 8pt "DejaVu Sans"; color: #526476; }
    @bottom-right { content: "Page " counter(page) " of " counter(pages); font: 8pt "DejaVu Sans"; color: #526476; }
}
@page:first { @top-left { content: none; } }
@page wide { size: A4 landscape; margin: 14mm; }
html, body { margin: 0 !important; padding: 0 !important; width: auto !important; }
body, .jp-Notebook, .jp-RenderedHTMLCommon {
    font-family: "DejaVu Sans", sans-serif !important;
    color: #203347; font-size: 9pt !important; line-height: 1.47 !important;
}
main, .jp-Notebook { display: block !important; padding: 0 !important; }
.jp-Cell, .jp-Cell-outputWrapper, .jp-OutputArea, .jp-OutputArea-child,
.jp-OutputArea-output, .jp-Cell-inputWrapper, .jp-InputArea,
.jp-MarkdownOutput, .jp-RenderedHTMLCommon {
    display: block !important; width: auto !important; min-width: 0 !important;
    max-width: 100% !important; overflow: visible !important; margin-left: 0 !important;
}
.jp-InputPrompt, .jp-OutputPrompt, .jp-InputArea-prompt, .jp-OutputArea-prompt { display: none !important; }
.jp-Cell { margin: 0 0 3mm !important; padding: 0 !important; }
h1 { font-size: 25pt !important; margin: 3mm 0 5mm !important; }
h2 { font-size: 15pt !important; color: #137c8b !important; margin: 7mm 0 3mm !important; break-after: avoid; }
h3 { font-size: 11pt !important; break-after: avoid; }
.anchor-link { display: none !important; }
p { margin: 0 0 3mm !important; orphans: 3; widows: 3; }
li { margin-bottom: 1mm; }
a { color: #137c8b; text-decoration: none; }
.assignment-cover { border-radius: 0 !important; padding: 8mm !important; margin-bottom: 5mm !important; break-inside: avoid; }
.assignment-cover p { color: white; }
table, table.dataframe {
    table-layout: fixed !important; border-collapse: collapse !important;
    width: 100% !important; max-width: 100% !important; margin: 2mm 0 5mm !important;
    font-size: 7.5pt !important; line-height: 1.3 !important; text-align: left !important;
}
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td {
    white-space: normal !important; overflow-wrap: anywhere !important;
    word-break: normal !important; text-align: left !important;
    border: 0.2mm solid #d8e3ea !important; padding: 1.5mm !important;
    max-width: none !important;
}
th { background: #edf5f8 !important; color: #173d54; font-weight: bold; }
tbody tr:nth-child(even) { background: #f6f9fb !important; }
.table-part { font-size: 8pt; color: #526476; margin: 2mm 0; break-after: avoid; }
code { font-family: "DejaVu Sans Mono", monospace !important; font-size: 8pt !important;
       white-space: pre-wrap !important; overflow-wrap: anywhere; }
pre { font-size: 8pt !important; white-space: pre-wrap !important; overflow-wrap: anywhere; background: #f5f8fa; padding: 3mm; }
img { max-width: 100% !important; height: auto !important; display: block; margin: 2mm auto 4mm; }
.wide-figure { page: wide; break-before: page; break-after: page; }
.wide-figure img { width: 260mm !important; max-height: 172mm !important; object-fit: contain; }
.wide-figure .figure-label { font-size: 9pt; color: #526476; }
.report-figure { break-inside: avoid; }
.report-figure img { max-height: 195mm !important; object-fit: contain; }
'''


def split_wide_tables(soup):
    """Keep every cell but split wide previews into legible column groups."""
    for table in list(soup.find_all('table')):
        rows = table.find_all('tr')
        if not rows:
            continue
        width = max(len(row.find_all(['th', 'td'], recursive=False)) for row in rows)
        if width <= 8:
            continue
        first = rows[0].find_all(['th', 'td'], recursive=False)
        has_index = bool(first and not first[0].get_text(strip=True))
        start_column = 1 if has_index else 0
        for start in range(start_column, width, 6):
            indices = ([0] if has_index else []) + list(range(start, min(start+6, width)))
            label = soup.new_tag('p', attrs={'class': 'table-part'})
            label.string = f'Columns {start-start_column+1}-{min(start+6,width)-start_column} of {width-start_column}'
            table.insert_before(label)
            new_table = soup.new_tag('table', attrs={'class': 'dataframe'})
            for section_name in ['thead', 'tbody']:
                original_section = table.find(section_name, recursive=False)
                if original_section is None:
                    continue
                section = soup.new_tag(section_name)
                for row in original_section.find_all('tr', recursive=False):
                    new_row = soup.new_tag('tr')
                    values = row.find_all(['th','td'],recursive=False)
                    for index in indices:
                        if index < len(values):
                            item = soup.new_tag(values[index].name)
                            item.string = values[index].get_text(' ',strip=True)
                            new_row.append(item)
                    section.append(new_row)
                new_table.append(section)
            table.insert_before(new_table)
        table.decompose()


def main():
    source = ROOT / 'output/html/GROUP_NO.html'
    soup = BeautifulSoup(source.read_text(), 'html.parser')
    if soup.title:
        soup.title.string = 'Mobile Price Prediction'
    for node in soup.find_all(['script', 'style']):
        node.decompose()
    split_wide_tables(soup)
    # Put the histogram interpretation before its full-page dashboard so it does
    # not occupy an otherwise nearly empty portrait page between landscape figures.
    interpretation = next((p for p in soup.find_all('p')
                           if p.get_text().startswith('The distributions show')), None)
    chart_images = [img for img in soup.find_all('img')
                    if img.get('src','').startswith('data:image/png')]
    if interpretation is not None and len(chart_images) > 1:
        chart_cell = chart_images[1].find_parent(class_='jp-Cell')
        if chart_cell is not None:
            chart_cell.insert_before(interpretation.extract())
    for cell in list(soup.select('.jp-Cell')):
        if not cell.get_text(strip=True) and not cell.find('img'):
            cell.decompose()
    # Dense attribute, bivariate and correlation dashboards get a landscape page.
    wide_figure_numbers = {2, 3, 4}
    for number, img in enumerate(soup.find_all('img'), 1):
        if not img.get('src', '').startswith('data:image/png'):
            continue
        wrapper = soup.new_tag('div',attrs={'class':'wide-figure' if number in wide_figure_numbers else 'report-figure'})
        img.wrap(wrapper)
        label = soup.new_tag('p',attrs={'class':'figure-label'})
        label.string = f'Figure {number}'
        wrapper.insert(0,label)
    # Flatten notebook layout containers into a print flow. Empty cell wrappers
    # between named landscape pages otherwise create a blank portrait page.
    from bs4 import Tag
    blocks = []
    def collect(node):
        if not isinstance(node, Tag):
            return
        classes = set(node.get('class', []))
        if (node.name in {'h1','h2','h3','h4','p','ul','ol','table','pre'}
                or classes & {'assignment-cover','wide-figure','report-figure'}):
            blocks.append(node)
            return
        for child in list(node.children):
            collect(child)
    collect(soup.body)
    print_body = soup.new_tag('main')
    for block in blocks:
        print_body.append(block.extract())
    soup.body.clear()
    soup.body.append(print_body)
    text = str(soup)
    # DejaVu fonts handle the technical symbols; normalize typographic hyphens.
    text = text.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
    output = ROOT / 'submission/GROUP_NO.pdf'
    HTML(string=text, base_url=str(source.parent)).write_pdf(
        output, stylesheets=[CSS(string=PRINT_STYLE)], font_config=FontConfiguration())
    reader = PdfReader(output)
    combined = '\n'.join(page.extract_text() or '' for page in reader.pages)
    for expected in ['Mobile Price Prediction','Decision Tree','Random Forest','SVM','Logistic Regression','0.9800']:
        assert expected in combined, expected
    assert 'https://mobile-price-prediction-aml.streamlit.app/' in combined
    assert len(reader.pages) >= 10
    print(f'Created {output.name}: {len(reader.pages)} pages, {output.stat().st_size:,} bytes')
    # Keep the print HTML for repeatable layout inspection, outside submission.
    (ROOT / 'output/html/GROUP_NO.print.html').write_text(text)


if __name__ == '__main__':
    main()

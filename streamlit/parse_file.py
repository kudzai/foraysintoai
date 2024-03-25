from io import StringIO
import fitz


def read_text_file(file):
    return StringIO(file.getvalue().decode("utf-8"))


def parse_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text.strip()


def parse(file):
    file_type = file.type
    if file_type == 'application/pdf':
        yield parse_pdf(file)
    elif file_type == 'text/plain':
        yield read_text_file(file)
    else:
        print(f"Unsupported file type: {file_type}")
        yield None

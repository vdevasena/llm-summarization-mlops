from pathlib import Path

from pypdf import PdfReader
from docx import Document


def extract_text_from_txt(file_bytes):
    """
    Extract text from a TXT file.
    """

    return file_bytes.decode(
        "utf-8",
        errors="ignore"
    )


def extract_text_from_md(file_bytes):
    """
    Extract text from a Markdown file.
    """

    return file_bytes.decode(
        "utf-8",
        errors="ignore"
    )


def extract_text_from_pdf(file_bytes):
    """
    Extract text from a PDF file.
    """

    import io

    pdf_file = io.BytesIO(
        file_bytes
    )

    reader = PdfReader(
        pdf_file
    )

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)


def extract_text_from_docx(file_bytes):
    """
    Extract text from a DOCX file.
    """

    import io

    doc_file = io.BytesIO(
        file_bytes
    )

    document = Document(
        doc_file
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():

            paragraphs.append(
                paragraph.text
            )

    return "\n\n".join(paragraphs)


def extract_text(
    file_name,
    file_bytes
):
    """
    Detect the file type and
    extract text accordingly.
    """

    extension = (
        Path(file_name)
        .suffix
        .lower()
    )

    if extension == ".txt":

        return extract_text_from_txt(
            file_bytes
        )

    elif extension == ".md":

        return extract_text_from_md(
            file_bytes
        )

    elif extension == ".pdf":

        return extract_text_from_pdf(
            file_bytes
        )

    elif extension == ".docx":

        return extract_text_from_docx(
            file_bytes
        )

    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )
from docling.document_converter import DocumentConverter
from docling.backend.idp_backend import IDPChunker
from pathlib import Path
def main():
    source =Path("file.docx")
    converter = DocumentConverter(use_idp=True)
    result = converter.convert(source)
    chunks = list(IDPChunker().chunk(result.document))
    print(chunks)
    breakpoint()


if __name__ == "__main__":
    main()
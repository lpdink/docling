from docling_core.types.doc import DoclingDocument
from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from io import BytesIO
from docling.datamodel.document import InputDocument
from docling.backend.md_backend import MarkdownDocumentBackend
import json

def walk(idp_raw_result) -> DoclingDocument:
    # Create base document structure
    doc = DoclingDocument(
        schema_name="DoclingDocument",
        version="1.0.0",
        name="file",
        origin={
            "mimetype": "text/markdown",
            "binary_hash": 12345678987654321, #TODO: calculate hash
            "filename": "file.md"
        },
        furniture={
            "self_ref": "#/furniture",
            "children": [],
            "name": "_root_",
            "label": "unspecified"
        },
        body={
            "self_ref": "#/body",
            "children": [],
            "name": "_root_",
            "label": "unspecified"
        },
        groups=[],
        texts=[],
        pictures=[],
        tables=[],
        key_value_items=[],
        pages={}
    )

    # Helper function to create text node
    def create_text_node(item, index):
        return {
            "self_ref": f"#/texts/{index}",
            "parent": {"$ref": "#/body"},
            "children": [],
            "label": "paragraph" if item["type"] == "text" else item["type"],
            "prov": [],
            "orig": item["text"],
            "text": item["text"]
        }

    # Helper function to create picture node 
    def create_picture_node(item, index):
        return {
            "self_ref": f"#/pictures/{index}",
            "parent": {"$ref": "#/body"},
            "children": [],
            "label": "picture",
            "prov": [],
            "captions": [],
            "references": [],
            "footnotes": [],
            "image": {
                "mimetype": "image/png",
                "dpi": 300,
                "size": {
                    "width": float(item.get("width", 0)),
                    "height": float(item.get("height", 0))
                },
                "uri": item.get("markdownContent", "")
            },
            "annotations": []
        }

    # Process items and build tree structure
    text_index = 0
    picture_index = 0
    stack = []  # Stack to track parent nodes
    
    for item in idp_raw_result:
        level = item["level"]
        
        # Create node based on type
        if item["type"] == "figure":
            node = create_picture_node(item, picture_index)
            picture_index += 1
            doc.pictures.append(node)
        else:
            node = create_text_node(item, text_index)
            text_index += 1
            doc.texts.append(node)

        # Pop stack until we find the parent level
        while stack and stack[-1]["level"] >= level:
            stack.pop()

        # Update parent reference if we have a parent
        if stack:
            parent = stack[-1]["node"]
            node["parent"] = {"$ref": parent["self_ref"]}
            parent["children"].append({"$ref": node["self_ref"]})
        else:
            # Top level nodes are children of body
            doc.body.children.append({"$ref": node["self_ref"]})

        # Push current node to stack
        stack.append({"level": level, "node": node})

    return doc

if __name__ == "__main__":
    with open("../../idp.json", "r", encoding='utf-8') as f:
        idp_raw_result = json.load(f)
    docling_result = walk(idp_raw_result)
    # Convert to dictionary first, then to JSON with ensure_ascii=False
    print(json.dumps(docling_result.dict(), indent=4, ensure_ascii=False))
    # save to out.json with UTF-8 encoding
    with open("out.json", "w", encoding='utf-8') as f:
        json.dump(docling_result.dict(), f, indent=4, ensure_ascii=False)
    # print(docling_result)
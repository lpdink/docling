import os
import time
from io import BytesIO
from typing import Any, Iterator, List, Optional, Set, Tuple, Union

from alibabacloud_docmind_api20220711 import models as docmind_api20220711_models
from alibabacloud_docmind_api20220711.client import Client as docmind_api20220711Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models
from docling_core.transforms.chunker import BaseChunk, BaseChunker
from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    ImageRef,
    NodeItem,
    PictureItem,
    TableCell,
    TableData,
    TextItem,
)
from pydantic import BaseModel

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat


class IDPDocumentBackend(DeclarativeDocumentBackend):
    def __init__(self, in_doc, path_or_stream):
        super().__init__(in_doc, path_or_stream)
        self._client = self._build_open_api_client()

    def is_valid(self):
        return True

    @classmethod
    def supports_pagination(cls) -> bool:
        return True

    @classmethod
    def supported_formats(cls) -> Set["InputFormat"]:
        return {
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.HTML,
            InputFormat.MD,
            InputFormat.CSV,
            InputFormat.JSON_DOCLING,
            InputFormat.XML_JATS,
            InputFormat.XML_USPTO,
            InputFormat.DOCX,
            InputFormat.PPTX,
            InputFormat.XLSX,
            InputFormat.ASCIIDOC,
        }

    def _build_open_api_client(self):
        ak = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID", None)
        sk = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET", None)
        if not ak or not sk:
            raise ValueError("Alibaba Cloud credentials not set")
        config = open_api_models.Config(access_key_id=ak, access_key_secret=sk)
        config.endpoint = "docmind-api.cn-hangzhou.aliyuncs.com"
        client = docmind_api20220711Client(config)
        return client

    def _submit_doc_parser_job(self, file_name):
        with open(file_name, "rb") as f:
            request = docmind_api20220711_models.SubmitDocParserJobAdvanceRequest(
                file_url_object=f, file_name=os.path.basename(file_name)
            )
            runtime = util_models.RuntimeOptions()
            response = self._client.submit_doc_parser_job_advance(request, runtime)
        if not response.body.data:
            raise RuntimeError(response.body.message)
        return response.body.data.id

    def _query_doc_parser_status(self, job_id):
        request = docmind_api20220711_models.QueryDocParserStatusRequest(id=job_id)
        response = self._client.query_doc_parser_status(request)
        if not response.body.data:
            raise RuntimeError(response.body.message)
        return (
            response.body.data.status,
            response.body.data.number_of_successful_parsing,
        )

    def _get_doc_parser_result(self, job_id, number_of_successful_parsing):
        ans = []
        while number_of_successful_parsing > 0:
            request = docmind_api20220711_models.GetDocParserResultRequest(
                id=job_id, layout_step_size=3000, layout_num=len(ans)
            )
            response = self._client.get_doc_parser_result(request)
            ans.extend(response.body.data["layouts"])
            number_of_successful_parsing -= 3000
        return ans

    def _get_idp_raw_result(self):
        job_id = self._submit_doc_parser_job(str(self.file))
        status = None
        while status != "success":
            status, number_of_successful_parsing = self._query_doc_parser_status(job_id)
            time.sleep(2)
        return self._get_doc_parser_result(job_id, number_of_successful_parsing)

    def convert(self) -> DoclingDocument:
        idp_raw_result = self._get_idp_raw_result()
        return self.walk(idp_raw_result)

    def _parse_img(self, img_markdown_text):
        import re

        pattern = r"(http[^)]+)"
        match = re.search(pattern, img_markdown_text)
        if match:
            return match.group(0)
        else:
            return ""

    def _get_img(self, img_markdown_text):
        import requests
        from PIL import Image

        url = self._parse_img(img_markdown_text)
        try:
            raise Exception("temp use idp uri")
            rep_image = requests.get(url)
            pil_image = Image.open(BytesIO(rep_image.content))
            if "dpi" in pil_image.info:
                dpi = pil_image.info["dpi"][0]
            else:
                dpi = 300
            image = ImageRef.from_pil(image=pil_image, dpi=dpi)
        except:
            from docling_core.types.doc import Size

            image = ImageRef(mimetype="image/png", dpi=300, size=Size(), uri=url)
        return image

    def walk(self, idp_raw_result) -> DoclingDocument:
        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="text/markdown",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        nodes_stack = []
        parent: NodeItem = None
        node = None
        for element in idp_raw_result:
            type = element["type"]
            # sub_type = element["subType"]
            level = element["level"]
            while len(nodes_stack) > 0 and level <= nodes_stack[-1]["level"]:
                nodes_stack.pop()
            parent = nodes_stack[-1]["node"] if len(nodes_stack) > 0 else None
            if type == "title":
                node = doc.add_title(text=element["text"], parent=parent)
            elif type == "text":
                node = doc.add_text(
                    text=element["text"], parent=parent, label=DocItemLabel.TEXT
                )
            elif type == "figure":
                # TODO:思考一下要不要OCR结果
                text_node = None  # doc.add_text(text=element["text"], parent=parent, label=DocItemLabel.CAPTION)
                image = self._get_img(element["markdownContent"])
                node = doc.add_picture(caption=text_node, image=image, parent=parent)
            nodes_stack.append({"node": node, "level": level})
        return doc


class IDPChunk(BaseModel):
    text: str = ""
    image: list[str] = []
    metadata: dict = {}

    @property
    def size(self):
        return len(self.text)

    def merge(self, other: Union["IDPChunk", str]):
        if isinstance(other, str):
            other = IDPChunk(text=other)
        self.text += "\n" + other.text
        self.image += other.image


class IDPChunker(BaseChunker):
    MAX_CHUNK_SIZE: int = 768  # 根据BERT等模型优化设置, text-embddding可以到1024等
    MIN_CHUNK_SIZE: int = 256  # 防止过小碎片
    CONTEXT_LEVELS: int = 3  # 保持3层上下文

    def _get_node_text(self, doc: DoclingDocument, node: NodeItem) -> str:
        if isinstance(node, TextItem):
            return node.text
        elif isinstance(node, PictureItem):
            # TODO:这里思考一下要不要用OCR结果。
            return ""
            # return node.caption_text(doc)
        return ""

    def _build_chunk_metadata(self, node: NodeItem, parent_path: List[str]) -> dict:
        return {
            "node_id": node.self_ref,
            "parent_ids": parent_path[-self.CONTEXT_LEVELS :],
            "depth": len(parent_path) + 1,
            "content_type": node.__class__.__name__,
        }

    def _recursive_merge(
        self,
        doc: DoclingDocument,
        node: NodeItem,
        current_chunk: IDPChunk,
        parent_path: List[str],
        chunks: List[IDPChunk],
    ) -> None:
        node_text = self._get_node_text(doc, node)
        node_size = len(node_text)
        if isinstance(node, PictureItem):
            current_chunk.image.append(str(node.image.uri))

        # 上下文路径维护
        new_parent_path = parent_path + [node.self_ref]

        # 子节点预处理（深度优先）
        child_chunks = []
        for child in node.children:
            resolved_child = child.resolve(doc)
            self._recursive_merge(
                doc, resolved_child, IDPChunk(), new_parent_path, child_chunks
            )

        # 当前节点与子块的合并策略
        if (
            current_chunk.size + node_size + sum(c[0].size for c in child_chunks)
            <= self.MAX_CHUNK_SIZE
        ):
            # 合并当前节点及其所有子块
            current_chunk.merge(node_text)
            for child_chunk, _ in child_chunks:
                current_chunk.merge(child_chunk)
            chunks.append((current_chunk, new_parent_path))
        else:
            # 先提交当前块
            if current_chunk:
                chunks.append((current_chunk, parent_path))
            # 处理当前节点
            # if node_size > self.MAX_CHUNK_SIZE:
            if False:
                # 节点自身超过最大尺寸，强制分割
                # TODO:不能要这个逻辑，否则图像无法分割.
                self._split_large_node(node_text, new_parent_path, chunks)
            else:
                # 创建新块并尝试合并子块
                image = [str(node.image.uri)] if isinstance(node, PictureItem) else []
                new_chunk = IDPChunk(text=node_text, image=image)
                new_size = node_size
                for child_chunk, _ in child_chunks:
                    if new_size + child_chunk.size > self.MAX_CHUNK_SIZE:
                        chunks.append((new_chunk, new_parent_path))
                        new_chunk = IDPChunk(
                            text=child_chunk.text, image=child_chunk.image[:]
                        )
                    else:
                        new_chunk.merge(child_chunk)
                chunks.append((new_chunk, new_parent_path))

    def _split_large_node(
        self, text: str, parent_path: List[str], chunks: List[BaseChunk]
    ) -> None:
        """处理超长节点的分割"""
        sentences = text.replace("。", ".").split(".")
        current_chunk = IDPChunk()
        for sent in sentences:
            sent_size = len(sent)
            if current_chunk.size + sent_size > self.MAX_CHUNK_SIZE:
                chunks.append((current_chunk, parent_path))
                current_chunk = IDPChunk(text=sent)
            else:
                current_chunk.merge(sent)
        if current_chunk:
            chunks.append((current_chunk, parent_path))

    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        chunks = []
        for item in dl_doc.body.children:
            root_node = item.resolve(dl_doc)
            self._recursive_merge(
                doc=dl_doc,
                node=root_node,
                current_chunk=IDPChunk(),
                parent_path=[],
                chunks=chunks,
            )

        # 后处理：合并过小碎片
        merged_chunks = []
        prev_chunk = None
        for chunk, path in chunks:
            full_text = chunk.text
            if (
                prev_chunk
                and len(prev_chunk.text) + len(full_text) < self.MIN_CHUNK_SIZE
            ):
                prev_chunk.merge(chunk)
                prev_chunk.metadata["merged"] = True
            else:
                if prev_chunk:
                    merged_chunks.append(prev_chunk)
                prev_chunk = IDPChunk(
                    text=full_text,
                    image=chunk.image[:],
                    metadata=self._build_chunk_metadata(
                        node=root_node, parent_path=path
                    ),
                )
        if prev_chunk:
            merged_chunks.append(prev_chunk)

        yield from merged_chunks

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

def dense_search(store:Chroma, query:str,k:int=5,metadata_filter : dict | None=None)->list[Document]:
    kwargs: dict = {"k": k}
    if metadata_filter:
        kwargs["filter"] = metadata_filter
    return store.similarity_search(query, **kwargs)
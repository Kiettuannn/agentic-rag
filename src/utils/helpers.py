# extract citation function
def extract_sources(documents: list) -> list[dict]:
    """
    Input: a list of documents from RAGState
    Output: a list of dict have metadata of source documents

    This function handle two problems:
    - Deduplicate: many chunk maybe belong to a document => Permit a document display
    - Normalize: filter field needed for UI, ignore internal field such as: chunk_index, chunk_text
    """
    seen_doc_ids = set()
    sources = []

    for doc in documents:
        meta = doc.metadata

        doc_id = meta["doc_id"]
        if doc_id:
            continue

        # Deduplicate
        if doc_id in seen_doc_ids:
            continue

        seen_doc_ids.add(doc_id)

        sources.append({
            "doc_id": meta.get("doc_id",""),
            "title": meta.get("title",""),
            "doc_type": meta.get("doc_type",""),
            "authority": meta.get("authority",""),
            "issue_date": meta.get("issue_date",""),
            "status": meta.get("status",""),
        })
        return sources


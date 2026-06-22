from src.retrieval.graph import build_graph, graph_search


class FakeDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata

    def __repr__(self):
        return f"FakeDoc({self.metadata})"


class FakeStore:
    def __init__(self):
        self.docs = {
            "A": [
                FakeDoc("A chunk 1", {"doc_id": "A", "chunk_index": 0}),
                FakeDoc("A chunk 2", {"doc_id": "A", "chunk_index": 1}),
            ],
            "B": [
                FakeDoc("B chunk 1", {"doc_id": "B", "chunk_index": 0}),
            ],
            "C": [
                FakeDoc("C chunk 1", {"doc_id": "C", "chunk_index": 0}),
            ],
        }

    def similarity_search(self, query, k=5, filter=None):
        if not filter:
            # initial dense retrieval → giả lập trả B
            return self.docs["B"][:k]

        doc_id = filter.get("doc_id")
        return self.docs.get(doc_id, [])[:k]


def test_graph_search_basic():
    """
    Flow:
    dense_search -> seed B
    expand_neighbors(B, 1 hop) -> A, B, C
    retrieve chunks from A,B,C
    """

    sample_relationships = [
        {"doc_id": "A", "other_doc_id": "B", "relationship": "Văn bản sửa đổi"},
        {"doc_id": "B", "other_doc_id": "C", "relationship": "Văn bản sửa đổi"},
    ]

    graph = build_graph(sample_relationships)
    store = FakeStore()

    results = graph_search(
        store=store,
        graph=graph,
        query="test query",
        k=5,
        initial_k=1,
        max_hops=1,
    )

    result_doc_ids = {
        doc.metadata["doc_id"]
        for doc in results
    }

    assert result_doc_ids == {"A", "B", "C"}

    print("PASS test_graph_search_basic")
    print(results)


if __name__ == "__main__":
    test_graph_search_basic()
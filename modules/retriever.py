# Setup retriever
def get_retriever(vector_db, k=5):
    return vector_db.as_retriever(search_kwargs={"k": k})

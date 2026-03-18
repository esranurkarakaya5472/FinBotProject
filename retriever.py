class DocumentRetriever:
    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        self.retriever = None

    def get_retriever(self, query: str = None):
        """
        Hazırlanmış vektör veritabanını kullanarak soruya uygun içerikleri getiren retriever'ı döndürür.
        """
        if not self.vector_store:
            raise ValueError("Hata: Vektör veritabanı bulunamadı!")

        # ---------------------------------------------------------
        # ADIM 5: Sorgu / Retrieval (Benzerlik Araması)
        # ---------------------------------------------------------
        if query:
            print(f"\nAdım 5: Kullanıcı sorusu için alakalı içerikler getiriliyor... Soru: {query}")
        else:
            print("\nAdım 5: Retriever hazırlanıyor...")
            
        self.retriever = self.vector_store.as_retriever()
        return self.retriever

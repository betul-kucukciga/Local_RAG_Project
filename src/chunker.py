import re 

class TextChunker:

    """Metinleri belirlenen karakter limitine (chunk_size) göre kelime bütünlüğünü

    koruyarak parçalayan ve parçalar arasında örtüşme (overlap) sağlayan sınıf.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):

        """Chunker için limit ve örtüşme parametrelerini yapılandırır.

        Args:
            chunk_size (int, optional): Bir parçanın içerebileceği maksimum
              karakter sayısı. Defaults to 500.
            overlap (int, optional): Parçalar arasında bağlam kaybını önlemek
              için devredilecek örtüşme karakter miktarı. Defaults to 50.
        """

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> list[str]:

        """Kelimeleri ortadan bölmeden, belirlenen boyutlara göre metni parçalar.

        Metindeki fazla boşlukları temizler, kelime bazlı ilerler ve paket
        sınırı dolduğunda son birkaç kelimeyi bir sonraki parçanın başına
        devreder (overlap).

        Args:
            text (str): Parçalanacak olan ham metin.

        Returns:
            list[str]: Parçalanmış metin dizilerinin (chunk) listesi.
        """

        if not text or not text.strip():
            return []

        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split(" ")
        
        chunks = []
        current_words = []
        current_length = 0

        for word in words:
            word_len = len(word) + 1  
            
            if current_length + word_len > self.chunk_size and current_words:
                chunk_str = " ".join(current_words).strip()
                chunks.append(chunk_str)
                
                overlap_words = []
                overlap_len = 0
                for w in reversed(current_words):

                    if overlap_len + len(w) + 1 <= self.overlap:
                        overlap_words.insert(0, w)
                        overlap_len += len(w) + 1

                    else:
                        break
                
                current_words = overlap_words
                current_length = overlap_len

            current_words.append(word)
            current_length += word_len

        if current_words:
            chunk_str = " ".join(current_words).strip()
            chunks.append(chunk_str)

        return chunks

    def load_and_chunk_file(self, file_path: str) -> list[str]:

        """Belirtilen yoldaki dosyayı UTF-8 formatında okur ve parçalar.

        Args:
            file_path (str): Okunacak dosyanın tam veya bağıl yolu.

        Returns:
            list[str]: Dosya içeriğinin parçalanmış halini içeren liste.
        """

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return self.chunk_text(text)

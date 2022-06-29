# Crawling Data

## Melakukan Crawling link jurnal
```python
import scrapy

class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = []
        for i in range(1, 33+1):
            urls.append(f'https://pta.trunojoyo.ac.id/c_search/byprod/10/{i}')
        
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for jurnal in response.css('#content_journal > ul > li'):
            yield {
                'link': response.css('div:nth-child(3) > a::attr(href)').get(),
            }
```

Hasil Crawling link jurnal :

|link                                                   |
|-------------------------------------------------------|
|https://pta.trunojoyo.ac.id/welcome/detail/070411100124|
|https://pta.trunojoyo.ac.id/welcome/detail/070411100124|
|https://pta.trunojoyo.ac.id/welcome/detail/070411100124|
|https://pta.trunojoyo.ac.id/welcome/detail/070411100124|
|https://pta.trunojoyo.ac.id/welcome/detail/070411100124|
|...|
|...|


# Crawling Informasi Detail Jurnal

```python
import scrapy
import pandas as pd


class QuotesSpider(scrapy.Spider):
    name = "quote"

    def start_requests(self):
        data_csv = pd.read_csv('link.csv').values
        start_urls = [ link[0] for link in data_csv ]

        for url in start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        yield {
            'Judul': response.css('#content_journal > ul > li > div:nth-child(2) > a::text').extract(),
            'Abstraksi': response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) > p::text').extract(),
        }
```

Hasil Crawling detail jurnal:
|Judul                                                  |Abstraksi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |FIELD3|
|-------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
|SISTEM PENDUKUNG KEPUTUSAN PEMILIHAN KARYAWAN BERPRESTASI DENGAN INTEGRASI FAHP dan ELECTRE II|Sumber daya manusia mutlak dibutuhkan untuk kemajuan suatu perusahaan guna menjadikan perusahaan itu menjadi perusahaan yang maju dan tidak kalah bersaing dengan perusahaan lainnya. Dalam hal ini maka dibuatlah Sistem Pendukung keputusan (SPK) pemilihan karyawan berprestasi untuk mencari karyawan berprestasi. Sistem Pendukung keputusan (SPK) ini menggunakan FAHP dan ELECTRE II. Metode FAHP merupakan metode yang cukup obyektif untuk proses penilaian berdasarkan hirarki kriteria yang digabungkan dengan konsep fuzzy sesuai kriteria penilaian kinerja karyawan perusahaan. Setelah mendapat bobot dilakukan proses selanjutnya dengan menggunakan metode ELECTRE II hingga mendapat karyawan berprestasi. Untuk menjaga bahwa penilaian ini tidak berpihak kepada salah satu karyawan dan bebas intervensi dari karyawan maka perusahaan menggunakan pihak luar yang independen dan profesional untuk melakukan penilaian terhadap karyawan. Hasil penilaian dari pihak luar yang berupa angka-angka kemudian oleh departemen sdm dilakukan penjumlahan nilai komulatif kriteria untuk mendapat karyawan terbaik tanpa ada prioritas kriteria yang lebih penting. Dengan menggunakan metode FAHP dan ELECTREII hasil keluaran mendekati keakuratan dengan hasil yang diperoleh dari areal manager di banding dengan hitung manual perusahaan tanpa memperhatikan bobot kepentingan.                                                              |CAI   |
|Gerak Pekerja Pada Game Real Time Strategy Menggunakan Finite State Machine|Gerak pekerja ada pada game yang memiliki genre RTS (Real-Time Strategy). Gerak pekerja memiliki berbagai macam gerak. Oleh sebab itu dibutuhkan sebuah pendekatan konsep AI  untuk mendesain perilaku pekerja tersebut. Perilaku karakter tersebut harus ditambahi dengan AI (Artifical intelegent) agar perilakunya menjadi lebih hidup dan realistis. Dalam penelitian ini AI yang digunakan adalah Finite State Machine. Finite State Machine digunakan untuk menentukan gerak pekerja terhadap parameter-parameter yang digunakan sebagai dasar gerak yang akan dilakukan . Selanjutnya akan disimulasikan pada game RTS dengan menggunakan game engine. Hasil yang di peroleh dalam penelitian ini adalah penerapan metode Finite State machine untuk menentukan gerak pekerja berdasarkan parameter jumlah harta, prajurit, kondisi bangunan, dan stockpile (jumlah resources yang di bawa).   Kata kunci : Game, Real-Time Strategy, Gerak Pekerja, Finite State Machine.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |CAI   |
|SISTEM PENENTUAN STATUS GIZI PASIEN RAWAT INAP MENGGUNAKAN METODE NAÏVE BAYES CLASSIFIER (STUDI KASUS : RSUD DR. H. SLAMET MARTODIRDJO PAMEKASAN)|Di Indonesia masalah perkembangan gizi adalah masalah yang perlu perhatian lebih. Jika seseorang tidak mengetahui tentang status gizinya, maka tidak akan dapat mengontrol berapa banyak jumlah gizi yang dibutuhkan dalam tubuh. Dalam penelitian ini dirancang aplikasi sistem pendukung keputusan yang digunakan untuk menentukan status gizi pasien dan memberikan solusi makanan pada pasien sesuai riwayat penyakit yang di derita pasien. Sistem yang dirancang ini berbasis Web, dan memudahkan pihak admin atau ahli gizi rumah sakit dalam penentuan status gizi pasien. Diharapkan dengan adanya aplikasi ini dapat memberikan efisien dan efektifitas kinerja setiap pihak. Metode yang digunakan dalam penelitian ini menggunakan Naïve Bayes Classifier (NBC). Metode terbaru yang di gunakan untuk memprediksi probabilitas.Metode Naïve bayes Classifier melakukan proses penentuan perhitungan probabilitas status gizi. Dimana dicari nilai probabilitas terbesar yang kemudian menjadi kesimpulan penentuan status gizi. Metode ini dapat diterapkan dalam studi kasus Sistem Penentuan Status Gizi Pasien dengan hasil akurasi terbesar 92%. Kata Kunci : Naïve Bayes Classifier, Sistem Pendukung Keputusan, Status Gizi, Web                                                                                                                                                                                                                  |CAI   |
|RANCANG BANGUN MANAJEMEN PEMBELAJARAN DAN TES TOEFL BERBASIS MOBILE|Penggunaan teknologi mobile saat ini sangat marak, disamping keunggulannya dapat mudah bawa dibawa kemana-mana, teknologi mobile sekarang sangat mudah untuk dieksplorasi, terbukti dengan adanya smartphone yang mempunyai banyak layanan yang dapat bermanfaat bagi para penggunanya. Toefl (Test Of English As A Foreign Languange) sangat dibutuhkan dalam menghadapi kemajuan teknologi saat ini, Kurang besarnya minat masyarakat dalam belajar bahasa inggris dan mengikuti tes-tes Toefl yang ada, berpengaruh pada kemajuan teknologi.  Android  merupakan  subset  perangkat  lunak  untuk perangkat  mobile  yang  meliputi  sistem  operasi, middleware,  dan  aplikasi  inti  yang  di  release  oleh Google. Android SDK adalah  tools  API (Application Programming Interface)  yang digunakan untuk memulai membuat aplikasi pada platform Android dengan menggunakan bahasa pemrograman Java. Eclipse adalah sebuah IDE (Integrated Development Environment) yang digunakan dalam  coding  aplikasi Android nantinya. Salah satu pemanfaatan teknologi mobile yaitu dengan membuat media pembelajaran dan tes TOEFL baik soal berupa teks dan audio dengan menggunakan  teknologi Android, tentunya berbasis mobile. Dengan tes TOEFL, pembelajaran test dan kelebihan listening menu di dalam aplikasi ini, diharapkan akan dapat membatu meningkatkan pemahaman pengguna mengenai tes TOEFL.  Kata Kunci: tes TOEFL, ANDROID, Java , Mobile, SDK |RPL   |
|RANCANG BANGUN APLIKASI SEARCH ENGINE DAN SISTEM PENDETEKSI PLAGIARISME MENGGUNAKAN METODE LSA-SOM|Plagiarisme adalah mencuri gagasan, kata-kata, kalimat atau hasil penelitian orang lain dan menyajikannya seolah-olah sebagai karya sendiri. Metode LSA merupakan salah satu metode dari beberapa metode untuk pendeteksian plagiarisme. Metode Latent Semantic Analysis (LSA) adalah sebuah teori dan metode untuk menggali dan mempresentasikan konteks yang digunakan sebagai sebuah arti kata dengan memanfaatkan komputasi statistik untuk sejumlah corpus yang besar. Penggunaan LSA konvensional bersifat sekuensial dikarenakan masih belum adanya klasterisasi sehingga proses analisa akan dilakukan secara global pada keseluruhan dokumen yang tentunya memerlukan waktu komputasi yang besar. Untuk itu dapat digunakan teknik klasifikasi tak terawasi seperti penggunaan Self Organizing Map (SOM) dan pengujian similaritas secara global hanya dilakukan pada centroid masing-masing kluster untuk kemudian dilakukan pengujian similaritas secara lokal pada klaster yang terpilih yang menjadi bagian dari klaster tersebut. Pada hasil penelitian  LSA-SOM memiliki rata-rata precision 94.4% dan recall 64.04%.  Kata kunci : Cosine Similarity, Latent Semantic Analysis, plagiarisme Self Organizing Map, Singular Value Decomposition                                                                                                                                                                                                       |CAI   |

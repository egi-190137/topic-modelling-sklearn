# Crawling Data

## Membuat Projek Scrapy

Sebelum melakukan crawling data, kita perlu membuat projek scrapy. Untuk cara membuatnya cukup dengan mengetikkan perintah berikut pada _terminal_ atau _command prompt_.

```
Scrapy startproject <nama-project>
```

## Melakukan Crawling link jurnal

Setelah membuat projek buka direktori projek tersebut dengan perintah berikut:

```
cd <nama-projek>
```

Kemudian buat 1 file spider dengan perintah berikut:

```
Scrapy genspider link example.com
```

Lalu edit file spider menjadi seperti code di bawah ini:

```python
import scrapy


class LinkSpider(scrapy.Spider):
    name = 'link'
    start_urls = []

    for i in range(1, 120+1):
        start_urls.append(f'https://pta.trunojoyo.ac.id/c_search/byprod/10/{i}')

    def parse(self, response):
        for jurnal in response.css('#content_journal > ul > li'):
            yield {
                'link': response.css('div:nth-child(3) > a::attr(href)').get(),
            }
```

Perintah untuk melakukan crawling data dan memasukkan data ke dalam file csv:

```
Scrapy crawl link -O <nama-file>.csv
```

Hasil Crawling link jurnal : [file](https://github.com/egi-190137/topic-modelling-sklearn/blob/main/contents/link.csv)
|link |
|------------------------------------------------------------------------------------------------------------------------|
|https://pta.trunojoyo.ac.id/welcome/detail/070411100007 |
|https://pta.trunojoyo.ac.id/welcome/detail/070411100007 |
|https://pta.trunojoyo.ac.id/welcome/detail/070411100007 |
|https://pta.trunojoyo.ac.id/welcome/detail/070411100007 |
|https://pta.trunojoyo.ac.id/welcome/detail/070411100007 |

## Crawling Judul dan abstraksi Jurnal

Buat file spider baru untuk crawling judul dan abstraksi jurnal

```
Scrapy genspider detail example.com
```

Lalu edit file spider seperti code berikut:

```python
import scrapy
import pandas as pd

class DetailSpider(scrapy.Spider):
    name = 'detail'
    data_csv = pd.read_csv('link_baru.csv').values
    start_urls = [ link[0] for link in data_csv ]

    def parse(self, response):
        yield {
            'Judul': response.css('#content_journal > ul > li > div:nth-child(2) > a::text').extract(),
            'Abstraksi': response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) > p::text').extract(),
        }
```

Perintah untuk melakukan crawling data dan memasukkan data ke dalam file csv:

```
Scrapy crawl link -O <nama-file>.csv
```

Hasil Crawling jududl dan abstraksi jurnal: [file](https://github.com/egi-190137/topic-modelling-sklearn/blob/main/contents/detail_pta.csv)

| Judul                                   | Abstraksi                                                         |
| --------------------------------------- | ----------------------------------------------------------------- |
| PERANCANGAN DAN IMPLEMENTASI SISTEM ... | Sistem informasi akademik (SIAKAD) merupakan sistem informasi ... |
| PERANCANGAN DAN IMPLEMENTASI SISTEM ... | Sistem informasi akademik (SIAKAD) merupakan sistem informasi ... |
| PERANCANGAN DAN IMPLEMENTASI SISTEM ... | Sistem informasi akademik (SIAKAD) merupakan sistem informasi ... |
| Gerak Pekerja Pada Game Real Time ...   | Gerak pekerja ada pada game yang memiliki genre RTS (Real-Time... |
| Gerak Pekerja Pada Game Real Time ...   | Gerak pekerja ada pada game yang memiliki genre RTS (Real-Time... |

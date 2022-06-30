# Crawling Data

## Melakukan Crawling link jurnal

Buat 1 file python, lalu edit file spider menjadi seperti code di bawah ini:

```python
import scrapy


class LinkSpider(scrapy.Spider):
    name = 'link'
    start_urls = []

    def start_requests(self):
        urls = []
        for i in range(1, 24+1):
            urls.append(f'https://pta.trunojoyo.ac.id/c_search/byprod/7/{i}')

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
    def parse(self, response):
        for i in range(1, 5+1):
            yield {
                'link': response.css(f'#content_journal > ul > li:nth-child({i}) > div:nth-child(3) > a::attr(href)').get(),
            }
```

Perintah untuk melakukan crawling data dan memasukkan data ke dalam file csv:

```
Scrapy runspider <code-python>.py -O <nama-file>.csv
```

Hasil Crawling link jurnal : [file](https://github.com/egi-190137/topic-modelling-sklearn/blob/main/contents/link.csv)

## Crawling Judul dan abstraksi Jurnal

Buat file python baru untuk crawling judul dan abstraksi jurnal, Lalu edit file spider seperti code berikut:

```python
import scrapy
import pandas as pd

class DetailSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        data_csv = pd.read_csv('link.csv').values
        urls = [ link[0] for link in data_csv ]

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        yield {
            'judul': response.css('#content_journal > ul > li > div:nth-child(2) > a::text').get(),
            'abstraksi': response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) > p::text').get(),
        }
```

Perintah untuk melakukan crawling data dan memasukkan data ke dalam file csv:

```
Scrapy runspider <code-python>.py -O <nama-file>.csv
```

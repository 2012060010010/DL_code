#!/usr/bin/env python
# coding=utf-8

import scrapy
from tech_news.items import BilingualArticleItem
from bs4 import BeautifulSoup
import re

class BilingualSpider(scrapy.Spider):
    name = 'bilingual'
    allowed_domains = ["tech.qq.com","digi.tech.qq.com"]
    start_urls = ["http://tech.qq.com","http://tech.qq.com/web/intelligent.htm","http://digi.tech.qq.com"]
    def parse(self,response):
        item = BilingualArticleItem()
        soup = BeautifulSoup(response.body,'lxml',from_encoding='utf-8')
        article = soup.find_all('p',text=True)
        item['link'] = response.url
        #item['time'] = ''
        #seq = response.xpath('//*[@id="C-Main-Article-QQ"]/div[1]/div/div/span[2]/text()').extract()
        #print type(seq),len(seq)
        #for s in response.xpath('//*[@id="C-Main-Article-QQ"]/div[1]/div/div/span[2]/text()').extract():
         #   pp = s.encode('utf-8')
          #  item['time'] = item['time'] + str(pp)
           # print type(pp),len(item['time'])


        if len(article) > 50 :
            try:
                link = soup.find_all(href=re.compile("tech.qq.com.+.htm"))
                for tag in link:
                    yield scrapy.Request(tag['href'],callback=self.parse)
            except:
                pass
        else:
            try:
                link = soup.find_all(href=re.compile("http://tech.qq.com.+.htm"))
                for tag in link:
                    yield scrapy.Request(tag['href'],callback=self.parse)
            except:
                pass
            text = ''
            for tag in article:
                try:
                    text += tag.string
                except:
                    pass
            item['content'] = text
            if len(text)>50:
                try:
                    item['title'] = soup.title.string.strip()
                except:
                    try:
                        item['title'] = soup.find_all('div',id="ArticleTit")[0].string.strip()
                    except:
                        return
                if '[' not in item['title'] and ']' not in item['title'] and len(item['title'])>2:
                    yield item

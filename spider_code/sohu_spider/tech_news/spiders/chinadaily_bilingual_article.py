#!/usr/bin/env python
# coding=utf-8

import scrapy
from tech_news.items import BilingualArticleItem
from bs4 import BeautifulSoup
import re

class BilingualSpider(scrapy.Spider):
    name = 'bilingual'
    allowed_domains = ["it.sohu.com"]
    start_urls = ["http://it.sohu.com","http://it.sohu.com/yidonghulian/index.shtml","http://it.sohu.com/tele.shtml","http://it.sohu.com/techchanpin/index.shtml","http://it.sohu.com/shenghuo/index.shtml","http://it.sohu.com/science.shtml","http://it.sohu.com/yejie.shtml","http://it.sohu.com/chuangye/index.shtml"]
    def parse(self,response):
        item = BilingualArticleItem()
        soup = BeautifulSoup(response.body,'lxml',from_encoding='utf-8')
        article = soup.find_all('p',text=True)
        item['link'] = response.url
        if len(article) > 50 :
            try:
                link = soup.find_all(href=re.compile("it.sohu.com.+.shtml"))
                for tag in link:
                    yield scrapy.Request(tag['href'],callback=self.parse)
            except:
                pass
        else:
            try:
                link = soup.find_all(href=re.compile("it.sohu.com.+.shtml"))
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
                item['title'] = soup.title.string.strip()
                #item['title'] = soup.find_all('h1',itemprop="headline")[0].string.strip()
                if '[' not in item['title'] and ']' not in item['title']:
                    yield item

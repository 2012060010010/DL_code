#!/usr/bin/env python
# coding=utf-8

import scrapy
from tech_news.items import BilingualArticleItem
from bs4 import BeautifulSoup
import re

class BilingualSpider(scrapy.Spider):
    name = 'bilingual'
    allowed_domains = ["tech.163.com"]
    start_urls = ["http://tech.163.com","http://tech.163.com/vr/","http://tech.163.com/smart/","http://tech.163.com/internet/"]
    def parse(self,response):
        item = BilingualArticleItem()
        soup = BeautifulSoup(response.body)
        article = soup.find_all('p',text=True)
        item['link'] = response.url


        if len(article) > 50 :
            try:
                link = soup.find_all(href=re.compile("tech.163.com.+"))
                for tag in link:
                    yield scrapy.Request(tag['href'],callback=self.parse)
            except:
                pass
        else:
            try:
                link = soup.find_all(href=re.compile("tech.163.com.+"))
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
                    item['title'] = soup.find_all('h1')[0].string.strip()
                except:
                    return
                if '[' not in item['title'] and ']' not in item['title']:
                    yield item

# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
from  scrapy.exceptions import DropItem
import os

class BilingualArticlePipeline(object):
    def __init__(self):
        self.ids_seen = set()
        self.a = 0
        self.num_dir = 1
        if (os.path.exists('./tech_1')):  # 为测试时不用频繁删除
            os.rmdir('./tech_1')
        self.dir = 'tech_1'
        os.mkdir('./'+self.dir)
    def process_item(self, item, spider):
        if item['link'] in self.ids_seen:#去重
            raise DropItem("Duplicate item found: %s" % item)
        else:
            self.ids_seen.add(item['link'])
        if self.a > 10000:
            self.dir = self.dir[:5]+str(self.num_dir)
            self.num_dir += 1
            self.a = 0
            os.mkdir('./'+self.dir)
        filename = self.dir+'/'+item['title']+'.txt'
        f = open(filename,'w')
        file_content = item['link']+'\n'+item['title']+'\n'+item['content']
        f.write(file_content.encode('utf-8'))
        f.close()
        self.a += 1
        return item

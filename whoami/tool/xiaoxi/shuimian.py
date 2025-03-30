import re
import time
import paho.mqtt.client as mqtt
from difflib import SequenceMatcher

class SleepReportDetector:
    def __init__(self):
        # 报告相关的词语
        self.core_keywords = ["报告", "睡眠报告", "报表", "数据", "统计"]
        # 睡眠质量询问短语
        self.sleep_quality_phrases = [
            "睡得怎么样", "睡得好吗", "睡眠质量", "睡眠情况", 
            "昨晚睡得", "睡了多久", "睡了几个小时", "几点睡的",
            "睡眠时间", "睡了多长时间", "如何睡的", "睡得如何",
            "睡得好不好", "睡眠分析", "睡眠监测", "睡眠状况"
        ]
        # 谐音词
        self.homophone_keywords = ["宝告", "报糕", "保高", "包高", "暴告"]
        # 拼音匹配
        self.pinyin_matches = ["baogao", "shuimianbaogao", "report", "data", "sleep report", "shuimian"]
        
        # 设置模糊匹配阈值
        self.similarity_threshold = 0.7
        
        # 缓存系统
        self.hit_cache = {}
        self.miss_cache = set()
        
        # 编译正则表达式
        self.compile_regex()
        
    def compile_regex(self):
        """编译关键词正则表达式"""
        all_keywords = self.core_keywords + self.homophone_keywords
        pattern_str = '|'.join([re.escape(word) for word in all_keywords])
        self.keyword_pattern = re.compile(pattern_str)
        
        # 编译睡眠质量询问短语的正则表达式
        quality_phrases = '|'.join([re.escape(phrase) for phrase in self.sleep_quality_phrases])
        self.quality_pattern = re.compile(quality_phrases)
        
    def detect(self, text):
        """检测文本是否包含报告相关的关键词"""
        # 标准化文本
        text = text.lower().strip()
        
        # 检查缓存
        if text in self.hit_cache:
            return self.hit_cache[text]
        if text in self.miss_cache:
            return {"matched": False, "keyword": None, "method": "cache"}
        
        # 结果字典
        result = {"matched": False, "keyword": None, "method": None}
        
        # 1. 直接关键词匹配
        if self.keyword_pattern.search(text):
            for keyword in self.core_keywords + self.homophone_keywords:
                if keyword in text:
                    result = {"matched": True, "keyword": keyword, "method": "exact"}
                    break
        
        # 1.5 睡眠质量询问短语匹配
        if not result["matched"]:
            if self.quality_pattern.search(text):
                for phrase in self.sleep_quality_phrases:
                    if phrase in text:
                        result = {"matched": True, "keyword": phrase, "method": "sleep_quality"}
                        break
        
        # 2. 拼音/英文匹配
        if not result["matched"]:
            for pinyin in self.pinyin_matches:
                if pinyin in text:
                    result = {"matched": True, "keyword": pinyin, "method": "pinyin"}
                    break
        
        # 3. 模糊匹配（处理输入错误的情况）
        if not result["matched"]:
            for keyword in self.core_keywords:
                for i in range(len(text) - len(keyword) + 1):
                    substring = text[i:i+len(keyword)]
                    similarity = SequenceMatcher(None, substring, keyword).ratio()
                    if similarity >= self.similarity_threshold:
                        result = {
                            "matched": True, 
                            "keyword": keyword, 
                            "method": "fuzzy",
                            "similarity": similarity
                        }
                        break
        
        # 更新缓存
        if result["matched"]:
            self.hit_cache[text] = result
        else:
            self.miss_cache.add(text)
            
        return result
import re
import time
import json
import numpy as np
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import pypinyin  # 用于中文拼音处理

class TextWakeSystem:
    def __init__(self, config_path=None):
        # 基本配置
        self.core_wake_phrases = ["小熙", "智能助手", "小气", "小区", "小辛", "小节", "小姐", "老师", 
                                 "小爱", "小美", "小跟", "小丽", "美丽"]
        self.extended_wake_phrases = []
        self.greeting_words = ["你好", "您好", "嗨", "喂", "hey", "hi", "hello", "哈喽"]
        self.wake_threshold = 0.75
        self.hit_cache = {}
        self.miss_cache = set()
        self.cache_size = 1000
        self.stats = {"total_calls": 0, "detected": 0, "cache_hits": 0}
        self.user_patterns = defaultdict(int)
        
        # 拼音相关
        self.pinyin_map = {}
        self.homophone_groups = {
            'xiao': ['小', '晓', '笑', '校', '效', '肖', '消', '销', '宵', '萧', '霄', '硝', '哮', '淆', '啸', '枭'],
            'xi': ['习', '系', '席', '西', '息', '希', '惜', '悉', '析', '夕', '牺', '稀', '溪', '熙', '熄', '膝', '昔', '袭', '喜', '洗'],
            'qi': ['气', '起', '期', '七', '其', '奇', '企', '齐', '弃', '妻', '骑', '欺', '戚', '岂', '啟', '祁'],
            'qu': ['区', '去', '取', '趣', '曲', '屈', '驱', '渠', '娶', '蛆', '躯', '衢', '瞿', '觑'],
            'xin': ['辛', '新', '心', '信', '欣', '芯', '薪', '馨', '鑫', '昕', '歆', '忻', '锌'],
            'jie': ['节', '杰', '结', '捷', '接', '街', '姐', '界', '阶', '解', '介', '戒', '劫', '竭', '洁'],
            'jie': ['姐', '节', '杰', '结', '捷', '接', '街', '界', '阶', '解', '介', '戒', '劫', '竭', '洁'],
            'shi': ['师', '士', '市', '示', '适', '式', '视', '试', '释', '室', '世', '事', '饰', '是', '时', '实', '食', '识', '史'],
            'ai': ['爱', '艾', '碍', '隘', '哎', '埃', '癌', '矮', '挨', '哀', '袄'],
            'mei': ['美', '妹', '媒', '梅', '眉', '煤', '每', '魅', '昧', '没', '玫', '枚', '霉'],
            'gen': ['跟', '根', '艮', '亘', '茛', '哏'],
            'li': ['丽', '立', '力', '利', '例', '栗', '粒', '励', '理', '礼', '里', '莉', '俐', '历', '隶', '厉']
        }
        
        # 从配置文件加载
        if config_path:
            self.load_config(config_path)
            
        # 初始化拼音映射
        self._init_pinyin_mapping()
        
        # 预编译的正则表达式
        self._compile_regexes()
    
    def _compile_regexes(self):
        """预编译常用的正则表达式以提高性能"""
        # 基础唤醒词正则
        wake_patterns = '|'.join([re.escape(p) for p in self.core_wake_phrases])
        self.base_pattern = re.compile(rf"({wake_patterns})", re.IGNORECASE)
        
        # 问候语 + 唤醒词组合
        greetings = '|'.join([re.escape(g) for g in self.greeting_words])
        self.greeting_pattern = re.compile(
            rf"({greetings})[,，\s]*({wake_patterns})", 
            re.IGNORECASE
        )
        
    def _init_pinyin_mapping(self):
        """初始化拼音字符映射"""
        # 构建拼音 -> 字符列表的映射
        all_chars = set()
        for group in self.homophone_groups.values():
            all_chars.update(group)
            
        for char in all_chars:
            try:
                py = pypinyin.lazy_pinyin(char)[0]
                if py not in self.pinyin_map:
                    self.pinyin_map[py] = []
                self.pinyin_map[py].append(char)
            except Exception:
                continue
    
    def update_config(self, config_dict):
        """更新配置参数"""
        if 'core_wake_phrases' in config_dict:
            self.core_wake_phrases = config_dict['core_wake_phrases']
        if 'extended_wake_phrases' in config_dict:
            self.extended_wake_phrases = config_dict['extended_wake_phrases']
        if 'greeting_words' in config_dict:
            self.greeting_words = config_dict['greeting_words']
        if 'wake_threshold' in config_dict:
            self.wake_threshold = config_dict['wake_threshold']
        if 'cache_size' in config_dict:
            self.cache_size = config_dict['cache_size']
        if 'homophone_groups' in config_dict:
            self.homophone_groups = config_dict['homophone_groups']
            self._init_pinyin_mapping()
            
        # 重新编译正则表达式
        self._compile_regexes()
        
    def load_config(self, config_path):
        """从文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.update_config(config)
            return True
        except Exception as e:
            print(f"加载配置失败: {str(e)}")
            return False
    
    def save_config(self, config_path):
        """保存当前配置到文件"""
        config = {
            'core_wake_phrases': self.core_wake_phrases,
            'extended_wake_phrases': self.extended_wake_phrases,
            'greeting_words': self.greeting_words,
            'wake_threshold': self.wake_threshold,
            'cache_size': self.cache_size,
            'homophone_groups': self.homophone_groups
        }
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存配置失败: {str(e)}")
            return False
    
    def _check_cache(self, text):
        """检查文本是否在缓存中"""
        if text in self.hit_cache:
            self.stats["cache_hits"] += 1
            return self.hit_cache[text]
        if text in self.miss_cache:
            self.stats["cache_hits"] += 1
            return False
        return None
    
    def _update_cache(self, text, result):
        """更新缓存"""
        if len(self.hit_cache) + len(self.miss_cache) > self.cache_size:
            # 清理缓存
            self.hit_cache = {}
            self.miss_cache = set()
            
        if result["matched"]:
            self.hit_cache[text] = result
        else:
            self.miss_cache.add(text)
    
    def _exact_match(self, text):
        """精确匹配核心唤醒词"""
        # 基础匹配模式
        if self.base_pattern.search(text):
            return {"matched": True, "method": "exact", "phrase": "core_wake_word"}
        
        # 检查问候语+唤醒词组合
        greeting_match = self.greeting_pattern.search(text)
        if greeting_match:
            return {"matched": True, "method": "exact", "phrase": "greeting_wake_word"}
        
        # 检查扩展唤醒词
        for phrase in self.extended_wake_phrases:
            if phrase in text:
                return {"matched": True, "method": "exact", "phrase": "extended_wake_word"}
                
        return {"matched": False}
    
    def _fuzzy_match(self, text):
        """模糊匹配，容忍轻微的拼写错误"""
        for phrase in self.core_wake_phrases + self.extended_wake_phrases:
            for i in range(len(text) - len(phrase) + 1):
                substring = text[i:i+len(phrase)]
                similarity = SequenceMatcher(None, substring, phrase).ratio()
                if similarity >= self.wake_threshold:
                    return {
                        "matched": True, 
                        "method": "fuzzy", 
                        "phrase": phrase,
                        "similarity": similarity
                    }
        return {"matched": False}
    
    def _pinyin_match(self, text):
        """基于拼音的匹配，捕获同音字"""
        try:
            text_pinyin = pypinyin.lazy_pinyin(text)
            
            # 检查所有可能的拼音组合
            pinyin_combinations = [
                ("xiao", "xi", "xiao_xi"),
                ("xiao", "qi", "xiao_qi"),
                ("xiao", "qu", "xiao_qu"),
                ("xiao", "xin", "xiao_xin"),
                ("xiao", "jie", "xiao_jie"),
                ("xiao", "mei", "xiao_mei"),
                ("xiao", "gen", "xiao_gen"),
                ("xiao", "li", "xiao_li"),
                ("xiao", "ai", "xiao_ai"),
                ("lao", "shi", "lao_shi"),
                ("mei", "li", "mei_li")
            ]
            
            for i in range(len(text_pinyin) - 1):
                # 检查所有拼音组合
                for first, second, phrase_name in pinyin_combinations:
                    if text_pinyin[i] == first and text_pinyin[i+1] == second:
                        return {
                            "matched": True, 
                            "method": "pinyin", 
                            "phrase": phrase_name,
                            "position": i
                        }
                
                # 同时检查纯拼音输入
                if i < len(text.split()) - 1:
                    words = text.split()
                    for first, second, phrase_name in pinyin_combinations:
                        if words[i].lower() == first and words[i+1].lower() == second:
                            return {
                                "matched": True,
                                "method": "pinyin",
                                "phrase": f"{phrase_name}_raw",
                                "position": i
                            }
                    
            return {"matched": False}
        except Exception as e:
            print(f"拼音匹配出错: {str(e)}")
            return {"matched": False}
    
    def _update_user_patterns(self, text, result):
        """记录用户交互模式以进行自适应学习"""
        if result["matched"]:
            # 如果是成功匹配，记录匹配上下文
            words = text.split()
            if len(words) > 3:
                # 记录上下文模式
                pattern = ' '.join(words[:2]) + " ... " + ' '.join(words[-2:])
                self.user_patterns[pattern] += 1
    
    def detect(self, text):
        """
        检测文本是否包含唤醒词
        返回: 包含匹配信息的字典
        """
        start_time = time.time()
        self.stats["total_calls"] += 1
        
        # 标准化处理
        text = text.lower().strip()
        
        # 检查缓存
        cached_result = self._check_cache(text)
        if cached_result is not None:
            return cached_result
        
        # 多级检测
        result = {"matched": False, "detection_time": 0}
        
        # 1. 快速精确匹配 (最高效)
        exact_result = self._exact_match(text)
        if exact_result["matched"]:
            result = exact_result
        else:
            # 2. 拼音匹配 (针对中文)
            pinyin_result = self._pinyin_match(text)
            if pinyin_result["matched"]:
                result = pinyin_result
            else:
                # 3. 模糊匹配 (容错能力)
                fuzzy_result = self._fuzzy_match(text)
                if fuzzy_result["matched"]:
                    result = fuzzy_result
        
        # 记录检测时间
        result["detection_time"] = (time.time() - start_time) * 1000  # 毫秒
        
        # 更新统计
        if result["matched"]:
            self.stats["detected"] += 1
            
        # 更新缓存
        self._update_cache(text, result)
        
        # 更新用户模式
        self._update_user_patterns(text, result)
        
        return result
    
    def learn_from_feedback(self, text, is_correct):
        """基于用户反馈学习改进"""
        if not is_correct and text not in self.miss_cache:
            # 误判为唤醒词，移除相关缓存
            if text in self.hit_cache:
                del self.hit_cache[text]
                
            # 可以添加到特定的排除列表中
            self.miss_cache.add(text)
        
        elif is_correct and text not in self.hit_cache:
            # 漏判，应该是唤醒词但没检测到
            # 分析这个文本，找出可能的新模式
            words = text.split()
            for i in range(len(words)):
                for j in range(i+1, min(i+4, len(words))):
                    candidate = ' '.join(words[i:j])
                    # 如果与现有唤醒词相似，考虑添加到扩展列表
                    for phrase in self.core_wake_phrases:
                        if SequenceMatcher(None, candidate, phrase).ratio() > 0.6:
                            if candidate not in self.extended_wake_phrases:
                                self.extended_wake_phrases.append(candidate)
                                
            # 更新缓存
            self.hit_cache[text] = {"matched": True, "method": "learned", "phrase": "feedback"}
            
            # 重新编译正则表达式
            self._compile_regexes()
    
    def get_stats(self):
        """获取系统统计信息"""
        stats = self.stats.copy()
        
        # 计算命中率
        if stats["total_calls"] > 0:
            stats["hit_ratio"] = stats["detected"] / stats["total_calls"]
            stats["cache_hit_ratio"] = stats["cache_hits"] / stats["total_calls"]
        else:
            stats["hit_ratio"] = 0
            stats["cache_hit_ratio"] = 0
            
        # 获取常见用户模式
        if self.user_patterns:
            stats["common_patterns"] = dict(Counter(self.user_patterns).most_common(5))
            
        return stats

if __name__ == '__main__':
    
    # 创建唤醒系统实例
    wake_system = TextWakeSystem()
    
    # 添加额外的唤醒词
    wake_system.update_config({
        "extended_wake_phrases": ["智能小助手", "AI助理", "语音助手"]
    })
    
    # 测试样例
    test_cases = [
        "你好小熙，今天天气怎么样？",
        "嗨，智能助手，能帮我查一下明天的会议吗？",
        "晓熙，播放一首周杰伦的歌",
        "我想请问一下小企业的贷款政策",
        "告诉我智能小助手能做什么",
        "效芝麻开门",
        "小熙，现在几点了",
        "笑熙真好玩",
        "你好，xiao zhi",
        "你好，xiao xi",
        "这是一个普通的句子，没有唤醒词",
        "您好，小气",
        "您好，小区。"
    ]
    
    print("=== 唤醒系统测试 ===")
    for i, text in enumerate(test_cases):
        result = wake_system.detect(text)
        is_wake = "✓" if result["matched"] else "✗"
        print(f"{i+1}. [{is_wake}] {text}")
        if result["matched"]:
            print(f"   方法: {result['method']}, 耗时: {result['detection_time']:.2f}ms")
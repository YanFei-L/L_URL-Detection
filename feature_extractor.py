import urllib.parse
import os
import re
import math

class FeatureExtractor:
    def __init__(self):
        pass

    # ==========================================
    # 第一类：长度与结构特征 (Structural & Length Features)
    # ==========================================
    
    def get_url_length(self, url):
        """URL总长度"""
        if not isinstance(url, str):
            return 0
        return len(url)

    def get_hostname_length(self, url):
        """域名长度"""
        if not isinstance(url, str):
            return 0
        try:
            parsed = urllib.parse.urlparse(url)
            return len(parsed.netloc)
        except:
            return 0

    def get_path_length(self, url):
        """路径长度"""
        if not isinstance(url, str):
            return 0
        try:
            parsed = urllib.parse.urlparse(url)
            return len(parsed.path)
        except:
            return 0

    def get_dir_depth(self, url):
        """目录深度 (路径中 / 的出现次数)"""
        if not isinstance(url, str):
            return 0
        try:
            parsed = urllib.parse.urlparse(url)
            path = parsed.path
            return path.count('/')
        except:
            return 0

    def get_filename_length(self, url):
        """文件名长度"""
        if not isinstance(url, str):
            return 0
        try:
            parsed = urllib.parse.urlparse(url)
            path = parsed.path
            # 如果路径以/结尾，文件名长度可能认为是0或者取上一级，这里按通常理解，取最后一个segment
            if path.endswith('/'):
                path = path[:-1]
            if not path:
                return 0
            filename = os.path.basename(path)
            return len(filename)
        except:
            return 0

    def extract_structural_features(self, url):
        """提取所有结构特征"""
        return {
            'url_length': self.get_url_length(url),
            'hostname_length': self.get_hostname_length(url),
            'path_length': self.get_path_length(url),
            'dir_depth': self.get_dir_depth(url),
            'filename_length': self.get_filename_length(url)
        }

    # ==========================================
    # 第二类：特殊符号统计特征 (Special Character Frequency)
    # ==========================================

    def get_count_dots(self, url):
        """点号数量"""
        if not isinstance(url, str):
            return 0
        return url.count('.')

    def get_count_hyphens(self, url):
        """连字符数量"""
        if not isinstance(url, str):
            return 0
        return url.count('-')

    def get_has_at_symbol(self, url):
        """@ 符号是否存在"""
        if not isinstance(url, str):
            return 0
        return 1 if '@' in url else 0

    def get_double_slash_position(self, url):
        """双斜杠位置 (最后一次出现的位置)"""
        if not isinstance(url, str):
            return -1
        return url.rfind('//')

    def get_count_query_params(self, url):
        """查询参数数量"""
        if not isinstance(url, str):
            return 0
        try:
            parsed = urllib.parse.urlparse(url)
            if not parsed.query:
                return 0
            return len(urllib.parse.parse_qs(parsed.query))
        except:
            return 0

    def extract_special_char_features(self, url):
        """提取所有特殊符号特征"""
        return {
            'count_dots': self.get_count_dots(url),
            'count_hyphens': self.get_count_hyphens(url),
            'has_at_symbol': self.get_has_at_symbol(url),
            'double_slash_position': self.get_double_slash_position(url),
            'count_query_params': self.get_count_query_params(url)
        }

    # ==========================================
    # 第三类：异常与混淆特征 (Abnormality & Obfuscation)
    # ==========================================

    def get_is_ip_address(self, url):
        """是否使用 IP 地址"""
        if not isinstance(url, str):
            return 0
        try:
            parsed = urllib.parse.urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0
            # Simple check for IPv4
            parts = hostname.split('.')
            if len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts):
                return 1
            # Check for IPv6 (simple check)
            if ':' in hostname:
                return 1
            return 0
        except:
            return 0

    def get_is_shortened(self, url):
        """是否使用短链接服务"""
        if not isinstance(url, str):
            return 0
        shorteners = {
            'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co', 
            'is.gd', 'buff.ly', 'bdf.ly', 'bit.do', 'tr.im',
            'forms.gle', 'rb.gy'
        }
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www.
            if domain.startswith('www.'):
                domain = domain[4:]
            return 1 if domain in shorteners else 0
        except:
            return 0

    def get_https_in_hostname(self, url):
        """HTTPS 敏感性 (Hostname中包含 http/https)"""
        if not isinstance(url, str):
            return 0
        try:
            parsed = urllib.parse.urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0
            if 'http' in hostname.lower():
                return 1
            return 0
        except:
            return 0

    def get_digit_letter_ratio(self, url):
        """数字占比"""
        if not isinstance(url, str) or len(url) == 0:
            return 0.0
        digits = sum(c.isdigit() for c in url)
        return digits / len(url)

    def extract_abnormality_features(self, url):
        """提取所有异常与混淆特征"""
        return {
            'is_ip_address': self.get_is_ip_address(url),
            'is_shortened': self.get_is_shortened(url),
            'https_in_hostname': self.get_https_in_hostname(url),
            'digit_letter_ratio': self.get_digit_letter_ratio(url)
        }

    # ==========================================
    # 第四类：语义与熵值特征 (Lexical & Entropy)
    # ==========================================

    def get_hostname_entropy(self, url):
        """主机名信息熵"""
        if not isinstance(url, str):
            return 0.0
        try:
            parsed = urllib.parse.urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0.0
            
            # Calculate Shannon Entropy
            prob = [float(hostname.count(c)) / len(hostname) for c in dict.fromkeys(list(hostname))]
            entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
            return entropy
        except:
            return 0.0

    def get_sensitive_word_count(self, url):
        """敏感词计数"""
        if not isinstance(url, str):
            return 0
        sensitive_words = ['confirm', 'account', 'banking', 'secure', 'login', 'verify']
        url_lower = url.lower()
        count = 0
        for word in sensitive_words:
            if word in url_lower:
                count += 1
        return count

    def get_tld_risk(self, url):
        """顶级域名风险度 (.xyz, .top, .club, .info)"""
        if not isinstance(url, str):
            return 0
        risky_tlds = {'.xyz', '.top', '.club', '.info'}
        try:
            parsed = urllib.parse.urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0
            # Check if hostname ends with any risky tld
            for tld in risky_tlds:
                if hostname.endswith(tld):
                    return 1
            return 0
        except:
            return 0

    def get_longest_token_length(self, url):
        """最长词元长度"""
        if not isinstance(url, str):
            return 0
        try:
            parsed = urllib.parse.urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0
            # Split by . and -
            tokens = re.split(r'[.-]', hostname)
            if not tokens:
                return 0
            return max(len(token) for token in tokens)
        except:
            return 0

    def extract_lexical_features(self, url):
        """提取所有语义与熵值特征"""
        return {
            'hostname_entropy': self.get_hostname_entropy(url),
            'sensitive_word_count': self.get_sensitive_word_count(url),
            'tld_risk': self.get_tld_risk(url),
            'longest_token_length': self.get_longest_token_length(url)
        }
    
    def extract_all_features(self, url):
        """提取所有特征 (汇总)"""
        features = {}
        features.update(self.extract_structural_features(url))
        features.update(self.extract_special_char_features(url))
        features.update(self.extract_abnormality_features(url))
        features.update(self.extract_lexical_features(url))
        return features

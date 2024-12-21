import traceback
import os
import shutil
# from simhash import Simhash, SimhashIndex
import re
import yaml

from whoami.utils.log import Logger
logger = Logger('Utils')

class Utils:
    """Utils class what aims to code some generation tools what can be used in all tool, agent or other function.
    """
    def __init__(self) -> None:
        pass
        
    def get_error_info(self, error_info: str, e: Exception):
        """get the error information that involved the error code line and reason.

        Args:
            error_info (str): the error information that you want to raise.
            e (Exception): the error reason.

        Returns:
            _type_: error infomation.
        """
        error_info = traceback.format_exc()
        error = f"{error_info}{str(e)}！\n{error_info}"
        return error

    def init_directory(self, directory: str, delete_flag: int = 0):
        """_summary_

        Args:
            directory (str): the directory path.
            delete_flag (int, optional): whether delete all the files in the exist directory. Defaults to 0.

        Returns:
            _type_: (bool, error_info/success_info)
        """
        try:
            if os.path.exists(directory) and delete_flag == 1:
                shutil.rmtree(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)
            return True, f"success to init the directory: {directory}！"
        except Exception as e:
            error_info = f"fail to init the directory: {directory}\n{str(e)}！\n{traceback.format_exc()}"
            logger.error(error_info)
            return False, error_info
    
    def get_files_based_extension(self, directory, file_extension: str):
        """list all the file with the file_extension, no recursive

        Args:
            directory (_type_): _description_
            file_extension (str): file extension just like '.txt'

        Returns:
            _type_: (bool, error_info/list)
        """
        try:
            txt_files = []
            for file in os.listdir(directory):
                if file.endswith(file_extension):
                    txt_files.append(os.path.join(directory, file))
        except Exception as e:
            error_info = self.get_error_info(f"fail to get the extention: {file_extension} file！", e)
            logger.error(error_info)
            return False, error_info
        return True, txt_files

    def count_chinese_characters(self, text):
        try:
            chinese_char_pattern = r'[\u4e00-\u9fff]'
            chinese_chars = re.findall(chinese_char_pattern, text)
        except Exception as e:
            error_info = self.get_error_info("fail to count chinese characters!", e)
            logger.error(error_info)
            return False, error_info
        return True, len(chinese_chars)

    def count_english_words(self, text):
        try:
            words = re.findall(r'\b\w+\b', text)
        except Exception as e:
            error_info = self.get_error_info("fail to count english characters!", e)
            logger.error(error_info)
            return False, error_info
        return True, len(words)

    def read_yaml(self, yaml_file: str):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
        except Exception as e:
            raise ValueError('fail to load yaml file!') from e
        return config
    
    # def remove_duplicate_simhash(self, contents: list[str]) -> list[str]:
    #     """remove duplicate paragraph used simhash

    #     Args:
    #         contents (list[str]): the paragraph list.
    #     """
    #     def get_features(s):
    #         width = 3
    #         s = s.lower()
    #         s = re.sub(r'[^\w]+', '', s)
    #         return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]
        
    #     try:
    #         index = SimhashIndex([], k=3)
    #         unique_contents = []
    #         for content in contents:
    #             simhash_value = Simhash(get_features(content))
    #             if not index.get_near_dups(simhash_value):
    #                 unique_contents.append(content)
    #                 index.add(content, simhash_value)
    #     except Exception as e:
    #         return False, self.get_error_info("fail to remove duplicate!", e)
    #     return True, unique_contents
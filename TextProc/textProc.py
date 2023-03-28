# Text processing for English, Hindi and Tamil
from num_to_words import num_to_word
import re

class TextNormalizer:
    def __init__(self):
        '''
        Initialize some maps
        '''
        self.lang_map = {"english":"en", "English":"en", "tamil":"ta", "Tamil":"ta", "en":"en", "ta":"ta"}
        self.point_map = {"en":"point", "ta":"போயிண்ட்"}
    
    def replace_currency(self, string):
        '''
        '''
        string = re.sub(r"(?<=\d)Rs\.", "Rupees ", string)
        string = re.sub(r"(?<=\d)Rs", "Rupees ", string)
        string = re.sub(r"(?<=\d)Re\.", "Rupee ", string)
        string = re.sub(r"(?<=\d)Re", "Rupee ", string)
        string = re.sub(r"Rs\.(?=\d)", "Rupees ", string)
        string = re.sub(r"Rs(?=\d)", "Rupees ", string)
        string = re.sub(r"Re\.(?=\d)", "Rupee ", string)
        string = re.sub(r"Re(?=\d)", "Rupee ", string)
        string = re.sub(r"(?<=\s)Rs\.", "Rupees ", string)
        string = re.sub(r"(?<=\s)Rs", "Rupees ", string)
        string = re.sub(r"(?<=\s)Re\.", "Rupee ", string)
        string = re.sub(r"(?<=\s)Re", "Rupee ", string)
        return string
    
    def __is_float(self, string):
        '''
        '''
        parts = string.split('.')
        if len(parts) != 2:
            return False
        return parts[0].isdecimal() and parts[1].isdecimal()
    
    def number_to_words(self, string, language):
        '''
        '''
        try:
            language = self.lang_map[language]
            string = string.replace('/-',' ') # Remove trailing /- from numbers like 1000/- to 1000
            string = self.replace_currency(string)
            string = re.sub(r'(?<=\d),(?=\d)', '', string) # Remove commas between numbers in a string
            numbers = re.findall(r'\d+(?:\.\d+)?', string)
            num_dict = {}
            for number in numbers:
                if number.isnumeric():
                    num_str = num_to_word(number, lang=language).replace('-',' ')
                    num_dict[number] = num_str
                elif self.__is_float(number):
                    parts = number.split('.')
                    num_str = num_to_word(parts[0], lang=language).replace('-',' ')
                    if language in self.point_map.keys():
                        num_str += ' '+self.point_map[language]
                    else:
                        print(f"Language without point string defined")
                        return (False, f"Language without point string defined")
                    for i in range(0,len(parts[1])):
                        num_str += ' '+num_to_word(parts[1][i], lang=language)
                    num_dict[number] = num_str
                else:
                    print(f"Unknown type for number: {number}")
                    return (False, f"Unknown type for number: {number}")
            sorted_dict = dict(sorted(num_dict.items(), key=lambda x: len(x[0]), reverse=True))
            for key, value in sorted_dict.items():
                string = string.replace(key, value)
            string = re.sub(" +", " ", string.strip())
            return (True, string)
        except Exception as e:
            return (False, e)
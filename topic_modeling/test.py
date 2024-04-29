import re

str = "[url=store.steampowered.com] 웹사이트 [ah1]링크 [/url]"
str = re.sub(r'\[(h1|h2|h3|b|u|i|strike|spoiler|hr|noparse)]', '', str)
str = re.sub(r'\[/(h1|h2|h3|b|u|i|strike|spoiler|hr|noparse)]', '', str)
str = re.sub(r'\[url=[^\]]*\]|\[/url\]', '', str)

print(str)

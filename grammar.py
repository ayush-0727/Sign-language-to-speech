# First, install the LanguageTool Python wrapper
# pip install language-tool-python

import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

text = "New Delhi capital India"
matches = tool.check(text)

corrected_text = language_tool_python.utils.correct(text, matches)

print(corrected_text)

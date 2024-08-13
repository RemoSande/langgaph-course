
""" 
The __init__.py file with the content you've shown serves several important purposes in 
Python package structuring:

- Package Initialization: The presence of __init__.py tells Python that the directory should be treated as a package.

- Simplified Imports: It allows users to import from the package more conveniently. 
Instead of single imports users can import multiple modules from the package in a single line.
Users can now do:
pythonCopyfrom graph.nodes import generate, grade_documents

"""

from graph.nodes.generate import generate
from graph.nodes.grade_documents import grade_documents
from graph.nodes.retrieve import retrieve
from graph.nodes.web_search import web_search

__all__ = ["generate", "grade_documents", "retrieve", "web_search"]

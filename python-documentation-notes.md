# PYTHON DOCUMENTATION GUIDE (NOTES)
## IMPORT STATEMENTS
###### The import statement has two problems:

- Long import statements can be difficult to write, requiring various contortions to fit Pythonic style guidelines.
- Imports can be ambiguous in the face of packages; within a package, it's not clear whether import foo refers to a module within the package or some module outside the package. (More precisely, a local module or package can shadow another hanging directly off sys.path.)

For the first problem, it is proposed that parentheses be permitted to enclose multiple names, thus allowing Python's standard mechanisms for multi-line values to apply. For the second problem, it is proposed that all import statements be absolute by default (searching sys.path only) with special syntax (leading dots) for accessing package-relative imports.

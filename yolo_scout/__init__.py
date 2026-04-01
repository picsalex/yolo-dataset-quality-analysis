import warnings

warnings.filterwarnings(
    "ignore", message=r"invalid escape sequence.*", category=SyntaxWarning, module=r"glob2\.fnmatch"
)
warnings.filterwarnings("ignore", message="QuickGELU mismatch.*", category=UserWarning, module=r"open_clip\.factory")
